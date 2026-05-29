import asyncio
import base64
import glob
import io
import json
import os
import subprocess
import time
from typing import List

import httpx
from fastapi import UploadFile
from PIL import Image

from config import BASE_DIR, GEMINI_URL, OUTPUT_DIR, PRELOAD_MANUAL_STEPS, RUNPOD_INFERENCE_BASE_URL
from state import save_session, state
from vlm import REGISTERED_STEP_IDS, WARMED_STEP_IDS, preload_steps_to_gpu

_preview_steps: list = []
_preview_updated: bool = False
_analysis_start_time: float = 0.0


def reset_preview_state() -> None:
    global _preview_steps, _preview_updated
    _preview_steps = []
    _preview_updated = False


def get_elapsed_time() -> float:
    if state["progress_step"] not in ("upload", "done") and _analysis_start_time > 0:
        return round(time.time() - _analysis_start_time, 1)
    return state["analysis_time"]


def extract_gemini_steps(response_json: dict) -> list:
    text = response_json["candidates"][0]["content"]["parts"][0]["text"]
    payload = json.loads(text)

    if isinstance(payload, dict):
        raw_steps = payload.get("steps", [])
    elif isinstance(payload, list):
        raw_steps = payload
    else:
        raw_steps = []

    if isinstance(raw_steps, dict):
        raw_steps = [raw_steps]
    if not isinstance(raw_steps, list):
        return []

    normalized = []
    for item in raw_steps:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, list):
            normalized.extend(x for x in item if isinstance(x, dict))
    return normalized


async def preview_event_stream():
    last_count = 0
    idle_ticks = 0

    for _ in range(60):
        if state["progress_step"] in ("render", "analyze"):
            break
        await asyncio.sleep(0.5)

    while True:
        await asyncio.sleep(0.4)
        if len(_preview_steps) > last_count:
            new_steps = _preview_steps[last_count:]
            last_count = len(_preview_steps)
            data = json.dumps(new_steps, ensure_ascii=False)
            yield f"data: {data}\n\n"
            idle_ticks = 0
        else:
            idle_ticks += 1

        if state["progress_step"] == "done" and idle_ticks >= 5:
            yield "data: __done__\n\n"
            break


async def analyze_pdf_page(client, page_img_path: str, page_num: int, job_dir: str) -> list:
    try:
        with Image.open(page_img_path) as img:
            orig_w, orig_h = img.size
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=88)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = """당신은 조립 매뉴얼 디지털화 전문가입니다.
이 이미지는 조립 매뉴얼의 한 페이지입니다.
모든 STEP을 찾아 JSON으로 반환하세요.
- step_number: STEP 번호(정수)
- title: 이미지에 "STEP N" 레이블이 있으면 그대로. 없으면 반드시 "STEP N" 형식으로만 작성 (N=step_number). 이미지 내용 설명 절대 금지.
- desc: 이미지 바로 아래 지시문을 한 글자도 빠짐없이 복사. 한국어 우선, 없으면 영문 그대로. 요약/해석 금지. 문장 내 줄바꿈 금지.
- box_2d: 이미지 영역만 [ymin,xmin,ymax,xmax] 0~1000 스케일
규칙: Teaching STEAM 로고/브랜드 무시. desc 없으면 제외.
{"steps":[{"step_number":int,"title":"STEP N","desc":"원문","box_2d":[int,int,int,int]}]}"""

        res = await client.post(
            GEMINI_URL,
            json={
                "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            },
            timeout=60.0,
        )
        if res.status_code != 200:
            return []

        raw_steps = extract_gemini_steps(res.json())
        steps = []
        with Image.open(page_img_path) as full_img:
            for s in raw_steps:
                box = s.get("box_2d")
                desc = s.get("desc", "").strip()
                if not box or not desc:
                    continue
                ymin, xmin, ymax, xmax = box
                left = max(0, int(xmin * orig_w / 1000))
                top = max(0, int(ymin * orig_h / 1000))
                right = min(orig_w, int(xmax * orig_w / 1000))
                bottom = min(orig_h, int(ymax * orig_h / 1000))
                if right <= left or bottom <= top:
                    continue

                step_num = s.get("step_number", 0)
                crop_path = os.path.join(job_dir, f"step_p{page_num}_{step_num}.jpg")
                full_img.crop((left, top, right, bottom)).convert("RGB").save(crop_path, "JPEG", quality=92)
                step_data = {
                    "step": step_num,
                    "title": f"STEP {step_num}",
                    "desc": desc,
                    "image_url": f"/outputs/{os.path.relpath(crop_path, OUTPUT_DIR)}".replace("\\", "/"),
                }
                steps.append(step_data)

                global _preview_steps, _preview_updated
                _preview_steps.append(step_data)
                _preview_updated = True
                existing_urls = {item["image_url"] for item in state["manual_steps"]}
                if step_data["image_url"] not in existing_urls:
                    state["manual_steps"] = state["manual_steps"] + [step_data]
                    state["file_info"]["steps"] = len(state["manual_steps"])
                print(f"  ✅ STEP {step_num}: {desc[:40]}...")
        return steps
    except Exception as e:
        print(f"🚨 페이지 분석 에러 (page {page_num}): {e}")
        return []


async def detect_and_crop_image(client, img_path, base_idx):
    try:
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size
            img_rgb = img.convert("RGB")
            img_rgb.thumbnail((1600, 1600))
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=90)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = """조립/공예 매뉴얼 이미지에서 모든 STEP을 탐지하세요.
- step_number: 정수
- title: 이미지에 레이블이 있으면 그대로, 없으면 "STEP N" 형식으로만 (이미지 내용 설명 절대 금지)
- desc: 텍스트 있으면 원문, 없으면 시각적 동작 한국어 설명
- box_2d: [ymin,xmin,ymax,xmax] 0~1000
{"steps":[{"step_number":int,"title":str,"desc":str,"box_2d":[int,int,int,int]}]}"""

        res = await client.post(
            GEMINI_URL,
            json={
                "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}],
                "generationConfig": {"responseMimeType": "application/json"},
            },
            timeout=60.0,
        )
        if res.status_code != 200:
            return []

        raw_steps = extract_gemini_steps(res.json())
        job_dir = os.path.dirname(img_path)
        steps = []
        with Image.open(img_path) as full_img:
            orig_w, orig_h = full_img.size
            for s in raw_steps:
                box = s.get("box_2d")
                desc = s.get("desc", "").strip()
                if not box or not desc:
                    continue
                ymin, xmin, ymax, xmax = box
                left = max(0, int(xmin * orig_w / 1000))
                top = max(0, int(ymin * orig_h / 1000))
                right = min(orig_w, int(xmax * orig_w / 1000))
                bottom = min(orig_h, int(ymax * orig_h / 1000))
                if right <= left or bottom <= top:
                    continue
                step_num = s.get("step_number") or (base_idx + len(steps) + 1)
                crop_path = os.path.join(job_dir, f"img_step_{step_num}_{int(time.time() * 1000)}.jpg")
                full_img.crop((left, top, right, bottom)).convert("RGB").save(crop_path, "JPEG", quality=92)
                steps.append(
                    {
                        "step": step_num,
                        "title": s.get("title", f"STEP {step_num}"),
                        "desc": desc,
                        "image_url": f"/outputs/{os.path.relpath(crop_path, OUTPUT_DIR)}".replace("\\", "/"),
                    }
                )
        return steps
    except Exception as e:
        print(f"🚨 이미지 분석 에러: {e}")
    return []


async def process_manual_files(files: List[UploadFile]):
    global _analysis_start_time, _preview_steps, _preview_updated
    start_time = time.time()
    _analysis_start_time = start_time
    _preview_steps = []
    _preview_updated = False
    state["file_info"] = {"name": "", "pages": 0, "steps": 0}
    state.update(
        {
            "is_analyzed": False,
            "step_locked": False,
            "progress_step": "upload",
            "ai_result": "WAIT",
            "analysis_time": 0.0,
        }
    )
    job_dir = os.path.join(OUTPUT_DIR, f"sess_{int(start_time)}")
    os.makedirs(job_dir, exist_ok=True)
    all_steps = []

    async with httpx.AsyncClient() as client:
        for upload in files:
            file_path = os.path.join(job_dir, upload.filename)
            with open(file_path, "wb") as out:
                out.write(await upload.read())
            ext = upload.filename.lower()
            state["file_info"]["name"] = upload.filename
            state["uploaded_preview"] = ""

            if ext.endswith(".pdf"):
                subprocess.run(
                    ["pdftoppm", "-jpeg", "-r", "72", "-f", "1", "-l", "1", file_path, os.path.join(job_dir, "thumb")],
                    capture_output=True,
                )
                thumb_files = sorted(glob.glob(os.path.join(job_dir, "thumb-*.jpg")))
                if thumb_files:
                    state["uploaded_preview"] = f"/outputs/{os.path.relpath(thumb_files[0], OUTPUT_DIR)}".replace("\\", "/")
            elif ext.endswith((".png", ".jpg", ".jpeg")):
                state["uploaded_preview"] = f"/outputs/{os.path.relpath(file_path, OUTPUT_DIR)}".replace("\\", "/")

            if ext.endswith(".pdf"):
                pages_dir = os.path.join(job_dir, "pages")
                os.makedirs(pages_dir, exist_ok=True)
                state["progress_step"] = "render"
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["pdftoppm", "-jpeg", "-r", "200", file_path, os.path.join(pages_dir, "page")],
                        capture_output=True,
                    ),
                )
                page_imgs = sorted(glob.glob(os.path.join(pages_dir, "page-*.jpg")))
                state["progress_step"] = "analyze"
                results = await asyncio.gather(
                    *[analyze_pdf_page(client, page_path, i + 1, job_dir) for i, page_path in enumerate(page_imgs)]
                )
                for result in results:
                    all_steps.extend(result)
            elif ext.endswith((".png", ".jpg", ".jpeg")):
                all_steps.extend(await detect_and_crop_image(client, file_path, len(all_steps)))

    unique, seen = [], set()
    for step in all_steps:
        if step["image_url"] not in seen:
            unique.append(step)
            seen.add(step["image_url"])

    def step_num(item):
        value = item.get("step")
        return int(value) if value and str(value).isdigit() else 999

    unique.sort(key=step_num)

    filtered = []
    for i, step in enumerate(unique):
        is_boundary = i == 0 or i == len(unique) - 1
        title = step.get("title", "").lower()
        desc = step.get("desc", "").lower()
        if is_boundary and any(keyword in title or keyword in desc for keyword in ["안내", "steam", "teaching", "copyright"]):
            if step_num(step) in (0, 999):
                continue
        filtered.append(step)

    with open(os.path.join(job_dir, "instruction.json"), "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=4)

    state.update(
        {
            "manual_steps": filtered,
            "is_analyzed": True,
            "analysis_time": round(time.time() - start_time, 2),
            "current_step_idx": 0,
            "pending_step_idx": None,
            "pass_hold_until": 0.0,
            "pass_transition_id": 0,
            "progress_step": "done",
        }
    )
    save_session()
    REGISTERED_STEP_IDS.clear()
    WARMED_STEP_IDS.clear()

    if PRELOAD_MANUAL_STEPS and RUNPOD_INFERENCE_BASE_URL and filtered:
        asyncio.create_task(preload_steps_to_gpu(filtered))
        print(f"🚀 [PRELOAD] 백그라운드 등록 시작 — {len(filtered)}개 STEP")

    print(f"✅ [SUCCESS] 분석 완료 — {len(filtered)}개 STEP, {state['analysis_time']}s")
    return {"status": "success", "steps": filtered}
