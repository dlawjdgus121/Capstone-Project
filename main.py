import io
import os
import json
import time
import base64
import asyncio
import httpx
import uvicorn
import glob
import re
import subprocess
import cv2
import numpy as np
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from dotenv import load_dotenv
from livekit import rtc

# 1. 환경 변수 로드
load_dotenv()

try:
    import opendataloader_pdf
except ImportError:
    print("⚠️ [DEBUG] 'pip install opendataloader-pdf' 라이브러리가 필요합니다.")

# ==========================================
# 🌟 환경 변수 및 설정
# ==========================================
API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

raw_runpod_url = os.getenv("RUNPOD_INFERENCE_URL", "").strip()
if raw_runpod_url:
    if not raw_runpod_url.startswith("http"): raw_runpod_url = f"http://{raw_runpod_url}"
    if not raw_runpod_url.endswith("/predict"): raw_runpod_url = raw_runpod_url.rstrip("/") + "/predict"
    RUNPOD_INFERENCE_URL = raw_runpod_url
else:
    RUNPOD_INFERENCE_URL = None

LIVEKIT_URL = "wss://capstone-project-jvy5e1z6.livekit.cloud"
LIVEKIT_TOKEN = os.getenv("LIVEKIT_TOKEN")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- [시스템 상태 관리] ---
state = {
    "latest_frame": None,
    "latest_frame_raw": None,
    "manual_steps": [],
    "current_step_idx": 0,
    "ai_response": "대기 중...",
    "is_analyzed": False,
    "analysis_time": 0.0,
    "is_coaching_active": False,
    "step_locked": False,
    "progress_step": "upload",
    # WebRTC 추가 지표
    "client_send_time": 0.0,
    "last_rtt": 0.0,
    "current_fps": 0.0,
    "log": "모바일 연결 대기 중..."
}

prev_frame_time = 0.0
SESSION_FILE = os.path.join(OUTPUT_DIR, "last_session.json")

# ─── 세션 저장/복원 ──────────────────────────────────────────────────────────
def save_session():
    """현재 단계 인덱스와 매뉴얼 경로를 디스크에 저장."""
    if not state["is_analyzed"] or not state["manual_steps"]:
        return
    try:
        payload = {
            "current_step_idx": state["current_step_idx"],
            "manual_steps":     state["manual_steps"],
            "analysis_time":    state["analysis_time"],
            "step_locked":      state["step_locked"],
        }
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 세션 저장 실패: {e}")

def load_session() -> bool:
    """서버 시작 시 이전 세션을 복원. 성공하면 True 반환."""
    if not os.path.exists(SESSION_FILE):
        return False
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        steps = payload.get("manual_steps", [])
        if not steps:
            return False
        idx = int(payload.get("current_step_idx", 0))
        idx = max(0, min(idx, len(steps) - 1))
        state.update({
            "manual_steps":     steps,
            "current_step_idx": idx,
            "analysis_time":    payload.get("analysis_time", 0),
            "step_locked":      payload.get("step_locked", False),
            "is_analyzed":      True,
            "progress_step":    "done",
        })
        print(f"✅ [SESSION] 이전 세션 복원 완료 — STEP {idx + 1}/{len(steps)} 부터 재개")
        return True
    except Exception as e:
        print(f"⚠️ 세션 복원 실패: {e}")
        return False

# --- [실시간 코칭 인퍼런스 서버 통신] ---
async def call_runpod_inference(manual_img_path, camera_frame_bytes):
    if not RUNPOD_INFERENCE_URL:
        return {"result": "WAIT", "reason": "카메라 연결 대기 중..."}
    try:
        rel_path = manual_img_path.lstrip('/')
        full_manual_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(full_manual_path): return {"result": "ERROR", "reason": "이미지 없음"}

        async with httpx.AsyncClient() as client:
            files = {
                "manual_image": ("manual.jpg", open(full_manual_path, "rb"), "image/jpeg"),
                "camera_image": ("camera.jpg", io.BytesIO(camera_frame_bytes), "image/jpeg")
            }
            data = {"prompt": "비교 분석 수행"}
            response = await client.post(RUNPOD_INFERENCE_URL, files=files, data=data, timeout=15.0)
            if response.status_code == 200:
                return response.json().get("prediction", {"result": "UNKNOWN", "reason": "분석 오류"})
            return {"result": "ERROR", "reason": f"서버 오류: {response.status_code}"}
    except Exception as e:
        return {"result": "ERROR", "reason": f"통신 장애: {str(e)}"}

async def coaching_loop():
    while True:
        if state["is_analyzed"] and state["latest_frame"] and state["manual_steps"]:
            idx = state["current_step_idx"]
            if idx < len(state["manual_steps"]):
                current_step = state["manual_steps"][idx]
                prediction = await call_runpod_inference(current_step["image_url"], state["latest_frame"])
                if prediction:
                    state["ai_response"] = f"[{prediction.get('result', 'UNKNOWN')}] {prediction.get('reason', '분석 중...')}"
                    # step_locked가 아닐 때만 AI가 자동으로 단계를 올림
                    if not state["step_locked"]:
                        if prediction.get("result") == "PASS" and idx + 1 < len(state["manual_steps"]):
                            state["current_step_idx"] = idx + 1
                            save_session()  # AI 자동 진행도 저장
            await asyncio.sleep(3.0)
        else:
            await asyncio.sleep(1.0)

# ─── 생명주기 관리 (WebRTC 및 자동 코칭 백그라운드 태스크) ──────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_session()
    # coaching_task = asyncio.create_task(coaching_loop()) 
    livekit_task = asyncio.create_task(run_livekit())
    yield
    # coaching_task.cancel()
    livekit_task.cancel()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# --- [PDF 조각 분석: 한국어 강제 출력 및 정밀 필터링] ---
async def render_pdf_pages(pdf_path: str, out_dir: str, dpi: int = 200) -> list[str]:
    """pdftoppm으로 PDF 각 페이지를 JPEG 이미지로 변환 후 경로 목록 반환."""
    prefix = os.path.join(out_dir, "page")
    result = subprocess.run(
        ["pdftoppm", "-jpeg", "-r", str(dpi), pdf_path, prefix],
        capture_output=True
    )
    if result.returncode != 0:
        print(f"🚨 pdftoppm 오류: {result.stderr.decode()}")
        return []
    pages = sorted(glob.glob(f"{prefix}-*.jpg"))
    print(f"📄 [SYSTEM] PDF {len(pages)}페이지 렌더링 완료")
    return pages

async def analyze_pdf_page(client, page_img_path: str, page_num: int, job_dir: str) -> list[dict]:
    """
    페이지 전체 이미지를 Gemini에 넘겨 모든 STEP을 한 번에 추출.
    """
    try:
        with Image.open(page_img_path) as img:
            orig_w, orig_h = img.size
            img_rgb = img.convert("RGB")
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=88)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = """
당신은 조립 매뉴얼 디지털화 전문가입니다.
이 이미지는 조립 매뉴얼의 한 페이지입니다.

[임무]
1. 페이지에 있는 모든 STEP을 찾아 아래 JSON 배열로 반환하세요.
2. 각 STEP에 대해:
   - step_number: STEP 번호 (정수)
   - title: STEP 이미지 내부 또는 바로 위에 표시된 "STEP N" 레이블 텍스트 그대로
   - desc: STEP 이미지 바로 아래에 인쇄된 지시문 텍스트를 한 글자도 빠짐없이 그대로 복사하세요.
           반드시 한국어로 된 텍스트를 우선 사용하고, 한국어가 없으면 영문 원문 그대로 쓰세요.
           절대로 요약·번역·해석·재작성하지 마세요.
           줄바꿈은 문단이 완전히 바뀔 때만 사용하고, 한 문장 안에서는 줄바꿈하지 마세요.
   - box_2d: STEP 이미지(사진 박스)의 바운딩 박스 [ymin, xmin, ymax, xmax] (0~1000 정수 스케일)
             지시문 텍스트 영역은 포함하지 말고, 이미지(사진) 영역만 크롭하세요.

[중요 규칙]
- Teaching STEAM 로고, 페이지 배경, 브랜드 마크는 STEP이 아닙니다. 무시하세요.
- 이미지가 없고 텍스트만 있는 영역도 STEP이 아닙니다.
- desc가 비어있으면 해당 STEP을 포함하지 마세요.

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없음):
{"steps": [{"step_number": int, "title": "STEP N", "desc": "원문 지시문 그대로", "box_2d": [ymin, xmin, ymax, xmax]}]}
"""
        payload = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}
            ]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        res = await client.post(GEMINI_URL, json=payload, timeout=60.0)
        if res.status_code != 200:
            print(f"🚨 Gemini 오류 {res.status_code}: {res.text[:200]}")
            return []

        raw = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw)
        raw_steps = parsed.get("steps", [])

        steps = []
        with Image.open(page_img_path) as full_img:
            for s in raw_steps:
                box = s.get("box_2d")
                desc = s.get("desc", "").strip()
                if not box or not desc:
                    continue

                # 0~1000 스케일 → 픽셀 좌표
                ymin, xmin, ymax, xmax = box
                left   = int(xmin * orig_w / 1000)
                top    = int(ymin * orig_h / 1000)
                right  = int(xmax * orig_w / 1000)
                bottom = int(ymax * orig_h / 1000)

                # 범위 보정
                left, top     = max(0, left),   max(0, top)
                right, bottom = min(orig_w, right), min(orig_h, bottom)
                if right <= left or bottom <= top:
                    continue

                crop = full_img.crop((left, top, right, bottom))
                step_num = s.get("step_number", 0)
                c_path = os.path.join(job_dir, f"step_p{page_num}_{step_num}.jpg")
                crop.convert("RGB").save(c_path, "JPEG", quality=92)

                steps.append({
                    "step": step_num,
                    "title": s.get("title", f"STEP {step_num}"),
                    "desc": desc,
                    "image_url": f"/outputs/{os.path.relpath(c_path, OUTPUT_DIR)}".replace("\\", "/")
                })
                print(f"  ✅ STEP {step_num} 추출 완료: {desc[:40]}...")

        return steps

    except Exception as e:
        print(f"🚨 페이지 분석 에러 (page {page_num}): {e}")
        return []

async def detect_and_crop_image(client, img_path, base_idx):
    """
    이미지 파일(PNG/JPG 등) 전용 처리.
    """
    try:
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size
            img_rgb = img.convert("RGB")
            img_rgb.thumbnail((1600, 1600))
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=90)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = """
당신은 조립/공예 매뉴얼 분석 전문가입니다.
이 이미지는 여러 STEP이 격자 또는 순서대로 나열된 매뉴얼 이미지입니다.

[임무]
이미지에서 각 STEP을 찾아 아래 JSON으로 반환하세요.

각 STEP에 대해:
- step_number: 이미지에 표시된 번호 (원 안 숫자, 모서리 레이블 등). 없으면 순서대로 1부터.
- title: 이미지에 표시된 STEP 레이블 텍스트 그대로. 없으면 "STEP N" 형식으로.
- desc: 아래 두 경우 중 하나로 작성:
    (A) 이미지에 텍스트 지시문이 있으면 → 원문 그대로 한국어로 복사
    (B) 텍스트 지시문이 없으면 → 해당 STEP의 그림을 보고 수행해야 할 동작을 
        한국어로 구체적으로 설명 (예: "종이를 대각선으로 접어 삼각형을 만듭니다.")
        화살표, 점선, 접힘 방향 등 시각적 단서를 반드시 반영하세요.
        한 문장 안에서 줄바꿈하지 말고 자연스러운 문장으로 이어서 쓰세요.
- box_2d: 해당 STEP 그림 영역의 바운딩 박스 [ymin, xmin, ymax, xmax] (0~1000 정수 스케일)

[규칙]
- 배경, 로고, 제목 텍스트만 있는 영역은 STEP이 아닙니다.
- 모든 STEP을 빠짐없이 추출하세요.
- box_2d는 정확하게 해당 STEP 그림만 포함하세요 (여백 최소화).

반드시 아래 JSON 형식으로만 응답하세요:
{"steps": [{"step_number": int, "title": "STEP N", "desc": "한국어 설명", "box_2d": [ymin, xmin, ymax, xmax]}]}
"""
        payload = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}
            ]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        res = await client.post(GEMINI_URL, json=payload, timeout=60.0)
        if res.status_code != 200:
            print(f"🚨 Gemini 이미지 분석 오류 {res.status_code}")
            return []

        raw = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw)
        raw_steps = parsed.get("steps", [])

        job_dir = os.path.dirname(img_path)
        steps = []
        with Image.open(img_path) as full_img:
            orig_w, orig_h = full_img.size
            for s in raw_steps:
                box  = s.get("box_2d")
                desc = s.get("desc", "").strip()
                if not box or not desc:
                    continue

                ymin, xmin, ymax, xmax = box
                left   = max(0,      int(xmin * orig_w / 1000))
                top    = max(0,      int(ymin * orig_h / 1000))
                right  = min(orig_w, int(xmax * orig_w / 1000))
                bottom = min(orig_h, int(ymax * orig_h / 1000))
                if right <= left or bottom <= top:
                    continue

                crop     = full_img.crop((left, top, right, bottom))
                step_num = s.get("step_number") or (base_idx + len(steps) + 1)
                c_path   = os.path.join(job_dir, f"img_step_{step_num}_{int(time.time()*1000)}.jpg")
                crop.convert("RGB").save(c_path, "JPEG", quality=92)

                steps.append({
                    "step":      step_num,
                    "title":     s.get("title", f"STEP {step_num}"),
                    "desc":      desc,
                    "image_url": f"/outputs/{os.path.relpath(c_path, OUTPUT_DIR)}".replace("\\", "/")
                })
                print(f"  ✅ IMG STEP {step_num}: {desc[:50]}...")

        return steps

    except Exception as e:
        print(f"🚨 이미지 분석 에러: {e}")
    return []

# --- [API 라우팅] ---

@app.post("/process-manual")
async def handle_manual(files: List[UploadFile] = File(...)):
    start_time = time.time()
    state["is_analyzed"] = False
    state["step_locked"] = False
    state["progress_step"] = "upload"
    sess_id = f"sess_{int(start_time)}"
    job_dir = os.path.join(OUTPUT_DIR, sess_id)
    os.makedirs(job_dir, exist_ok=True)
    all_steps = []
    
    async with httpx.AsyncClient() as client:
        for f in files:
            file_path = os.path.join(job_dir, f.filename)
            with open(file_path, "wb") as b: b.write(await f.read())
            
            ext = f.filename.lower()
            if ext.endswith(".pdf"):
                pages_dir = os.path.join(job_dir, "pages")
                os.makedirs(pages_dir, exist_ok=True)
                print(f"📄 [SYSTEM] PDF 렌더링 시작: {f.filename}")
                state["progress_step"] = "render"

                # pdftoppm: blocking I/O → executor로 비동기 실행
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["pdftoppm", "-jpeg", "-r", "200", file_path,
                         os.path.join(pages_dir, "page")],
                        capture_output=True
                    )
                )
                page_imgs = sorted(glob.glob(os.path.join(pages_dir, "page-*.jpg")))
                state["progress_step"] = "analyze"
                print(f"🖼️ [SYSTEM] {len(page_imgs)}페이지 → Gemini 병렬 분석 시작")

                # 모든 페이지를 동시에 Gemini 호출 (순차 → 병렬)
                tasks = [
                    analyze_pdf_page(client, p_img, p_idx + 1, job_dir)
                    for p_idx, p_img in enumerate(page_imgs)
                ]
                page_results = await asyncio.gather(*tasks)
                for page_steps in page_results:
                    all_steps.extend(page_steps)
                
            elif ext.endswith((".png", ".jpg", ".jpeg")):
                img_steps = await detect_and_crop_image(client, file_path, len(all_steps))
                all_steps.extend(img_steps)
                
    unique_steps = []
    seen_urls = set()
    for s in all_steps:
        if s['image_url'] not in seen_urls:
            unique_steps.append(s)
            seen_urls.add(s['image_url'])
            
    def get_step_num(x):
        val = x.get('step')
        if val is None or not str(val).isdigit(): return 999
        return int(val)

    unique_steps.sort(key=get_step_num)
    
    final_filtered_steps = []
    for i, s in enumerate(unique_steps):
        is_boundary = (i == 0 or i == len(unique_steps) - 1)
        title = s.get("title", "").lower()
        desc = s.get("desc", "").lower()
        
        if is_boundary:
            if any(k in title or k in desc for k in ["안내", "steam", "teaching", "copyright"]):
                if get_step_num(s) == 999 or get_step_num(s) == 0:
                    continue
        
        final_filtered_steps.append(s)

    with open(os.path.join(job_dir, "instruction.json"), "w", encoding="utf-8") as f:
        json.dump(final_filtered_steps, f, ensure_ascii=False, indent=4)
    
    state.update({
        "manual_steps": final_filtered_steps, 
        "is_analyzed": True, 
        "analysis_time": round(time.time() - start_time, 2),
        "current_step_idx": 0
    })
    state["progress_step"] = "done"
    save_session()
    print(f"✅ [SUCCESS] 분석 완료.")
    return {"status": "success", "steps": final_filtered_steps}

@app.post("/set-step")
async def set_step(body: dict):
    total = len(state["manual_steps"])
    if total == 0: return {"status": "error", "message": "매뉴얼이 로드되지 않았습니다."}
    idx, locked = max(0, min(int(body.get("idx", state["current_step_idx"])), total - 1)), bool(body.get("locked", True))
    state["current_step_idx"], state["step_locked"] = idx, locked
    save_session()
    return {"status": "ok", "current_step_idx": idx, "step_locked": locked}




@app.get("/status")
async def get_status():
    return {
        "manual_steps": state["manual_steps"],
        "is_analyzed": state["is_analyzed"], "current_step_idx": state["current_step_idx"],
        "ai_response": state["ai_response"], "analysis_time": state["analysis_time"],
        "progress_step": state["progress_step"], "step_locked": state["step_locked"],
        "log": state["log"], "current_fps": state["current_fps"], "last_rtt": state["last_rtt"],
        "has_frame": state["latest_frame"] is not None
    }

@app.post("/reset")
async def reset_session():
    """새로운 매뉴얼을 올리기 위해 기존 분석 데이터를 싹 지웁니다."""
    state.update({
        "manual_steps": [],
        "current_step_idx": 0,
        "ai_response": "대기 중...",
        "is_analyzed": False,
        "analysis_time": 0.0,
        "step_locked": False,
        "progress_step": "upload"
    })
    # 저장되어 있던 세션 파일도 삭제
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
    print("🗑️ [SESSION] 이전 매뉴얼 데이터 초기화 완료")
    return {"status": "ok"}

@app.get("/stream")
async def stream():
    async def gen():
        while True:
            if state["latest_frame"]: yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + state["latest_frame"] + b"\r\n")
            await asyncio.sleep(0.04)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/trigger-vlm")
async def trigger_vlm_analysis():
    """프론트엔드에서 'VLM 분석' 버튼을 눌렀을 때 1회 호출되는 엔드포인트"""
    # 1. 상태 체크 (매뉴얼 로드 및 프레임 수신 여부)
    if not state["is_analyzed"] or not state["manual_steps"]:
        return {"status": "error", "message": "매뉴얼이 아직 준비되지 않았습니다."}
    if not state["latest_frame"]:
        return {"status": "error", "message": "카메라 렌즈를 통해 프레임을 수신하는 중입니다."}

    # 2. 현재 스텝 및 이미지 가져오기
    idx = state["current_step_idx"]
    current_step = state["manual_steps"][idx]

    # 3. GPU 인퍼런스 서버로 전송 (이미 만들어진 함수 재사용)
    prediction = await call_runpod_inference(current_step["image_url"], state["latest_frame"])

    # 4. 결과 처리 및 상태 업데이트
    if prediction and prediction.get("result") != "ERROR":
        result_status = prediction.get("result", "UNKNOWN")
        reason = prediction.get("reason", "분석 완료")
        
        # 전역 상태(Web UI 출력용) 업데이트
        state["ai_response"] = f"[{result_status}] {reason}"

        # 분석 결과가 PASS이고, 사용자가 단계를 고정(lock)하지 않았다면 다음 단계로 자동 이동
        if result_status == "PASS" and not state["step_locked"]:
            if idx + 1 < len(state["manual_steps"]):
                state["current_step_idx"] = idx + 1
                save_session()

        return {
            "status": "success", 
            "prediction": prediction,
            "current_step": state["current_step_idx"]
        }
    else:
        err_reason = prediction.get("reason", "알 수 없는 오류") if prediction else "응답 없음"
        return {"status": "error", "message": f"VLM 통신 실패: {err_reason}"}

# ==========================================
# 🌟 WebRTC (LiveKit) 영상 처리 파트 추가
# ==========================================
async def process_video_track(track: rtc.VideoTrack):
    global prev_frame_time
    video_stream = rtc.VideoStream(track)
    target_format = rtc.VideoFormatType.FORMAT_RGBA8888 if hasattr(rtc, 'VideoFormatType') else 3

    last_process_time = time.time()

    async for event in video_stream:
        await asyncio.sleep(0.001)

        curr_time = time.time()
        if prev_frame_time > 0:
            diff = curr_time - prev_frame_time
            if diff > 0.01: state["current_fps"] = 1.0 / diff
        prev_frame_time = curr_time
        
        # 20FPS 리밋 제어
        if curr_time - last_process_time < 0.05: continue
        last_process_time = time.time()

        frame = event.frame
        rgba_frame = frame.convert(target_format)
        img = np.frombuffer(rgba_frame.data, dtype=np.uint8).reshape((rgba_frame.height, rgba_frame.width, 4))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        h, w = img_bgr.shape[:2]
        target_size = 720
        new_w, new_h = (target_size, int(h * target_size / w)) if h > w else (int(w * target_size / h), target_size)
        
        img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        _, buffer = cv2.imencode('.jpg', img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        
        frame_bytes = buffer.tobytes()
        state["latest_frame"], state["latest_frame_raw"] = frame_bytes, frame_bytes
        state["log"] = f"📡 720p Stream | FPS: {round(state['current_fps'], 1):4} | ABS: {round(state['last_rtt']/2, 1):5}ms"

async def run_livekit():
    room = rtc.Room()
    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO: asyncio.create_task(process_video_track(track))
        
    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        try:
            payload = json.loads(data.data.decode("utf-8"))
            if payload.get("type") == "ping":
                state["client_send_time"] = payload.get("send_time")
                resp = json.dumps({"type": "pong", "client_time": payload.get("send_time")})
                asyncio.create_task(room.local_participant.publish_data(resp.encode("utf-8")))
            elif payload.get("type") == "metrics":
                state["last_rtt"] = float(payload.get("rtt", 0.0))
        except: pass
    
    try:
        if not LIVEKIT_TOKEN:
            print("🚨 [ERROR] LIVEKIT_TOKEN이 .env 파일에 없습니다!")
            return
        await room.connect(LIVEKIT_URL, LIVEKIT_TOKEN)
        print("✅ [SYSTEM] LiveKit 서버 접속 성공")
    except Exception as e:
        print(f"🚨 [ERROR] LiveKit 접속 실패: {e}")

@app.get("/")
@app.get("/mobile")
async def serve_ui():
    if os.path.exists("index.html"): return FileResponse("index.html")
    return {"error": "index.html not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)