import asyncio
import base64
import glob
import io
import json
import os
import re
import statistics
import subprocess
import time
from typing import List

import httpx
from fastapi import UploadFile
from PIL import Image

from config import BASE_DIR, GEMINI_URL, OUTPUT_DIR, PRELOAD_MANUAL_STEPS, RUNPOD_INFERENCE_BASE_URL
from state import save_session, state
from vlm import REGISTERED_STEP_IDS, WARMED_STEP_IDS, preload_steps_to_gpu


def _render_pdf_to_jpegs(pdf_path, out_prefix, dpi, first_page=None, last_page=None):
    """PDF → JPEG 렌더링(크로스플랫폼).

    pypdfium2(pip 휠, Windows/Linux/macOS 외부 바이너리 불필요)를 우선 사용하고,
    없으면 Poppler `pdftoppm`으로 폴백한다. 출력 파일명은 pdftoppm과 동일하게
    `{prefix}-{n}.jpg` 형식(문서 전체 페이지 자릿수로 zero-pad)이라 기존 sorted(glob) 정렬이 유지된다.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        pdfium = None

    if pdfium is not None:
        pdf = pdfium.PdfDocument(pdf_path)
        try:
            total = len(pdf)
            start = (first_page - 1) if first_page else 0
            end = min(last_page, total) if last_page else total
            width = max(1, len(str(total)))
            scale = dpi / 72.0
            out = []
            for i in range(start, end):
                pil = pdf[i].render(scale=scale).to_pil()
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
                path = f"{out_prefix}-{str(i + 1).zfill(width)}.jpg"
                pil.save(path, "JPEG", quality=90)
                out.append(path)
            return out
        finally:
            pdf.close()

    # 폴백: Poppler pdftoppm (설치돼 있는 리눅스/맥 환경)
    cmd = ["pdftoppm", "-jpeg", "-r", str(dpi)]
    if first_page:
        cmd += ["-f", str(first_page)]
    if last_page:
        cmd += ["-l", str(last_page)]
    cmd += [pdf_path, out_prefix]
    subprocess.run(cmd, capture_output=True)
    return sorted(glob.glob(f"{out_prefix}-*.jpg"))

_preview_steps: list = []
_preview_updated: bool = False
_analysis_start_time: float = 0.0


def _trace_timing(label: str, start_perf: float, **details) -> None:
    elapsed = time.perf_counter() - start_perf
    suffix = ""
    if details:
        suffix = " " + " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[MANUAL_TIMING] {label}: {elapsed:.2f}s{suffix}")

# gemini-3 thinking 양(낮을수록 빠름·정확도↓).
# - 번역: 항상 low (직역 품질 동일, 무조건 빠름)
# - 탐지/하위셀: 매뉴얼 타입으로 자동(단일 이미지→low, 복잡한 PDF→high) + 요청별 override.
TRANSLATE_THINKING = "low"
# env로 강제(low/high). 비어 있으면 타입 기반 자동.
_FORCED_THINKING = os.getenv("GEMINI_THINKING_LEVEL", "").strip() or None
# process_manual_files에서 파일별로 세팅됨(탐지/하위셀 호출이 참조).
_active_detect_thinking = "high"
GEMINI_HTTP_CLIENT: httpx.AsyncClient | None = None
GEMINI_WARMED = False


def get_gemini_http_client() -> httpx.AsyncClient:
    global GEMINI_HTTP_CLIENT
    if GEMINI_HTTP_CLIENT is None or GEMINI_HTTP_CLIENT.is_closed:
        GEMINI_HTTP_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
    return GEMINI_HTTP_CLIENT


def ensure_gemini_http_client() -> None:
    get_gemini_http_client()


async def close_gemini_http_client() -> None:
    global GEMINI_HTTP_CLIENT
    if GEMINI_HTTP_CLIENT is not None and not GEMINI_HTTP_CLIENT.is_closed:
        await GEMINI_HTTP_CLIENT.aclose()
    GEMINI_HTTP_CLIENT = None


async def warmup_gemini() -> None:
    global GEMINI_WARMED
    if GEMINI_WARMED or not GEMINI_URL or GEMINI_URL.endswith("key="):
        return

    client = get_gemini_http_client()
    start = time.perf_counter()
    try:
        res = await client.post(
            GEMINI_URL,
            json={
                "contents": [{"parts": [{"text": "warmup"}]}],
                "generationConfig": _gen_config(
                    temperature=0,
                    json_out=False,
                    thinking=TRANSLATE_THINKING,
                ),
            },
            timeout=15.0,
        )
        GEMINI_WARMED = res.status_code == 200
        print(
            f"[GEMINI] warmup status={res.status_code} "
            f"elapsed={(time.perf_counter() - start) * 1000:.0f}ms"
        )
    except Exception as e:
        print(f"[GEMINI] warmup failed: {type(e).__name__}: {e}")


def _decide_thinking(filename: str, override: str | None) -> str:
    """탐지/하위셀에 쓸 thinking 레벨 결정. 우선순위: env 강제 > 요청 override > 타입 자동."""
    if _FORCED_THINKING:
        return _FORCED_THINKING
    if override in ("low", "high"):
        return override
    ext = filename.lower()
    if ext.endswith((".png", ".jpg", ".jpeg")):
        return "low"   # 단일 이미지(종이학 등) = 단순 → 빠름
    return "high"      # PDF(레고 등 조립 매뉴얼) = 복잡 → 정확


def _gen_config(temperature: float = 0, json_out: bool = True, thinking: str | None = None) -> dict:
    cfg: dict = {"temperature": temperature}
    if json_out:
        cfg["responseMimeType"] = "application/json"
    # thinking 미지정 시 현재 활성 탐지 레벨 사용(번역은 thinking=TRANSLATE_THINKING로 명시).
    level = thinking if thinking is not None else _active_detect_thinking
    if level:
        cfg["thinkingConfig"] = {"thinkingLevel": level}
    return cfg


def reset_preview_state() -> None:
    global _preview_steps, _preview_updated
    _preview_steps = []
    _preview_updated = False


def get_elapsed_time() -> float:
    if state["progress_step"] not in ("upload", "done") and _analysis_start_time > 0:
        return round(time.time() - _analysis_start_time, 1)
    return state["analysis_time"]


def _normalized_manual_name(name: str) -> str:
    return os.path.basename(str(name or "")).strip().lower()


def saved_manual_name_exists(name: str, exclude_dir: str | None = None) -> bool:
    target = _normalized_manual_name(name)
    if not target:
        return False

    exclude_abs = os.path.abspath(exclude_dir) if exclude_dir else ""
    for sess_dir in glob.glob(os.path.join(OUTPUT_DIR, "sess_*")):
        if exclude_abs and os.path.abspath(sess_dir) == exclude_abs:
            continue
        meta_path = os.path.join(sess_dir, "session.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if _normalized_manual_name(meta.get("name", "")) == target:
            return True
    return False


def load_saved_manual_by_name(name: str, exclude_dir: str | None = None) -> dict | None:
    target = _normalized_manual_name(name)
    if not target:
        return None

    exclude_abs = os.path.abspath(exclude_dir) if exclude_dir else ""
    matches = []
    for sess_dir in glob.glob(os.path.join(OUTPUT_DIR, "sess_*")):
        if exclude_abs and os.path.abspath(sess_dir) == exclude_abs:
            continue
        meta_path = os.path.join(sess_dir, "session.json")
        instr_path = os.path.join(sess_dir, "instruction.json")
        if not os.path.isfile(meta_path) or not os.path.isfile(instr_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if _normalized_manual_name(meta.get("name", "")) != target:
            continue
        try:
            updated = float(meta.get("updated") or os.path.getmtime(instr_path))
        except OSError:
            updated = 0.0
        matches.append((updated, sess_dir, meta, instr_path))

    for _, sess_dir, meta, instr_path in sorted(matches, key=lambda item: item[0], reverse=True):
        try:
            with open(instr_path, "r", encoding="utf-8") as f:
                steps = json.load(f)
        except Exception:
            continue
        if not isinstance(steps, list) or not steps:
            continue

        missing = 0
        for step in steps:
            if not isinstance(step, dict):
                missing += 1
                continue
            image_url = step.get("image_url", "")
            image_path = os.path.join(BASE_DIR, image_url.lstrip("/"))
            if not image_url or not os.path.isfile(image_path):
                missing += 1
        if missing:
            print(f"[CACHE] 저장된 매뉴얼 이미지 누락으로 재사용 생략: {name} missing={missing}/{len(steps)}")
            continue

        preview = ""
        thumbs = sorted(glob.glob(os.path.join(sess_dir, "thumb-*.jpg")))
        if thumbs:
            preview = f"/outputs/{os.path.relpath(thumbs[0], OUTPUT_DIR)}".replace("\\", "/")
        elif steps and isinstance(steps[0], dict):
            preview = steps[0].get("image_url", "")

        return {
            "sess_dir": sess_dir,
            "steps": steps,
            "meta": meta,
            "preview": preview,
        }
    return None


def _has_foreign_text(text: str) -> bool:
    """영어 등 외국어 단어가 포함돼 있으면 True."""
    return bool(re.search(r'[a-zA-Z]{3,}', text))


async def _translate_to_korean(client: httpx.AsyncClient, text: str) -> str:
    """외국어 텍스트를 한국어로 번역. 실패 시 원문 반환."""
    try:
        res = await client.post(
            GEMINI_URL,
            json={
                "contents": [{"parts": [{"text": f"다음 텍스트를 한국어로 번역하세요. 번역문만 출력하고 다른 설명은 하지 마세요.\n\n{text}"}]}],
                "generationConfig": _gen_config(temperature=0.1, json_out=False, thinking=TRANSLATE_THINKING),
            },
            timeout=15.0,
        )
        if res.status_code == 200:
            translated = res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            if translated:
                return translated
    except Exception:
        pass
    return text


async def _translate_to_korean_batch(client: httpx.AsyncClient, texts: list) -> list:
    """외국어가 포함된 항목만 '한 번의 호출'로 한국어 직역. id 기준 되매핑이라 순서/누락에 강함.
    실패 시 해당 항목은 원문 유지. (단계별 직렬 번역 → 1회 배치로 단축)"""
    idxs = [i for i, t in enumerate(texts) if t and _has_foreign_text(t)]
    if not idxs:
        return list(texts)
    items = [{"id": i, "text": texts[i]} for i in idxs]
    prompt = (
        "다음 JSON 배열의 각 항목 text를 한국어로 '직역'하세요.\n"
        "- 원문 의미를 그대로 옮기고 요약·의역·내용 추가·생략을 절대 하지 마세요.\n"
        "- id는 그대로 두고 text만 한국어 번역으로 바꾼, 동일 구조의 JSON 배열만 출력하세요.\n\n"
        + json.dumps(items, ensure_ascii=False)
    )
    out = list(texts)
    try:
        res = await client.post(
            GEMINI_URL,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": _gen_config(thinking=TRANSLATE_THINKING),
            },
            timeout=30.0,
        )
        if res.status_code == 200:
            payload = json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"])
            if isinstance(payload, dict):
                payload = payload.get("translations") or payload.get("items") or payload.get("results") or []
            if isinstance(payload, list):
                for it in payload:
                    if isinstance(it, dict) and isinstance(it.get("id"), int) and it.get("text"):
                        i = it["id"]
                        if 0 <= i < len(out):
                            out[i] = str(it["text"]).strip()
    except Exception as e:
        print(f"🚨 배치 번역 실패(원문 유지): {e}")
    return out


async def generate_step_desc(image_path: str) -> str:
    """크롭 이미지 1장을 보고 조립 동작을 한국어 한 문장으로 생성(편집 UI '잘라 추가'용).
    Gemini 1회 호출. 실패 시 빈 문자열."""
    try:
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            rgb.thumbnail((1024, 1024))
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        prompt = (
            "이 그림은 조립 매뉴얼의 한 단계입니다. 무엇을 어떻게 조립하는지 "
            "한국어 한 문장으로 간단히 설명하세요. 설명 문장만 출력하고 다른 말은 하지 마세요."
        )
        client = get_gemini_http_client()
        res = await client.post(
            GEMINI_URL,
            json={
                "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": b64}}]}],
                "generationConfig": _gen_config(temperature=0.2, json_out=False, thinking=TRANSLATE_THINKING),
            },
            timeout=30.0,
        )
        if res.status_code == 200:
            return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        print(f"🚨 설명 생성 응답 {res.status_code}: {res.text[:200]}")
    except Exception as e:
        print(f"🚨 설명 생성 실패: {e}")
    return ""


def _tolerant_json(text: str):
    """모델이 낸 비표준 JSON도 최대한 파싱: 코드펜스 제거, 끝쉼표 제거, 후행 텍스트 허용.
    최상위 객체({}) 또는 배열([]) 모두 허용. 실패 시 None."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):  # ```json … ``` 코드펜스 제거
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t).rstrip("`").strip()
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        return None
    t = t[min(starts):]
    # 끝쉼표(예: {"a":1,} / [1,2,]) 제거한 버전을 우선 시도
    candidates = [re.sub(r",(\s*[}\]])", r"\1", t), t]
    for c in candidates:
        try:
            obj, _ = json.JSONDecoder().raw_decode(c)  # 첫 JSON 값만, 후행 텍스트 무시
            return obj
        except Exception:
            continue
    return None


def extract_gemini_steps(response_json: dict) -> list:
    text = response_json["candidates"][0]["content"]["parts"][0]["text"]
    payload = _tolerant_json(text)
    if payload is None:
        print(f"⚠️ JSON 파싱 실패 — 응답 일부: {text[:200]!r}")
        return []

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
            if isinstance(item.get("steps"), list):  # 리스트로 감싼 {"steps":[...]} 언랩
                normalized.extend(x for x in item["steps"] if isinstance(x, dict))
            else:
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


def _detect_columns(page_img_path: str):
    """페이지를 세로 구분선 기준으로 열(column)들로 분할.
    (w, content_left, content_right, [(x0,x1), ...]) 반환."""
    import numpy as np
    with Image.open(page_img_path) as im:
        w, h = im.size
        g = np.array(im.convert("L")).astype(int)
        rgb = np.array(im.convert("RGB")).astype(int)
    bg = np.median(rgb.reshape(-1, 3), axis=0)
    dist = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
    cc = np.where((dist > 40).sum(axis=0) / h > 0.01)[0]
    if cc.size == 0:
        return w, 0, w, [(0, w)]
    cl, cr = int(cc[0]), int(cc[-1]) + 1
    # 세로 구분선 = 높이의 절반 이상이 어두운 픽셀로 채워진 x열
    fr = (g < 110).sum(axis=0) / h
    seps, cluster = [], []
    for x in range(cl, cr):
        if fr[x] > 0.5:
            cluster.append(x)
        elif cluster:
            c = (cluster[0] + cluster[-1]) // 2
            if cl + w * 0.03 < c < cr - w * 0.03:
                seps.append(c)
            cluster = []
    if cluster:
        c = (cluster[0] + cluster[-1]) // 2
        if cl + w * 0.03 < c < cr - w * 0.03:
            seps.append(c)
    bounds = [cl] + seps + [cr]
    cols = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    return w, cl, cr, cols


def _crop_x_range(page_img_path: str, x0: int, x1: int, job_dir: str, tag: str) -> str | None:
    """페이지에서 [x0,x1] 가로 구간을 한 장(strip)으로 크롭. 상하/좌우 여백 트림 + 빈/얇은 영역 제외."""
    try:
        import numpy as np
        with Image.open(page_img_path) as im:
            w, h = im.size
            rgb = im.convert("RGB")
        x0, x1 = max(0, int(x0)), min(w, int(x1))
        if x1 - x0 < w * 0.05:
            return None
        region = rgb.crop((x0, 0, x1, h))
        arr = np.array(region)
        rows = np.where(arr.std(axis=(1, 2)) > 12)[0]
        if rows.size == 0:
            return None
        region = region.crop((0, int(rows[0]), region.width, int(rows[-1]) + 1))
        arr = np.array(region)
        colsm = np.where(arr.std(axis=(0, 2)) > 12)[0]
        if colsm.size == 0:
            return None
        region = region.crop((int(colsm[0]), 0, int(colsm[-1]) + 1, region.height))
        if region.width < 8 or region.height < 8:
            return None
        out = os.path.join(job_dir, f"cont_{tag}_{int(time.time() * 1000)}.jpg")
        region.save(out, "JPEG", quality=92)
        return f"/outputs/{os.path.relpath(out, OUTPUT_DIR)}".replace("\\", "/")
    except Exception as e:
        print(f"🚨 연속 영역 크롭 실패 ({tag}): {e}")
        return None


def _url_to_fs(image_url: str) -> str:
    """/outputs/... URL을 실제 파일 경로로 변환."""
    return os.path.join(OUTPUT_DIR, image_url.split("/outputs/", 1)[-1])


def _loads_first_json(text: str) -> dict:
    """모델 응답에서 첫 JSON 객체를 관대하게 파싱(코드펜스/끝쉼표/후행 텍스트 허용)."""
    obj = _tolerant_json(text)
    return obj if isinstance(obj, dict) else {}


def _stacked_cells(cells: list) -> list:
    """세로로 쌓인 셀만 남긴다(부품 픽업 인셋처럼 나란히 놓인 것은 제외).
    ymin 정렬 후, 직전 채택 셀과 y중첩이 40% 미만인 셀만 채택."""
    out: list = []
    for c in sorted(cells, key=lambda x: x["box_2d"][0]):
        if not out:
            out.append(c)
            continue
        p, q = out[-1]["box_2d"], c["box_2d"]
        inter = max(0, min(p[2], q[2]) - max(p[0], q[0]))
        small = min(p[2] - p[0], q[2] - q[0]) or 1
        if inter / small < 0.4:
            out.append(c)
    return out


async def _detect_cells_in_strip(client, strip_path: str) -> list:
    """단일 열(STEP 영역) strip 이미지에서 번호가 매겨진 하위 조립 단계 셀을 탐지.
    [{num, desc, box_2d}] 위→아래 순으로 반환(세로 스택만)."""
    try:
        with Image.open(strip_path) as im:
            rgb = im.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            "이 이미지는 조립 매뉴얼의 한 STEP 영역(세로 열)입니다.\n"
            "그 안의 번호가 매겨진 하위 조립 단계 셀을 위에서 아래 순서로 모두 찾으세요.\n"
            "- num: 셀 옆 작은 단계 번호(정수). 맨 위 큰 STEP 번호는 제외.\n"
            "- 노란색/강조색 인셋 박스 안 부품 픽업 번호는 제외.\n"
            "- desc: 그 하위 단계 조립 동작을 한국어 한 문장으로.\n"
            "- box_2d: [ymin,xmin,ymax,xmax] 0~1000\n"
            '{"cells":[{"num":int,"desc":str,"box_2d":[int,int,int,int]}]}'
        )
        res = await client.post(
            GEMINI_URL,
            json={
                "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": b64}}]}],
                "generationConfig": _gen_config(),
            },
            timeout=60.0,
        )
        if res.status_code != 200:
            print(f"🚨 하위셀 탐지 응답 {res.status_code}: {res.text[:400]}")
            return []
        text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        payload = _loads_first_json(text)
        raw = payload.get("cells", []) if isinstance(payload, dict) else []
        cells = [c for c in raw if isinstance(c, dict) and c.get("box_2d") and c.get("num")]
        kept = _stacked_cells(cells)
        # 출력 순서는 읽기 순서(열 좌→우, 같은 열은 위→아래)로 — 박스가 여러 열에 걸쳐도 올바른 순번
        kept.sort(key=lambda c: (c["box_2d"][1] // 300, c["box_2d"][0]))
        return kept
    except Exception as e:
        print(f"🚨 하위셀 탐지 실패: {e}")
        return []


def _crop_cell(src_url: str, box_2d: list, job_dir: str, tag: str) -> str | None:
    """strip 이미지(src_url)에서 box_2d(0~1000) 영역을 잘라 하위단계 이미지로 저장."""
    try:
        src_fs = _url_to_fs(src_url)
        with Image.open(src_fs) as im:
            w, h = im.size
            ymin, xmin, ymax, xmax = box_2d
            left = max(0, int(xmin * w / 1000))
            top = max(0, int(ymin * h / 1000))
            right = min(w, int(xmax * w / 1000))
            bottom = min(h, int(ymax * h / 1000))
            if right <= left or bottom <= top:
                return None
            out = os.path.join(job_dir, f"{tag}_{int(time.time() * 1000)}.jpg")
            im.crop((left, top, right, bottom)).convert("RGB").save(out, "JPEG", quality=92)
        return f"/outputs/{os.path.relpath(out, OUTPUT_DIR)}".replace("\\", "/")
    except Exception as e:
        print(f"🚨 하위단계 크롭 실패 ({tag}): {e}")
        return None


def _img_sig(image_url: str):
    """이미지의 32x32 그레이스케일 시그니처(중복 비교용). 실패 시 None."""
    try:
        import numpy as np
        with Image.open(_url_to_fs(image_url)) as im:
            return np.asarray(im.convert("L").resize((32, 32)), dtype=float)
    except Exception:
        return None


def _sig_diff(a, b) -> float:
    import numpy as np
    return float(np.abs(a - b).mean())


async def _expand_substeps(client, steps: list, job_dir: str) -> list:
    """하위셀이 2개 이상인 메인 STEP을 N-1, N-2 … 개별 STEP(평면)으로 확장.
    1개 이하면 원본 STEP을 그대로 유지."""
    async def cells_of(stp):
        main_url = stp["image_url"]
        extra_urls = stp.get("extra_images", [])
        found = await asyncio.gather(
            *[_detect_cells_in_strip(client, _url_to_fs(u)) for u in [main_url] + extra_urls]
        )
        main_cells, extra_found = found[0], found[1:]
        out = [(main_url, c) for c in main_cells]  # 메인 열의 하위셀
        for url, cells in zip(extra_urls, extra_found):
            if cells:
                out.extend((url, c) for c in cells)
            else:
                # 번호 검출 실패한 연속 strip은 통째로 하나의 하위단계로(내용 유실 방지)
                out.append((url, {"num": None, "box_2d": None, "desc": stp.get("desc", ""), "_whole": True}))
        return out

    per_step = await asyncio.gather(*[cells_of(s) for s in steps])
    expanded: list = []
    for stp, cells in zip(steps, per_step):
        if len(cells) < 2:
            expanded.append(stp)  # 단일 셀 STEP은 그대로(연속 strip 있으면 유지)
            continue
        n = stp["step"]
        # 1) 각 하위셀 크롭
        built: list = []
        for src_url, c in cells:
            crop_url = src_url if c.get("_whole") or not c.get("box_2d") else _crop_cell(src_url, c["box_2d"], job_dir, f"sub_{n}_{len(built)+1}")
            built.append({"image_url": crop_url or src_url, "desc": c.get("desc") or ""})
        # 2) 이미지가 거의 동일한 하위단계 제거(메인 이미지+연속 strip에 같은 영역이 두 번 잡히는 중복)
        kept, sigs = [], []
        for b in built:
            sig = _img_sig(b["image_url"])
            if sig is not None and any(s is not None and _sig_diff(s, sig) < 18 for s in sigs):
                continue  # 앞 하위단계와 거의 동일 → 중복으로 제거
            kept.append(b); sigs.append(sig)
        # 3) 중복 제거 후 2개 미만이면 분할 의미 없음 → 단일 STEP 유지
        if len(kept) < 2:
            expanded.append(stp)
            continue
        for i, b in enumerate(kept, 1):
            expanded.append({
                "step": n,
                "sub": i,
                "label": f"{n}-{i}",
                "title": f"STEP {n}",
                "image_url": b["image_url"],
                "desc": b["desc"] or f"{n}-{i} 단계",
            })
        if len(kept) < len(built):
            print(f"  🧹 STEP {n} 중복 하위단계 {len(built)-len(kept)}개 제거 → {len(kept)}개")
        else:
            print(f"  ✂️ STEP {n} → {len(kept)}개 하위단계로 분할")
    return expanded


async def analyze_picture_page(client, page_img_path: str, page_num: int, job_dir: str) -> list:
    """그림형 매뉴얼 전용 파이프라인: 4분할(TL/TR/BL/BR) 병렬 탐지, 위치 기준 정렬."""
    try:
        with Image.open(page_img_path) as img:
            orig_w, orig_h = img.size
            rgb = img.convert("RGB")
            rgb.thumbnail((1600, 1600))
            thumb_w, thumb_h = rgb.size
            mx, my = thumb_w // 2, thumb_h // 2

            def to_b64(pil_img):
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=90)
                return base64.b64encode(buf.getvalue()).decode("utf-8")

            # 4분면: TL, TR, BL, BR
            quads = [
                (to_b64(rgb.crop((0,  0,  mx,      my))),      0,   0),   # TL
                (to_b64(rgb.crop((mx, 0,  thumb_w, my))),      500, 0),   # TR
                (to_b64(rgb.crop((0,  my, mx,      thumb_h))), 0,   500), # BL
                (to_b64(rgb.crop((mx, my, thumb_w, thumb_h))), 500, 500), # BR
            ]

        prompt = (
            "그림 전용 조립 매뉴얼입니다.\n"
            "이미지에서 번호가 붙은 조립 단계 셀을 모두 찾아 반환하세요.\n"
            "- step_number: 셀에 표시된 번호 (정수, 없으면 0)\n"
            "- desc: 조립 동작을 한국어로 간단히 설명\n"
            "- box_2d: [ymin,xmin,ymax,xmax] 0~1000\n"
            '{"steps":[{"step_number":int,"desc":str,"box_2d":[int,int,int,int]}]}'
        )

        async def call(b64):
            res = await client.post(
                GEMINI_URL,
                json={
                    "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": b64}}]}],
                    "generationConfig": _gen_config(),
                },
                timeout=60.0,
            )
            return extract_gemini_steps(res.json()) if res.status_code == 200 else []

        quad_results = await asyncio.gather(*[call(b64) for b64, _, _ in quads])

        # 각 사분면 좌표 → 전체 페이지 좌표 (0~1000)로 변환
        # x_off, y_off는 해당 사분면이 전체에서 차지하는 시작 비율 × 1000
        def remap(steps, x_off, y_off):
            out = []
            for s in steps:
                box = s.get("box_2d")
                if box:
                    ymin, xmin, ymax, xmax = box
                    s = dict(s, box_2d=[
                        y_off + ymin // 2,
                        x_off + xmin // 2,
                        y_off + ymax // 2,
                        x_off + xmax // 2,
                    ])
                out.append(s)
            return out

        all_raw = []
        for (_, x_off, y_off), steps in zip(quads, quad_results):
            all_raw.extend(remap(steps, x_off, y_off))

        # IoU 기반 중복 제거: 공간적으로 30% 이상 겹치면 더 작은(정밀한) 박스를 유지
        def iou(b1, b2):
            iy1, ix1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            iy2, ix2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            if iy2 <= iy1 or ix2 <= ix1:
                return 0.0
            inter = (iy2 - iy1) * (ix2 - ix1)
            a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            return inter / (a1 + a2 - inter)

        deduped: list = []
        for s in all_raw:
            box = s.get("box_2d")
            if not box:
                continue
            replaced = False
            for i, d in enumerate(deduped):
                dbox = d.get("box_2d")
                if dbox and iou(box, dbox) > 0.3:
                    # 더 작고 정밀한 박스를 유지
                    area_s = (box[2]-box[0]) * (box[3]-box[1])
                    area_d = (dbox[2]-dbox[0]) * (dbox[3]-dbox[1])
                    if area_s < area_d:
                        deduped[i] = s
                    replaced = True
                    break
            if not replaced:
                deduped.append(s)

        # 위치 기준 정렬: y 100단위 행 그룹 → 같은 행은 x 순
        ordered = sorted(
            deduped,
            key=lambda s: (s["box_2d"][0] // 100, s["box_2d"][1]) if s.get("box_2d") else (999, 999)
        )

        # 외국어 desc는 1회 배치 호출로 직역(단계별 직렬 번역 제거)
        descs = await _translate_to_korean_batch(
            client, [s.get("desc", "").strip() or f"조립 단계 {i + 1}" for i, s in enumerate(ordered)]
        )

        steps = []
        with Image.open(page_img_path) as full_img:
            import numpy as np
            for i, s in enumerate(ordered):
                box = s.get("box_2d")
                if not box:
                    continue
                desc = descs[i]
                ymin, xmin, ymax, xmax = box
                left   = max(0,      int(xmin * orig_w / 1000))
                top    = max(0,      int(ymin * orig_h / 1000))
                right  = min(orig_w, int(xmax * orig_w / 1000))
                bottom = min(orig_h, int(ymax * orig_h / 1000))
                if right <= left or bottom <= top:
                    continue

                # 빈 크롭 필터: 배경만 있는 경우(픽셀 분산 낮음) 건너뜀
                crop_img = full_img.crop((left, top, right, bottom)).convert("RGB")
                if np.array(crop_img).std() < 12:
                    continue

                crop_path = os.path.join(job_dir, f"pic_p{page_num}_{i}_{int(time.time() * 1000)}.jpg")
                crop_img.save(crop_path, "JPEG", quality=92)
                steps.append({
                    "step": s.get("step_number") or (i + 1),
                    "title": f"STEP {s.get('step_number') or (i + 1)}",
                    "desc": desc,
                    "image_url": f"/outputs/{os.path.relpath(crop_path, OUTPUT_DIR)}".replace("\\", "/"),
                    "_page": page_num,
                    "_pos": i,
                })
        return steps
    except Exception as e:
        print(f"🚨 그림형 페이지 분석 에러 (page {page_num}): {e}")
        return []


def _crop_col_slice(page_img_path: str, x0: int, x1: int, ytop_1000: int, ybot_1000, job_dir: str, tag: str) -> str | None:
    """메인 STEP 이미지를 '자기 열 슬라이스'로 크롭한다.
    x0,x1=열 가로 px / ytop_1000=시작 y(0~1000) / ybot_1000=끝 y(0~1000, None이면 끝까지).
    Gemini의 들쭉날쭉한 메인 박스 대신 열 전체 슬라이스를 써서 하위셀을 모두 포함시킨다."""
    try:
        import numpy as np
        with Image.open(page_img_path) as im:
            w, h = im.size
            rgb = im.convert("RGB")
        x0, x1 = max(0, int(x0)), min(w, int(x1))
        top = max(0, int(ytop_1000 * h / 1000) - int(h * 0.01))  # 번호가 잘리지 않게 살짝 위
        bot = h if ybot_1000 is None else min(h, int(ybot_1000 * h / 1000))
        if x1 - x0 < w * 0.04 or bot - top < h * 0.02:
            return None
        region = rgb.crop((x0, top, x1, bot))
        arr = np.array(region)
        # 상하 여백 트림
        rows = np.where(arr.std(axis=(1, 2)) > 12)[0]
        if rows.size == 0:
            return None
        region = region.crop((0, int(rows[0]), region.width, int(rows[-1]) + 1))
        if region.width < 8 or region.height < 8:
            return None
        out = os.path.join(job_dir, f"{tag}_{int(time.time() * 1000)}.jpg")
        region.save(out, "JPEG", quality=92)
        return f"/outputs/{os.path.relpath(out, OUTPUT_DIR)}".replace("\\", "/")
    except Exception as e:
        print(f"🚨 열 슬라이스 크롭 실패 ({tag}): {e}")
        return None


def _clip_box_against(box: list, others: list) -> list:
    """box(0~1000)가 다른 검출 항목(others)의 중심을 포함하지 않도록 우/하단을 안쪽으로 클립.
    Gemini가 한 STEP 박스를 과도하게 크게 그려 오른쪽 열/아래 STEP을 삼키는 것을 방지.
    매뉴얼은 좌→우, 위→아래로 흐르므로 STEP 번호는 박스의 좌상단에 있다고 가정."""
    ymin, xmin, ymax, xmax = box
    inside = []
    for oy0, ox0, oy1, ox1 in others:
        ocx, ocy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
        if xmin < ocx < xmax and ymin < ocy < ymax:
            inside.append((oy0, ox0, oy1, ox1))
    for oy0, ox0, oy1, ox1 in sorted(inside, key=lambda t: (t[1], t[0])):
        ocx, ocy = (ox0 + ox1) / 2, (oy0 + oy1) / 2
        if not (xmin < ocx < xmax and ymin < ocy < ymax):
            continue  # 앞선 클립으로 이미 박스 밖이면 건너뜀
        if ox0 - xmin > 150:      # 항목이 뚜렷이 오른쪽에서 시작 → 다른 열 → 우측 클립
            xmax = min(xmax, ox0)
        elif oy0 - ymin > 80:     # 항목이 아래에서 시작 → 아래 STEP/셀 → 하단 클립
            ymax = min(ymax, oy0)
    if xmax - xmin < 20 or ymax - ymin < 20:
        return box  # 과도 축소 시 원본 유지
    return [ymin, xmin, ymax, xmax]


async def analyze_pdf_page(client, page_img_path: str, page_num: int, job_dir: str) -> list:
    try:
        with Image.open(page_img_path) as img:
            orig_w, orig_h = img.size
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=88)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt = (
            f"당신은 조립 매뉴얼 디지털화 전문가입니다.\n"
            f"이 이미지는 조립 매뉴얼의 {page_num}번째 페이지입니다.\n\n"
            "페이지에서 주요 조립 단계(MAIN STEP)를 모두 찾아 JSON으로 반환하세요.\n\n"
            "[추출 규칙]\n"
            "- step_number: 각 섹션을 구분하는 큰 숫자(섹션 상단/좌측에 단독 표시)만 추출. 정수.\n"
            "- title: \"STEP N\" 형식으로만 작성(N=step_number). 이미지 내용 설명 절대 금지.\n"
            "- desc: 해당 STEP에 표시된 모든 텍스트(본문 + Note/주의사항 포함)를 한 글자도 빠짐없이 한국어로 번역. 요약·생략 절대 금지. 줄바꿈은 공백으로 대체. 텍스트가 전혀 없으면(레고 등 그림 전용) 조립 동작을 한국어로 설명.\n"
            "- box_2d: 해당 메인 STEP 전체 이미지 영역 [ymin,xmin,ymax,xmax] 0~1000 스케일\n\n"
            "[매우 중요] 페이지에 보이는 큰 STEP 번호는 하나도 빠뜨리지 말고 모두 반환하세요. "
            "내용이 적은 STEP(번호 + 그림 1~2개뿐)도 반드시 포함합니다.\n\n"
            "[무시할 것]\n"
            "- 노란색/강조색 인셋 박스 안의 작은 번호(1, 2, 3…) — 부품 픽업 순서이며 메인 단계가 아님\n"
            "- 큰 STEP 번호 없이 그림만 있는 셀(이전 페이지에서 이어지는 연속 부분) — 반환하지 마세요\n"
            "- 'CREATOR 3 in 1', 'N in 1' 같은 제품 배지의 숫자(예: 3 in 1의 3)는 STEP이 아님 — 절대 STEP으로 반환 금지\n"
            "- 표지의 완성 모델 사진, 부품 목록, 로고, 브랜드, 저작권 문구\n\n"
            '{"steps":[{"step_number":int,"title":"STEP N","desc":"설명","box_2d":[int,int,int,int]}]}'
        )

        res = await client.post(
            GEMINI_URL,
            json={
                "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}],
                "generationConfig": _gen_config(),
            },
            timeout=60.0,
        )
        if res.status_code != 200:
            print(f"🚨 페이지 탐지 응답 {res.status_code}: {res.text[:400]}")
            return []

        raw_steps = extract_gemini_steps(res.json())
        all_boxes = [s["box_2d"] for s in raw_steps if s.get("box_2d")]
        # 외국어 desc는 1회 배치 호출로 직역(단계별 직렬 번역 제거)
        descs = await _translate_to_korean_batch(
            client, [s.get("desc", "").strip() or f"STEP {s.get('step_number', '?')} 조립" for s in raw_steps]
        )
        steps = []
        with Image.open(page_img_path) as full_img:
            for idx, s in enumerate(raw_steps):
                box = s.get("box_2d")
                if not box:
                    continue
                desc = descs[idx]
                # 다른 검출 항목을 삼키지 않도록 박스 클립(타이트 크롭)
                others = [b for b in all_boxes if b is not box]
                ymin, xmin, ymax, xmax = _clip_box_against(box, others)
                left = max(0, int(xmin * orig_w / 1000))
                top = max(0, int(ymin * orig_h / 1000))
                right = min(orig_w, int(xmax * orig_w / 1000))
                bottom = min(orig_h, int(ymax * orig_h / 1000))
                if right <= left or bottom <= top:
                    continue

                step_num = s.get("step_number", 0)

                # idx를 파일명에 포함 — 같은 페이지에 같은 번호 셀이 여러 개여도 충돌 방지
                crop_path = os.path.join(job_dir, f"step_p{page_num}_{step_num}_{idx}.jpg")
                full_img.crop((left, top, right, bottom)).convert("RGB").save(crop_path, "JPEG", quality=92)
                step_data = {
                    "step": step_num,
                    "title": f"STEP {step_num}",
                    "desc": desc,
                    "image_url": f"/outputs/{os.path.relpath(crop_path, OUTPUT_DIR)}".replace("\\", "/"),
                    "_box": [ymin, xmin, ymax, xmax],
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
    total_perf = time.perf_counter()
    filename = os.path.basename(img_path)
    try:
        prep_perf = time.perf_counter()
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size
            img_rgb = img.convert("RGB")
            img_rgb.thumbnail((1600, 1600))
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=90)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        _trace_timing("image.prepare", prep_perf, file=filename, size=f"{orig_w}x{orig_h}", kb=round(len(img_b64) * 3 / 4096, 1))

        prompt = """조립/공예 매뉴얼 이미지에서 모든 STEP을 탐지하세요.
- step_number: 정수
- title: 이미지에 레이블이 있으면 그대로, 없으면 "STEP N" 형식으로만 (이미지 내용 설명 절대 금지)
- desc: 텍스트 있으면 원문, 없으면 시각적 동작 한국어 설명
- box_2d: [ymin,xmin,ymax,xmax] 0~1000
{"steps":[{"step_number":int,"title":str,"desc":str,"box_2d":[int,int,int,int]}]}"""

        detect_perf = time.perf_counter()
        res = await client.post(
            GEMINI_URL,
            json={
                "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}],
                "generationConfig": _gen_config(),
            },
            timeout=60.0,
        )
        _trace_timing("image.gemini_detect", detect_perf, file=filename, status=res.status_code)
        if res.status_code != 200:
            print(f"🚨 이미지 탐지 응답 {res.status_code}: {res.text[:400]}")
            return []

        parse_perf = time.perf_counter()
        raw_steps = extract_gemini_steps(res.json())
        _trace_timing("image.parse_steps", parse_perf, file=filename, raw_steps=len(raw_steps))
        # 외국어 desc는 1회 배치 호출로 직역(단계별 직렬 번역 제거)
        translate_perf = time.perf_counter()
        descs = await _translate_to_korean_batch(client, [s.get("desc", "").strip() for s in raw_steps])
        _trace_timing("image.translate", translate_perf, file=filename, descs=len(descs))
        job_dir = os.path.dirname(img_path)
        steps = []
        crop_perf = time.perf_counter()
        with Image.open(img_path) as full_img:
            orig_w, orig_h = full_img.size
            for idx, s in enumerate(raw_steps):
                box = s.get("box_2d")
                desc = descs[idx]
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
                steps.append({
                    "step": step_num,
                    "title": s.get("title", f"STEP {step_num}"),
                    "desc": desc,
                    "image_url": f"/outputs/{os.path.relpath(crop_path, OUTPUT_DIR)}".replace("\\", "/"),
                })
        _trace_timing("image.crop_save", crop_perf, file=filename, steps=len(steps))
        _trace_timing("image.total", total_perf, file=filename, steps=len(steps))
        return steps
    except Exception as e:
        print(f"🚨 이미지 분석 에러: {e}")
    return []


async def _run_picture_pipeline(client, page_imgs: list, job_dir: str) -> list:
    """그림형 파이프라인: 페이지별 4분할 병렬 탐지 → 위치 순 정렬 → 1..N 재번호."""
    results = await asyncio.gather(
        *[analyze_picture_page(client, p, i + 1, job_dir) for i, p in enumerate(page_imgs)]
    )
    steps: list = []
    for page_steps in results:
        steps.extend(page_steps)
    steps.sort(key=lambda s: (s.get("_page", 0), s.get("_pos", 0)))
    for i, s in enumerate(steps):
        s["step"] = i + 1
        s["title"] = f"STEP {i + 1}"
        s.pop("_page", None)
        s.pop("_pos", None)
    return steps


_DETECT_PASSES = 1  # 페이지당 메인 탐지 반복 횟수. 1=단일 검출(빠름·저비용). 2~3=다수결(느림·정확↑).


async def _detect_page_consensus(client, page_img_path: str, page_num: int, job_dir: str) -> list:
    """페이지 메인 STEP 탐지. _DETECT_PASSES>1이면 여러 번 검출해 '다수결'로 확정.
    - 과반 패스에서 나온 STEP만 채택 → 1회성 오검출(표지 배지 등)·누락 완화
    - 박스는 패스들의 '중앙값'으로 → 가끔 튀는 거대/빈 박스 영향 제거"""
    if _DETECT_PASSES <= 1:
        return await analyze_pdf_page(client, page_img_path, page_num, job_dir)  # 단일 검출(다수결 우회)
    passes = await asyncio.gather(
        *[analyze_pdf_page(client, page_img_path, page_num, job_dir) for _ in range(_DETECT_PASSES)]
    )
    by_num: dict = {}
    for ps in passes:
        for s in ps:
            if s.get("_box"):
                by_num.setdefault(s["step"], []).append(s)
    need = _DETECT_PASSES // 2 + 1  # 과반
    merged: list = []
    for num, items in by_num.items():
        if len(items) < need:
            continue  # 소수 패스에서만 나온 건 노이즈로 간주(표지의 '3 in 1' 등)
        med = [int(statistics.median(it["_box"][k] for it in items)) for k in range(4)]
        rep = min(items, key=lambda it: sum(abs(it["_box"][k] - med[k]) for k in range(4)))
        merged.append(dict(rep, _box=med))
    print(f"  🗳️ page{page_num} 다수결 메인: {sorted(m['step'] for m in merged)}")
    return merged


async def _run_text_pipeline(client, page_imgs: list, job_dir: str) -> list:
    """텍스트형 파이프라인: 페이지별 메인 STEP 다수결 탐지 + 단조 증가 필터 + 열 단위 연속 병합."""
    results = await asyncio.gather(
        *[_detect_page_consensus(client, p, i + 1, job_dir) for i, p in enumerate(page_imgs)]
    )
    all_steps: list = []
    last_step = 0
    prev_main = None

    def _read_order(s):
        # 페이지 내 읽기 순서: 열 우선(좌→우) → 같은 열은 위→아래
        b = s.get("_box") or [999, 999, 999, 999]
        return (b[1], b[0])

    for pi, page_steps in enumerate(results):
        page_path = page_imgs[pi]
        w_pg, cl, cr, cols = _detect_columns(page_path)

        # 콘텐츠 영역 왼쪽(여백/표지)에서 잡힌 검출은 STEP이 아님(예: 표지 'CREATOR 3 in 1' 배지) → 제외
        def _in_margin(s):
            b = s.get("_box")
            return bool(b) and ((b[1] + b[3]) / 2) / 1000 * w_pg < cl

        ordered = sorted([s for s in page_steps if not _in_margin(s)], key=_read_order)
        new_mains = [s for s in ordered if s.get("step", 0) > last_step]

        # 메인 STEP이 전혀 없는 페이지 = 통째로 직전 STEP의 연속
        if not new_mains:
            if prev_main is not None:
                url = _crop_x_range(page_path, cl, cr, job_dir, f"p{pi+1}full")
                if url:
                    prev_main.setdefault("extra_images", []).append(url)
                    print(f"  🔗 STEP {prev_main['step']} ← page{pi+1} 전체 연속 병합")
            continue

        # 각 메인 STEP을 박스 왼쪽(xmin) 기준으로 열에 배정.
        # STEP 번호는 박스 좌상단에 있으므로, 박스가 여러 열에 걸쳐도 번호가 있는 열로 배정됨.
        def _col_idx(box):
            b = box or [0, 0, 0, 0]
            x = b[1] / 1000 * w_pg
            for i, (a, bnd) in enumerate(cols):
                if a <= x < bnd:
                    return i
            return 0 if x < cols[0][0] else len(cols) - 1

        mains_by_col: dict = {}
        for m in new_mains:
            mains_by_col.setdefault(_col_idx(m.get("_box")), []).append(m)
        # (메인 STEP 이미지는 analyze_pdf_page의 '클립된 박스 크롭'을 그대로 사용 — 타이트하게 잘림)

        # 열을 좌→우로 보며, 메인이 없는 '빈 열'을 연속으로 귀속:
        #  - 왼쪽에 이 페이지 메인이 있으면 그 메인의 우측 연속
        #  - 없으면(선두 빈 열) 직전 페이지 STEP의 뒷장 연속
        page_owner = None
        for ci in range(len(cols)):
            if ci in mains_by_col:
                page_owner = sorted(mains_by_col[ci], key=_read_order)[-1]
                continue
            x0, x1 = cols[ci]
            url = _crop_x_range(page_path, x0, x1, job_dir, f"p{pi+1}c{ci}")
            if not url:
                continue
            target = page_owner if page_owner is not None else prev_main
            if target is not None:
                target.setdefault("extra_images", []).append(url)
                side = "오른쪽" if page_owner is not None else "왼쪽(앞 페이지)"
                print(f"  🔗 STEP {target['step']} ← page{pi+1} {side} 연속 열 병합")

        # 번호 오름차순으로 추가(거대/중첩 박스로 reading-order가 뒤집혀도 STEP 누락 방지)
        for step in sorted(new_mains, key=lambda s: s.get("step", 0)):
            n = step.get("step", 0)
            if n > last_step:
                step.pop("_box", None)
                all_steps.append(step)
                last_step = n
                prev_main = step
            # else: 단조 증가 위반(중복) → 무시
    return all_steps


async def process_manual_files(files: List[UploadFile], picture_mode: bool = False, thinking_override: str | None = None):
    global _analysis_start_time, _preview_steps, _preview_updated, _active_detect_thinking
    start_time = time.time()
    process_perf = time.perf_counter()
    _analysis_start_time = start_time
    _preview_steps = []
    _preview_updated = False
    state["file_info"] = {"name": "", "pages": 0, "steps": 0}
    state.update(
        {
            "is_analyzed": False,
            "step_locked": False,
            "progress_step": "upload",
            "ai_response": "대기 중...",
            "ai_feedback": "",
            "ai_result": "WAIT",
            "analysis_time": 0.0,
        }
    )
    job_dir = os.path.join(OUTPUT_DIR, f"sess_{int(start_time)}")
    all_steps = []

    if len(files) == 1:
        upload_name = files[0].filename
        cache_perf = time.perf_counter()
        cached_manual = load_saved_manual_by_name(upload_name, exclude_dir=job_dir)
        _trace_timing("manual.cache_lookup", cache_perf, file=upload_name, hit=bool(cached_manual))
        if cached_manual:
            cached_steps = cached_manual["steps"]
            state.update(
                {
                    "manual_steps": cached_steps,
                    "is_analyzed": True,
                    "analysis_time": round(time.time() - start_time, 2),
                    "current_step_idx": 0,
                    "pending_step_idx": None,
                    "pass_hold_until": 0.0,
                    "pass_transition_id": 0,
                    "progress_step": "done",
                    "camera_setup_done": False,
                    "camera_setup_active": False,
                    "camera_setup_phase": "idle",
                    "file_info": {"name": upload_name, "pages": 0, "steps": len(cached_steps)},
                    "uploaded_preview": cached_manual.get("preview", ""),
                }
            )
            save_perf = time.perf_counter()
            save_session()
            _trace_timing("manual.save_session", save_perf)
            print(f"[CACHE] 동일한 매뉴얼 이름 감지 — 저장된 분석 재사용: {upload_name}")
            print(f"[PRELOAD] 동일한 매뉴얼 이름이 이미 저장되어 있어 백그라운드 preload 생략: {upload_name}")
            print(f"✅ [SUCCESS] 분석 완료 — {len(cached_steps)}개 STEP, {state['analysis_time']}s")
            _trace_timing("manual.total", process_perf, steps=len(cached_steps), duplicate=True, cache=True)
            return {"status": "success", "steps": cached_steps}

    os.makedirs(job_dir, exist_ok=True)
    client = get_gemini_http_client()
    for upload in files:
            file_perf = time.perf_counter()
            file_path = os.path.join(job_dir, upload.filename)
            with open(file_path, "wb") as out:
                out.write(await upload.read())
            ext = upload.filename.lower()
            state["file_info"]["name"] = upload.filename
            state["uploaded_preview"] = ""
            try:
                file_size_kb = round(os.path.getsize(file_path) / 1024, 1)
            except OSError:
                file_size_kb = 0
            _trace_timing("manual.file_save", file_perf, file=upload.filename, kb=file_size_kb)

            # 탐지/하위셀 thinking 레벨: 파일 타입 자동 + 요청 override(번역은 항상 low 유지)
            _active_detect_thinking = _decide_thinking(upload.filename, thinking_override)
            print(f"🧠 탐지 thinking={_active_detect_thinking} ({upload.filename})")

            if ext.endswith(".pdf"):
                thumb_perf = time.perf_counter()
                _render_pdf_to_jpegs(file_path, os.path.join(job_dir, "thumb"), 72, first_page=1, last_page=1)
                thumb_files = sorted(glob.glob(os.path.join(job_dir, "thumb-*.jpg")))
                if thumb_files:
                    state["uploaded_preview"] = f"/outputs/{os.path.relpath(thumb_files[0], OUTPUT_DIR)}".replace("\\", "/")
                _trace_timing("manual.pdf_thumb", thumb_perf, file=upload.filename, thumbs=len(thumb_files))
            elif ext.endswith((".png", ".jpg", ".jpeg")):
                state["uploaded_preview"] = f"/outputs/{os.path.relpath(file_path, OUTPUT_DIR)}".replace("\\", "/")

            if ext.endswith(".pdf"):
                pages_dir = os.path.join(job_dir, "pages")
                os.makedirs(pages_dir, exist_ok=True)
                state["progress_step"] = "render"
                loop = asyncio.get_event_loop()
                render_perf = time.perf_counter()
                await loop.run_in_executor(
                    None,
                    lambda: _render_pdf_to_jpegs(file_path, os.path.join(pages_dir, "page"), 200),
                )
                page_imgs = sorted(glob.glob(os.path.join(pages_dir, "page-*.jpg")))
                _trace_timing("manual.pdf_render", render_perf, file=upload.filename, pages=len(page_imgs))
                state["progress_step"] = "analyze"
                if picture_mode:
                    # 명시적 그림형 강제(토글 ON 등)
                    pipeline_perf = time.perf_counter()
                    all_steps.extend(await _run_picture_pipeline(client, page_imgs, job_dir))
                    _trace_timing("manual.picture_pipeline", pipeline_perf, file=upload.filename, total_steps=len(all_steps))
                else:
                    # 표준(텍스트형) 분석을 먼저 시도
                    pipeline_perf = time.perf_counter()
                    text_steps = await _run_text_pipeline(client, page_imgs, job_dir)
                    _trace_timing("manual.text_pipeline", pipeline_perf, file=upload.filename, steps=len(text_steps))
                    if len(text_steps) >= 2:
                        # 하위셀이 여러 개인 STEP은 N-1, N-2 … 로 분할
                        state["progress_step"] = "substep"
                        substep_perf = time.perf_counter()
                        text_steps = await _expand_substeps(client, text_steps, job_dir)
                        _trace_timing("manual.substep_expand", substep_perf, file=upload.filename, steps=len(text_steps))
                        all_steps.extend(text_steps)
                    else:
                        # STEP 번호가 거의 안 잡힘 → 번호 없는 그림형 매뉴얼로 판단, 자동 전환
                        print("ℹ️ STEP 번호 미검출 — 그림형 파이프라인으로 자동 전환")
                        _preview_steps.clear()
                        _preview_updated = True
                        state["manual_steps"] = []
                        state["file_info"]["steps"] = 0
                        fallback_perf = time.perf_counter()
                        all_steps.extend(await _run_picture_pipeline(client, page_imgs, job_dir))
                        _trace_timing("manual.picture_fallback", fallback_perf, file=upload.filename, total_steps=len(all_steps))
            elif ext.endswith((".png", ".jpg", ".jpeg")):
                pipeline_perf = time.perf_counter()
                all_steps.extend(await detect_and_crop_image(client, file_path, len(all_steps)))
                _trace_timing("manual.image_pipeline", pipeline_perf, file=upload.filename, total_steps=len(all_steps))

    postprocess_perf = time.perf_counter()
    unique, seen = [], set()
    for step in all_steps:
        if step["image_url"] not in seen:
            unique.append(step)
            seen.add(step["image_url"])

    def step_num(item):
        value = item.get("step")
        return int(value) if value and str(value).isdigit() else 999

    unique.sort(key=lambda s: (step_num(s), s.get("sub", 0)))

    filtered = []
    for i, step in enumerate(unique):
        is_boundary = i == 0 or i == len(unique) - 1
        title = step.get("title", "").lower()
        desc = step.get("desc", "").lower()
        if is_boundary and any(keyword in title or keyword in desc for keyword in ["안내", "steam", "teaching", "copyright"]):
            if step_num(step) in (0, 999):
                continue
        filtered.append(step)
    _trace_timing("manual.postprocess", postprocess_perf, raw_steps=len(all_steps), unique=len(unique), filtered=len(filtered))

    json_perf = time.perf_counter()
    with open(os.path.join(job_dir, "instruction.json"), "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=4)
    _trace_timing("manual.write_instruction", json_perf, steps=len(filtered))

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
            "camera_setup_done": False,
            "camera_setup_active": False,
            "camera_setup_phase": "idle",
        }
    )
    duplicate_perf = time.perf_counter()
    duplicate_manual_name = saved_manual_name_exists(
        state["file_info"].get("name", ""),
        exclude_dir=job_dir,
    )
    _trace_timing("manual.duplicate_check", duplicate_perf, duplicate=duplicate_manual_name)
    save_perf = time.perf_counter()
    save_session()
    _trace_timing("manual.save_session", save_perf)
    if duplicate_manual_name:
        print(
            f"[PRELOAD] 동일한 매뉴얼 이름이 이미 저장되어 있어 "
            f"백그라운드 preload 생략: {state['file_info'].get('name', '')}"
        )
    elif PRELOAD_MANUAL_STEPS and RUNPOD_INFERENCE_BASE_URL and filtered:
        REGISTERED_STEP_IDS.clear()
        WARMED_STEP_IDS.clear()
        asyncio.create_task(preload_steps_to_gpu(filtered))
        print(f"🚀 [PRELOAD] 백그라운드 등록 시작 — {len(filtered)}개 STEP")

    print(f"✅ [SUCCESS] 분석 완료 — {len(filtered)}개 STEP, {state['analysis_time']}s")
    _trace_timing("manual.total", process_perf, steps=len(filtered), duplicate=duplicate_manual_name)
    return {"status": "success", "steps": filtered}
