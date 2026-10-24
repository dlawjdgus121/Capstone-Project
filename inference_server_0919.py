import os
import re
import time
import base64
import hashlib
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Optional, Dict, Any, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header, Depends


# ============================================================
# 1. SGLang 서버 / API 설정
# ============================================================
SGLANG_BASE_URL = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000")
SGLANG_MODEL = os.getenv("SGLANG_MODEL", "qwen3.6-27b")

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
TIMEOUT_KEEP_ALIVE = int(os.getenv("TIMEOUT_KEEP_ALIVE", "120"))

SGLANG_TIMEOUT = float(os.getenv("SGLANG_TIMEOUT", "120"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))

# 빠른 검사 목적이면 thinking off
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "0") == "1"

# /set_step 호출 시 prefix warmup을 수행할지 여부
WARMUP_ON_SET_STEP = os.getenv("WARMUP_ON_SET_STEP", "1") == "1"

# raw 모델 응답을 API 응답에 포함할지 여부
RETURN_RAW = os.getenv("RETURN_RAW", "1") == "1"

# Single-adapter mode: the served SGLang model already owns the adapter choice.
# Keep incoming lora_path form fields backward-compatible, but ignore them.
LORA_MODE = "single_adapter"
DEFAULT_LORA_PATH = ""

# 공개 RunPod 프록시로 노출되므로 API 키가 설정돼 있으면 X-API-Key 헤더를 강제한다.
# 비어 있으면(미설정) 기존과 동일하게 인증 없이 열려 있다.
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")

# STEP_CACHE에 무제한으로 이미지가 쌓이는 것을 막기 위한 상한
MAX_STEP_CACHE_SIZE = int(os.getenv("MAX_STEP_CACHE_SIZE", "200"))


async def require_api_key(x_api_key: str = Header(default="", alias="X-API-Key")) -> None:
    if INFERENCE_API_KEY and x_api_key != INFERENCE_API_KEY:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


# ============================================================
# 2. 프롬프트
# ============================================================
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """
You are a strict visual grading AI.
Compare the reference manual image and the current camera image against the user's instruction.
You must output exactly this format and nothing else:
VERDICT: [PASS 또는 FAIL] || [이유 1문장] || [짧은 피드백 1문장]
""".strip(),
)

DEFAULT_STEP_PROMPT = os.getenv(
    "DEFAULT_STEP_PROMPT",
    "사용자의 조립 화면을 보고 매뉴얼 단계와 일치하는지 판별해줘.",
)


def build_step_instruction(raw_prompt: str) -> str:
    """
    이 문자열은 manual image 뒤, camera image 앞에 들어간다.
    prefix caching을 위해 같은 step에서는 이 문자열이 매번 완전히 같아야 한다.
    """
    raw_prompt = (raw_prompt or DEFAULT_STEP_PROMPT).strip()

    return f"""
[입력 순서]
1. 이 메시지의 첫 번째 이미지는 기준 매뉴얼/스텝 이미지다.
2. 이 메시지의 두 번째 이미지는 사용자의 현재 카메라 이미지다.

[검사 지시문]
{raw_prompt}

[판정 기준]
- 현재 카메라 이미지가 기준 매뉴얼 이미지의 조립 단계와 일치하면 PASS다.
- 부품 방향, 위치, 체결 여부, 누락, 순서가 다르면 FAIL이다.
- 애매하거나 확신이 낮으면 FAIL이다.

[출력 형식]
반드시 아래 형식만 출력한다.
VERDICT: [PASS 또는 FAIL] || [이유 1문장] || [짧은 피드백 1문장]
""".strip()


# ============================================================
# 3. Step prefix cache 상태
# ============================================================
@dataclass
class StepPrefix:
    step_id: str
    image_sha256: str
    image_data_url: str
    raw_prompt: str
    instruction: str
    updated_at: float


# 쓰기 순서를 보존하는 OrderedDict라 가장 오래된 항목부터 popitem(last=False)로 정리 가능
STEP_CACHE: "OrderedDict[str, StepPrefix]" = OrderedDict()
STEP_LOCK = asyncio.Lock()

app = FastAPI(title="Qwen3.6-27B SGLang VLM Proxy")


# ============================================================
# 4. 유틸
# ============================================================
def guess_mime(upload: UploadFile) -> str:
    content_type = (upload.content_type or "").lower()
    if content_type.startswith("image/"):
        return content_type

    filename = (upload.filename or "").lower()
    if filename.endswith(".png"):
        return "image/png"
    if filename.endswith(".webp"):
        return "image/webp"
    if filename.endswith(".bmp"):
        return "image/bmp"

    return "image/jpeg"


def bytes_to_data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def normalize_lora_path(lora_path: Optional[str]) -> str:
    return ""


def parse_response(text: str) -> Tuple[str, str, str]:
    text = (text or "").strip()
    status = "UNKNOWN"
    reason = "파싱 실패 또는 모델 출력 형식 오류"
    feedback = ""

    status_match = re.search(r"\[?\b(PASS|FAIL)\b\]?", text, re.IGNORECASE)
    if status_match:
        status = status_match.group(1).upper()

    parts = [p.strip() for p in text.split("||")]
    if len(parts) >= 3:
        reason = parts[1] or reason
        feedback = parts[2]
    elif len(parts) == 2:
        reason = parts[1] or reason
    else:
        cleaned = re.sub(r"\[?\b(PASS|FAIL)\b\]?", "", text, flags=re.IGNORECASE)
        cleaned = cleaned.replace("||", " ").strip(" \n\t:-")
        if cleaned:
            reason = cleaned[:300]

    return status, reason, feedback


def build_messages(step: StepPrefix, camera_data_url: Optional[str]) -> list:
    """
    prefix caching 구조:
    system prompt -> manual image -> instruction -> camera image

    camera image가 마지막에 오므로, camera image 앞까지가 반복 prefix가 된다.
    """
    content = [
        {
            "type": "image_url",
            "image_url": {"url": step.image_data_url},
        },
        {
            "type": "text",
            "text": step.instruction,
        },
    ]

    if camera_data_url:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": camera_data_url},
            }
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def build_payload(
    step: StepPrefix,
    camera_data_url: Optional[str],
    max_tokens: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": SGLANG_MODEL,
        "messages": build_messages(step, camera_data_url),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }

    payload["chat_template_kwargs"] = {"enable_thinking": ENABLE_THINKING}

    return payload


def extract_output_text(sglang_json: Dict[str, Any]) -> str:
    choices = sglang_json.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}

    # thinking off면 보통 content만 온다.
    content = message.get("content") or ""

    # reasoning만 오는 경우 방어.
    if not content:
        content = message.get("reasoning_content") or ""

    return str(content).strip()


async def call_sglang(payload: Dict[str, Any]) -> Dict[str, Any]:
    client: httpx.AsyncClient = app.state.http_client
    url = f"{SGLANG_BASE_URL.rstrip('/')}/v1/chat/completions"

    resp = await client.post(url, json=payload)
    if resp.status_code >= 400:
        raise RuntimeError(f"SGLang HTTP {resp.status_code}: {resp.text[:1000]}")

    return resp.json()


def _evict_over_capacity_locked() -> None:
    """호출 전 STEP_LOCK을 잡고 있어야 한다. 가장 오래전에 쓰여진 step부터 정리한다."""
    while len(STEP_CACHE) > MAX_STEP_CACHE_SIZE:
        STEP_CACHE.popitem(last=False)


async def upsert_step_from_bytes(
    step_id: str,
    image_bytes: bytes,
    mime: str,
    raw_prompt: str,
    lora_path: str = "",
) -> StepPrefix:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="manual_image is empty")

    image_sha = hashlib.sha256(image_bytes).hexdigest()
    data_url = bytes_to_data_url(image_bytes, mime)
    instruction = build_step_instruction(raw_prompt)
    normalize_lora_path(lora_path)

    step = StepPrefix(
        step_id=step_id,
        image_sha256=image_sha,
        image_data_url=data_url,
        raw_prompt=raw_prompt,
        instruction=instruction,
        updated_at=time.time(),
    )

    async with STEP_LOCK:
        STEP_CACHE[step_id] = step
        STEP_CACHE.move_to_end(step_id)
        _evict_over_capacity_locked()

    return step


async def get_step(step_id: str) -> Optional[StepPrefix]:
    async with STEP_LOCK:
        return STEP_CACHE.get(step_id)


async def get_step_and_upsert(
    step_id: str,
    image_bytes: bytes,
    mime: str,
    prompt_override: Optional[str],
) -> StepPrefix:
    """
    이전 step(raw_prompt 상속용)을 읽는 것과 새 step을 쓰는 것을 하나의 락 구간으로 묶는다.
    분리돼 있으면 그 사이에 다른 요청이 끼어들어 stale raw_prompt로 진행되는 레이스가 생긴다.
    """
    if not image_bytes:
        raise HTTPException(status_code=400, detail="manual_image is empty")

    image_sha = hashlib.sha256(image_bytes).hexdigest()
    data_url = bytes_to_data_url(image_bytes, mime)

    async with STEP_LOCK:
        old = STEP_CACHE.get(step_id)
        raw_prompt = prompt_override or (old.raw_prompt if old else DEFAULT_STEP_PROMPT)

        step = StepPrefix(
            step_id=step_id,
            image_sha256=image_sha,
            image_data_url=data_url,
            raw_prompt=raw_prompt,
            instruction=build_step_instruction(raw_prompt),
            updated_at=time.time(),
        )
        STEP_CACHE[step_id] = step
        STEP_CACHE.move_to_end(step_id)
        _evict_over_capacity_locked()

    return step


async def update_step_instruction(step_id: str, raw_prompt: str) -> StepPrefix:
    async with STEP_LOCK:
        old = STEP_CACHE.get(step_id)
        if old is None:
            raise HTTPException(
                status_code=400,
                detail=f"step_id={step_id!r} is not cached. Call /set_step first or send manual_image in /predict.",
            )

        new_step = replace(
            old,
            raw_prompt=raw_prompt,
            instruction=build_step_instruction(raw_prompt),
            updated_at=time.time(),
        )
        STEP_CACHE[step_id] = new_step
        STEP_CACHE.move_to_end(step_id)
        return new_step


async def warm_prefix(step: StepPrefix) -> Tuple[bool, Optional[str]]:
    """
    step image + instruction까지만 보내서 prefix를 미리 SGLang cache에 올린다.
    실제 /predict에서는 같은 prefix 뒤에 camera image가 붙는다.
    """
    payload = build_payload(step=step, camera_data_url=None, max_tokens=1)

    try:
        await call_sglang(payload)
        return True, None
    except Exception as e:
        return False, str(e)


# ============================================================
# 5. FastAPI lifecycle
# ============================================================
@app.on_event("startup")
async def startup() -> None:
    timeout = httpx.Timeout(SGLANG_TIMEOUT, connect=10.0)
    app.state.http_client = httpx.AsyncClient(timeout=timeout)
    print(
        f"✅ FastAPI proxy started. SGLang={SGLANG_BASE_URL}, model={SGLANG_MODEL}, "
        f"lora_mode={LORA_MODE}, auth={'on' if INFERENCE_API_KEY else 'off'}",
        flush=True,
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.http_client.aclose()


# ============================================================
# 6. API endpoints
# ============================================================
@app.get("/health", dependencies=[Depends(require_api_key)])
async def health() -> Dict[str, Any]:
    try:
        client: httpx.AsyncClient = app.state.http_client
        resp = await client.get(f"{SGLANG_BASE_URL.rstrip('/')}/health")
        sglang_ok = resp.status_code == 200
    except Exception:
        sglang_ok = False

    return {
        "status": "ok",
        "sglang_ok": sglang_ok,
        "sglang_base_url": SGLANG_BASE_URL,
        "model": SGLANG_MODEL,
        "lora_mode": LORA_MODE,
        "cache_size": len(STEP_CACHE),
        "cached_steps": list(STEP_CACHE.keys()),
    }


@app.post("/set_step", dependencies=[Depends(require_api_key)])
async def set_step(
    manual_image: UploadFile = File(...),
    step_id: str = Form("default"),
    prompt: Optional[str] = Form(None),
    warmup: bool = Form(WARMUP_ON_SET_STEP),
    lora_path: str = Form(DEFAULT_LORA_PATH),
) -> Dict[str, Any]:
    """
    권장 사용:
    - 매뉴얼/스텝 이미지가 바뀔 때 한 번 호출
    - 이후 /predict에는 camera_image만 보내기

    이렇게 해야 step image + instruction prefix가 안정적으로 재사용
    """
    t0 = time.time()
    normalized_lora = normalize_lora_path(lora_path)

    print(
        f"\n📥 [GPU SET_STEP RECV] "
        f"step_id={step_id} "
        f"filename={manual_image.filename} "
        f"content_type={manual_image.content_type} "
        f"warmup={warmup} "
        f"lora_mode={LORA_MODE}",
        flush=True,
    )

    raw_prompt = (prompt or DEFAULT_STEP_PROMPT).strip()
    image_bytes = await manual_image.read()
    mime = guess_mime(manual_image)

    step = await upsert_step_from_bytes(
        step_id=step_id,
        image_bytes=image_bytes,
        mime=mime,
        raw_prompt=raw_prompt,
        lora_path=normalized_lora,
    )

    print(
        f"💾 [GPU SET_STEP STORE] "
        f"step_id={step.step_id} "
        f"sha={step.image_sha256[:12]} "
        f"lora_mode={LORA_MODE} "
        f"cache_size={len(STEP_CACHE)} "
        f"elapsed_ms={round((time.time() - t0) * 1000, 2)}",
        flush=True,
    )

    warmed = False
    warmup_error = None
    if warmup:
        warmed, warmup_error = await warm_prefix(step)

    return {
        "status": "success",
        "step_id": step.step_id,
        "image_sha256": step.image_sha256,
        "lora_mode": LORA_MODE,
        "warmed": warmed,
        "warmup_error": warmup_error,
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
    }




@app.post("/predict", dependencies=[Depends(require_api_key)])
async def predict(
    camera_image: UploadFile = File(...),
    manual_image: Optional[UploadFile] = File(None),
    step_id: str = Form("default"),
    prompt: Optional[str] = Form(None),
    web_send_time: str = Form("0.0"),
    lora_path: str = Form(DEFAULT_LORA_PATH),
) -> Dict[str, Any]:
    """
    기존 API와 호환:
    - manual_image + camera_image를 매번 보내도 동작한다.

    추천 고속 API:
    - /set_step으로 manual_image를 먼저 등록
    - /predict에는 camera_image만 전송
    """
    print("\n[DEBUG] --- 요청 수신됨 ---", flush=True)
    t2_wall = time.time()
    t2_proxy_recv_perf = time.perf_counter()

    try:
        normalize_lora_path(lora_path)

        # 1) step/manual image 처리
        t_step_prepare_start_perf = time.perf_counter()
        if manual_image is not None:
            image_bytes = await manual_image.read()
            mime = guess_mime(manual_image)

            prompt_override = prompt.strip() if prompt and prompt.strip() else None
            step = await get_step_and_upsert(
                step_id=step_id,
                image_bytes=image_bytes,
                mime=mime,
                prompt_override=prompt_override,
            )
        else:
            step = await get_step(step_id)
            if step is None:
                print(
                    f"❌ [GPU PREDICT CACHE MISS] "
                    f"step_id={step_id} "
                    f"cache_size={len(STEP_CACHE)} "
                    f"cache_keys={list(STEP_CACHE.keys())[:5]}",
                    flush=True,
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"step_id={step_id!r} is not cached. "
                        "Call /set_step first or send manual_image in /predict."
                    ),
                )
            print(
                f"✅ [GPU PREDICT CACHE HIT] "
                f"step_id={step_id} "
                f"sha={step.image_sha256[:12]} "
                f"lora_mode={LORA_MODE} "
                f"cache_size={len(STEP_CACHE)}",
                flush=True,
            )

            # prompt를 명시적으로 보낸 경우만 instruction 변경
            if prompt and prompt.strip() and prompt.strip() != step.raw_prompt:
                step = await update_step_instruction(step_id, prompt.strip())
        t_step_prepare_end_perf = time.perf_counter()

        # 2) camera image는 항상 dynamic tail
        t_camera_read_start_perf = time.perf_counter()

        camera_bytes = await camera_image.read()
        if not camera_bytes:
            raise HTTPException(status_code=400, detail="camera_image is empty")

        t_camera_read_end_perf = time.perf_counter()

        camera_mime = guess_mime(camera_image)

        t_base64_start_perf = time.perf_counter()
        camera_data_url = bytes_to_data_url(camera_bytes, camera_mime)
        t_base64_end_perf = time.perf_counter()

        # 3) SGLang 추론
        t_payload_start_perf = time.perf_counter()

        payload = build_payload(
            step=step,
            camera_data_url=camera_data_url,
            max_tokens=MAX_NEW_TOKENS,
        )
        t_payload_end_perf = time.perf_counter()

        t_sglang_start_perf = time.perf_counter()
        sglang_json = await call_sglang(payload)
        t_sglang_end_perf = time.perf_counter()

        t_parse_start_perf = time.perf_counter()

        output_text = extract_output_text(sglang_json)

        if not output_text.strip():
            print("⚠️ [WARNING] 모델이 빈 문자열을 반환했습니다!", flush=True)
        else:
            print(f"🤖 [VLM RAW]: {output_text}", flush=True)

        pred_status, pred_reason, pred_feedback = parse_response(output_text)
        print(f"✅ [RESULT]: {pred_status} | {pred_reason}", flush=True)

        t_parse_end_perf = time.perf_counter()

        t3_proxy_send_perf = time.perf_counter()
        t3_wall = time.time()

        step_prepare_ms = round(
            (t_step_prepare_end_perf - t_step_prepare_start_perf) * 1000,
            2,
        )

        camera_read_ms = round(
            (t_camera_read_end_perf - t_camera_read_start_perf) * 1000,
            2,
        )

        base64_ms = round(
            (t_base64_end_perf - t_base64_start_perf) * 1000,
            2,
        )

        payload_build_ms = round(
            (t_payload_end_perf - t_payload_start_perf) * 1000,
            2,
        )

        sglang_latency_ms = round(
            (t_sglang_end_perf - t_sglang_start_perf) * 1000,
            2,
        )

        parse_ms = round(
            (t_parse_end_perf - t_parse_start_perf) * 1000,
            2,
        )

        proxy_total_s = t3_proxy_send_perf - t2_proxy_recv_perf
        sglang_latency_s = t_sglang_end_perf - t_sglang_start_perf
        proxy_overhead_s = max(0.0, proxy_total_s - sglang_latency_s)

        proxy_total_ms = round(proxy_total_s * 1000, 2)
        proxy_overhead_ms = round(proxy_overhead_s * 1000, 2)

        proxy_non_vlm_components_ms = round(
            step_prepare_ms
            + camera_read_ms
            + base64_ms
            + payload_build_ms
            + parse_ms,
            2,
        )

        print(
            "⏱️ [GPU TIMING] "
            f"step_id={step.step_id} "
            f"lora_mode={LORA_MODE} "
            f"vlm={sglang_latency_ms}ms "
            f"proxy_total={proxy_total_ms}ms "
            f"proxy_overhead={proxy_overhead_ms}ms "
            f"step_prepare={step_prepare_ms}ms "
            f"camera_read={camera_read_ms}ms "
            f"base64={base64_ms}ms "
            f"payload={payload_build_ms}ms "
            f"parse={parse_ms}ms",
            flush=True,
        )

        result = {
            "status": "success",

            # 기존 호환용. 실제 시각 timestamp
            "t2": t2_wall,
            "t3": t3_wall,

            "prediction": {
                "result": pred_status,
                "reason": pred_reason,
                "feedback": pred_feedback,
            },
            "timing": {
                "web_send_time": web_send_time,

                # wall clock timestamps
                "proxy_recv_wall": t2_wall,
                "proxy_send_wall": t3_wall,

                # duration measurements
                "step_prepare_ms": step_prepare_ms,
                "camera_read_ms": camera_read_ms,
                "base64_ms": base64_ms,
                "payload_build_ms": payload_build_ms,
                "sglang_latency_ms": sglang_latency_ms,
                "vlm_inference_ms": sglang_latency_ms,
                "parse_ms": parse_ms,

                # proxy_overhead_ms/proxy_processing_ms는 VLM 호출 시간을 제외한 proxy 자체 처리 시간
                "proxy_overhead_ms": proxy_overhead_ms,
                "proxy_processing_ms": proxy_overhead_ms,
                "proxy_non_vlm_components_ms": proxy_non_vlm_components_ms,
                # proxy_total_ms/e2e_proxy_ms는 VLM 호출 시간을 포함한 proxy handler 전체 시간
                "proxy_total_ms": proxy_total_ms,
                "e2e_proxy_ms": proxy_total_ms,
            },
            "step": {
                "step_id": step.step_id,
                "image_sha256": step.image_sha256,
                "lora_mode": LORA_MODE,
            },
            "usage": sglang_json.get("usage"),
        }

        if RETURN_RAW:
            result["raw_output"] = output_text

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"🚨 [ERROR]: {str(e)}", flush=True)
        # 예전엔 200 + {"status":"error"}를 반환해서 클라이언트가 실패를 "성공"으로
        # 오인했다(prediction 키가 없으니 UNKNOWN으로 채워짐). 이제 진짜 에러 상태코드로 알린다.
        raise HTTPException(status_code=502, detail=str(e))


if __name__ == "__main__":
    # 기본값 5s는 클라이언트 자동 추론 간격(5s)과 겹쳐 매 호출 TCP+TLS 재수립을
    # 유발했다(순 transport 438ms -> 1214ms). 간격보다 넉넉히 길게 잡는다.
    uvicorn.run(app, host=APP_HOST, port=APP_PORT, timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
