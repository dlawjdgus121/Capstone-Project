import asyncio
import hashlib
import io
import os
import re
import time

import httpx

from config import (
    BASE_DIR,
    PRELOAD_WARMUP,
    RUNPOD_HTTP_TIMEOUT,
    RUNPOD_INFERENCE_BASE_URL,
    RUNPOD_PREDICT_URL,
    RUNPOD_SET_STEP_URL,
)
from state import save_session, state

REGISTERED_STEP_IDS = set()
WARMED_STEP_IDS = set()
PASS_HOLD_SECONDS = 3.0
AUTO_VLM_INTERVAL_SECONDS = 5.0
VLM_ANALYSIS_LOCK = asyncio.Lock()
MANUAL_VLM_PENDING = False
last_vlm_step_key = None
RUNPOD_HTTP_CLIENT = None


def get_runpod_http_client() -> httpx.AsyncClient:
    global RUNPOD_HTTP_CLIENT
    if RUNPOD_HTTP_CLIENT is None or RUNPOD_HTTP_CLIENT.is_closed:
        RUNPOD_HTTP_CLIENT = httpx.AsyncClient(timeout=RUNPOD_HTTP_TIMEOUT)
    return RUNPOD_HTTP_CLIENT


def ensure_runpod_http_client() -> None:
    global RUNPOD_HTTP_CLIENT
    if RUNPOD_INFERENCE_BASE_URL and (RUNPOD_HTTP_CLIENT is None or RUNPOD_HTTP_CLIENT.is_closed):
        RUNPOD_HTTP_CLIENT = httpx.AsyncClient(timeout=RUNPOD_HTTP_TIMEOUT)
        print("✅ [RUNPOD HTTP] AsyncClient 재사용 모드 시작", flush=True)


async def close_runpod_http_client() -> None:
    global RUNPOD_HTTP_CLIENT
    if RUNPOD_HTTP_CLIENT is not None and not RUNPOD_HTTP_CLIENT.is_closed:
        await RUNPOD_HTTP_CLIENT.aclose()
        RUNPOD_HTTP_CLIENT = None
        print("✅ [RUNPOD HTTP] AsyncClient 종료", flush=True)


def normalize_lora_path(value: str | None) -> str:
    # Single-adapter mode: adapter selection is owned by the serving/proxy side.
    # Do not forward per-manual or per-step LoRA names from the web server.
    return ""


def get_step_lora_path(step: dict) -> str:
    return ""


def make_vlm_step_id(manual_img_path: str, step_desc: str = "", lora_path: str = "") -> str:
    raw = f"{manual_img_path}|{step_desc}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def parse_verdict_output(text: str) -> tuple[str, str, str]:
    text = (text or "").strip()
    if not text:
        return "", "", ""

    verdict_text = text
    match = re.search(r"VERDICT\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if match:
        verdict_text = match.group(1).strip()

    parts = [part.strip() for part in verdict_text.split("||")]
    status = ""
    reason = ""
    feedback = ""

    if parts:
        status_match = re.search(r"\b(PASS|FAIL)\b", parts[0], re.IGNORECASE)
        if status_match:
            status = status_match.group(1).upper()

    if len(parts) >= 2:
        reason = parts[1]
    elif len(parts) == 1:
        reason = re.sub(r"\b(PASS|FAIL)\b", "", parts[0], flags=re.IGNORECASE).strip(" \n\t:-[]")

    if len(parts) >= 3:
        feedback = " || ".join(parts[2:]).strip()

    return status, reason, feedback


def compact_vlm_text(text: str, max_len: int = 90) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) > max_len:
        return text[:max_len].rstrip() + "..."
    return text


async def prepare_step_prefix(step: dict, step_index=None, warmup: bool = True, force: bool = False, client=None) -> bool:
    if not RUNPOD_INFERENCE_BASE_URL:
        print("[PRELOAD] RUNPOD_INFERENCE_URL missing; skip prefix prepare")
        return False
    if not isinstance(step, dict):
        print("[PRELOAD] invalid step payload; skip prefix prepare")
        return False

    manual_img_path = step.get("image_url", "")
    step_desc = step.get("desc", "")
    step_label = f"STEP {step_index + 1}" if step_index is not None else "STEP"
    if not manual_img_path:
        print(f"[PRELOAD] {step_label}: image_url missing")
        return False

    full_manual_path = os.path.join(BASE_DIR, manual_img_path.lstrip("/"))
    if not os.path.exists(full_manual_path):
        print(f"[PRELOAD] {step_label}: image missing: {full_manual_path}")
        return False

    lora_path = get_step_lora_path(step)
    step_id = make_vlm_step_id(manual_img_path, step_desc, lora_path)
    if not force:
        if warmup and step_id in WARMED_STEP_IDS:
            print(f"[PRELOAD] {step_label}: prefix already warmed step_id={step_id}")
            return True
        if not warmup and step_id in REGISTERED_STEP_IDS:
            print(f"[PRELOAD] {step_label}: already registered step_id={step_id}")
            return True

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0))

    try:
        with open(full_manual_path, "rb") as f:
            resp = await client.post(
                RUNPOD_SET_STEP_URL,
                files={"manual_image": ("manual.jpg", f, "image/jpeg")},
                data={
                    "step_id": step_id,
                    "warmup": "true" if warmup else "false",
                    **({"lora_path": lora_path} if lora_path else {}),
                },
            )

        if resp.status_code != 200:
            print(f"[PRELOAD] {step_label}: HTTP {resp.status_code}: {resp.text[:300]}")
            return False

        payload = resp.json()
        if payload.get("status") != "success":
            print(f"[PRELOAD] {step_label}: failed payload={payload}")
            return False

        REGISTERED_STEP_IDS.add(step_id)
        if payload.get("warmed"):
            WARMED_STEP_IDS.add(step_id)

        print(
            f"[PRELOAD] {step_label}: step_id={step_id} "
            f"lora_path={lora_path or 'base'} "
            f"warmed={payload.get('warmed')} elapsed={payload.get('elapsed_ms')}ms"
        )
        return True
    except Exception as e:
        print(f"[PRELOAD] {step_label}: exception: {e}")
        return False
    finally:
        if owns_client:
            await client.aclose()


async def preload_steps_to_gpu(steps: list):
    if not RUNPOD_INFERENCE_BASE_URL:
        print("⚠️ [PRELOAD] RUNPOD_INFERENCE_URL 없음 — preload 생략")
        return
    if not steps:
        print("⚠️ [PRELOAD] step 없음 — preload 생략")
        return

    print(f"🚀 [PRELOAD] GPU 서버에 {len(steps)}개 STEP 등록 시작")
    ok_count = 0
    fail_count = 0
    timeout = httpx.Timeout(180.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        for i, step in enumerate(steps):
            manual_img_path = step.get("image_url", "")
            step_desc = step.get("desc", "")
            if not manual_img_path:
                fail_count += 1
                print(f"⚠️ [PRELOAD] STEP {i + 1}: image_url 없음")
                continue

            full_manual_path = os.path.join(BASE_DIR, manual_img_path.lstrip("/"))
            if not os.path.exists(full_manual_path):
                fail_count += 1
                print(f"⚠️ [PRELOAD] STEP {i + 1}: 이미지 없음: {full_manual_path}")
                continue

            lora_path = get_step_lora_path(step)
            step_id = make_vlm_step_id(manual_img_path, step_desc, lora_path)
            try:
                with open(full_manual_path, "rb") as f:
                    resp = await client.post(
                        RUNPOD_SET_STEP_URL,
                        files={"manual_image": ("manual.jpg", f, "image/jpeg")},
                        data={
                            "step_id": step_id,
                            "warmup": "true" if PRELOAD_WARMUP else "false",
                            **({"lora_path": lora_path} if lora_path else {}),
                        },
                    )

                if resp.status_code == 200:
                    payload = resp.json()
                    if payload.get("status") == "success":
                        REGISTERED_STEP_IDS.add(step_id)
                        if payload.get("warmed"):
                            WARMED_STEP_IDS.add(step_id)
                        ok_count += 1
                        print(
                            f"✅ [PRELOAD] {i + 1}/{len(steps)} step_id={step_id} "
                            f"lora_path={lora_path or 'base'} "
                            f"warmed={payload.get('warmed')} elapsed={payload.get('elapsed_ms')}ms"
                        )
                    else:
                        fail_count += 1
                        print(f"🚨 [PRELOAD] STEP {i + 1} 실패 payload={payload}")
                else:
                    fail_count += 1
                    print(f"🚨 [PRELOAD] STEP {i + 1} HTTP {resp.status_code}: {resp.text[:300]}")
            except Exception as e:
                fail_count += 1
                print(f"🚨 [PRELOAD] STEP {i + 1} 예외: {e}")

            await asyncio.sleep(0.05)

    print(f"✅ [PRELOAD] 완료: success={ok_count}, fail={fail_count}")


def reset_pass_transition() -> None:
    state["pending_step_idx"] = None
    state["pass_hold_until"] = 0.0
    state["pass_transition_id"] = int(state.get("pass_transition_id", 0) or 0) + 1


def schedule_pass_transition(current_idx: int, force: bool = False) -> bool:
    manual_steps = state.get("manual_steps") or []
    next_idx = current_idx + 1
    if next_idx >= len(manual_steps):
        return False

    now = time.time()
    if state.get("pending_step_idx") == next_idx and state.get("pass_hold_until", 0.0) > now:
        return True

    token = int(state.get("pass_transition_id", 0) or 0) + 1
    state["pass_transition_id"] = token
    state["pending_step_idx"] = next_idx
    state["pass_hold_until"] = now + PASS_HOLD_SECONDS

    next_step = manual_steps[next_idx]
    asyncio.create_task(prepare_step_prefix(next_step, next_idx, warmup=True))
    asyncio.create_task(_advance_after_pass_hold(token, current_idx, next_idx, force))
    print(f"[PASS] holding STEP {current_idx + 1} for {PASS_HOLD_SECONDS:.1f}s; warming STEP {next_idx + 1}")
    return True


async def _advance_after_pass_hold(token: int, current_idx: int, next_idx: int, force: bool = False) -> None:
    await asyncio.sleep(PASS_HOLD_SECONDS)
    if state.get("pass_transition_id") != token:
        return
    if state.get("current_step_idx") != current_idx:
        reset_pass_transition()
        return
    if state.get("pending_step_idx") != next_idx:
        return
    if state.get("step_locked") and not force:
        reset_pass_transition()
        return

    # ai_response 먼저 초기화 → 클라이언트가 이 poll에서 AI 응답 TTS 발동
    state["ai_result"] = "WAIT"
    state["ai_response"] = "잘 하셨습니다. 다음 단계로 넘어갑니다."
    state["ai_feedback"] = ""
    state["pending_step_idx"] = None
    state["pass_hold_until"] = 0.0

    # 1.2초 대기 → 다음 poll에서 스텝 전진 TTS만 발동 (겹침 방지)
    await asyncio.sleep(1.2)

    state["current_step_idx"] = next_idx
    if force:
        state["step_locked"] = False
    state["ai_response"] = ""
    state["ai_feedback"] = ""
    save_session()
    print(f"[PASS] advanced to STEP {next_idx + 1}")


async def call_runpod_inference(manual_img_path, camera_frame_bytes, step_desc="", lora_path=""):
    global last_vlm_step_key

    call_start_perf = time.perf_counter()
    set_step_ms = 0.0
    set_step_called = False
    predict_rtt_ms = 0.0
    pre_predict_ms = 0.0
    client_get_ms = 0.0
    camera_bytes_len = len(camera_frame_bytes or b"")
    camera_kb = round(camera_bytes_len / 1024.0, 2)

    def elapsed_ms(start_perf: float) -> float:
        return round((time.perf_counter() - start_perf) * 1000, 2)

    def timing_float(value, default: float = 0.0) -> float:
        try:
            return float(value if value is not None else default)
        except (TypeError, ValueError):
            return default

    def wall_delta_ms(start_wall, end_wall) -> float:
        start = timing_float(start_wall, 0.0)
        end = timing_float(end_wall, 0.0)
        if start <= 0.0 or end <= 0.0 or end < start:
            return 0.0
        return round((end - start) * 1000, 2)

    if not RUNPOD_INFERENCE_BASE_URL:
        return {"result": "WAIT", "reason": "카메라 연결 대기 중..."}

    try:
        full_manual_path = os.path.join(BASE_DIR, manual_img_path.lstrip("/"))
        if not os.path.exists(full_manual_path):
            return {"result": "ERROR", "reason": "이미지 없음"}

        lora_path = normalize_lora_path(lora_path or state.get("manual_lora", ""))
        step_id = make_vlm_step_id(manual_img_path, step_desc, lora_path)
        client_start_perf = time.perf_counter()
        client = get_runpod_http_client()
        client_get_ms = elapsed_ms(client_start_perf)

        if last_vlm_step_key != step_id and step_id not in REGISTERED_STEP_IDS:
            set_step_called = True
            set_step_start_perf = time.perf_counter()
            with open(full_manual_path, "rb") as f:
                set_resp = await client.post(
                    RUNPOD_SET_STEP_URL,
                    files={"manual_image": ("manual.jpg", f, "image/jpeg")},
                    data={"step_id": step_id, "warmup": "false", **({"lora_path": lora_path} if lora_path else {})},
                )
            set_step_ms = elapsed_ms(set_step_start_perf)

            if set_resp.status_code != 200:
                body_preview = set_resp.text[:300].replace("\n", " ")
                print(
                    f"🚨 [MAIN->GPU SET_STEP ERROR] step_id={step_id} "
                    f"http={set_resp.status_code} rtt={set_step_ms}ms | {body_preview}",
                    flush=True,
                )
                print(f"🚨 [SET_STEP URL] {RUNPOD_SET_STEP_URL}", flush=True)
                return {"result": "ERROR", "reason": f"스텝 등록 실패: {set_resp.status_code}"}

            set_payload = set_resp.json()
            print(
                f"⏱️ [MAIN->GPU SET_STEP] step_id={step_id} "
                f"lora_path={lora_path or 'base'} rtt={set_step_ms}ms "
                f"gpu_elapsed={set_payload.get('elapsed_ms')}ms warmed={set_payload.get('warmed')}",
                flush=True,
            )

            if set_payload.get("status") != "success":
                return {"result": "ERROR", "reason": f"스텝 등록 오류: {set_payload.get('message', 'unknown')}"}

            REGISTERED_STEP_IDS.add(step_id)
            if set_payload.get("warmed"):
                WARMED_STEP_IDS.add(step_id)

        last_vlm_step_key = step_id

        pre_predict_ms = round(max(0.0, elapsed_ms(call_start_perf) - set_step_ms), 2)
        request_prepare_start_perf = time.perf_counter()
        camera_upload = ("camera.jpg", io.BytesIO(camera_frame_bytes), "image/jpeg")
        predict_data = {
            "step_id": step_id,
            "web_send_time": "",
            **({"lora_path": lora_path} if lora_path else {}),
        }
        request_prepare_ms = elapsed_ms(request_prepare_start_perf)
        predict_start_perf = time.perf_counter()
        web_send_time = time.time()
        predict_data["web_send_time"] = str(web_send_time)
        pred_resp = await client.post(
            RUNPOD_PREDICT_URL,
            files={"camera_image": camera_upload},
            data=predict_data,
        )
        response_recv_wall = time.time()
        predict_rtt_ms = elapsed_ms(predict_start_perf)

        if pred_resp.status_code == 200:
            response_parse_start_perf = time.perf_counter()
            payload = pred_resp.json()
            response_parse_ms = elapsed_ms(response_parse_start_perf)

            pred = payload.get("prediction", {"result": "UNKNOWN", "reason": "분석 오류"})
            if not isinstance(pred, dict):
                pred = {"result": "UNKNOWN", "reason": str(pred)}
            raw_output = str(payload.get("raw_output") or pred.get("raw_output") or "")
            raw_result, raw_reason, raw_feedback = parse_verdict_output(raw_output)
            reason_result, reason_text, reason_feedback = parse_verdict_output(str(pred.get("reason", "")))
            explicit_feedback = str(pred.get("feedback") or "").strip()
            if raw_result:
                pred["result"] = raw_result
            elif reason_result and pred.get("result") in ("", "UNKNOWN", None):
                pred["result"] = reason_result
            if raw_reason:
                pred["reason"] = raw_reason
            elif reason_text:
                pred["reason"] = reason_text
            pred["feedback"] = raw_feedback or explicit_feedback or reason_feedback
            if raw_output:
                pred["raw_output"] = raw_output
            timing = payload.get("timing", {})
            if not isinstance(timing, dict):
                timing = {}

            gpu_vlm_ms = timing_float(timing.get("vlm_inference_ms", timing.get("sglang_latency_ms", 0.0)))
            gpu_proxy_total_ms = timing_float(timing.get("proxy_total_ms", timing.get("e2e_proxy_ms", 0.0)))
            gpu_proxy_overhead_ms = timing_float(timing.get("proxy_overhead_ms", timing.get("proxy_processing_ms", 0.0)))
            proxy_step_prepare_ms = timing_float(timing.get("step_prepare_ms", 0.0))
            proxy_camera_read_ms = timing_float(timing.get("camera_read_ms", 0.0))
            proxy_base64_ms = timing_float(timing.get("base64_ms", 0.0))
            proxy_payload_build_ms = timing_float(timing.get("payload_build_ms", 0.0))
            proxy_parse_ms = timing_float(timing.get("parse_ms", 0.0))
            proxy_pre_vlm_ms = round(
                proxy_step_prepare_ms
                + proxy_camera_read_ms
                + proxy_base64_ms
                + proxy_payload_build_ms,
                2,
            )
            transport_overhead_ms = round(max(0.0, predict_rtt_ms - gpu_proxy_total_ms), 2)
            total_call_ms = elapsed_ms(call_start_perf)
            image_send_ms = wall_delta_ms(timing.get("web_send_time", web_send_time), timing.get("proxy_recv_wall"))
            result_recv_ms = wall_delta_ms(timing.get("proxy_send_wall"), response_recv_wall)
            send_estimated = False
            if transport_overhead_ms > 0.0 and (
                image_send_ms <= 0.0
                or result_recv_ms <= 0.0
                or image_send_ms + result_recv_ms > predict_rtt_ms
            ):
                send_estimated = True
                image_send_ms = round(transport_overhead_ms * 0.8, 2)
                result_recv_ms = round(transport_overhead_ms - image_send_ms, 2)

            timing.update(
                {
                    "main_lora_path": lora_path,
                    "main_set_step_called": set_step_called,
                    "main_set_step_ms": set_step_ms,
                    "main_pre_predict_ms": pre_predict_ms,
                    "main_http_client_get_ms": client_get_ms,
                    "main_request_prepare_ms": request_prepare_ms,
                    "main_camera_bytes": camera_bytes_len,
                    "main_camera_kb": camera_kb,
                    "main_predict_rtt_ms": predict_rtt_ms,
                    "main_response_recv_wall": response_recv_wall,
                    "main_response_parse_ms": response_parse_ms,
                    "main_transport_overhead_ms": transport_overhead_ms,
                    "main_image_send_ms": image_send_ms,
                    "main_result_recv_ms": result_recv_ms,
                    "main_send_estimated": send_estimated,
                    "main_e2e_ms": total_call_ms,
                    "main_total_call_ms": total_call_ms,
                }
            )
            pred["_timing"] = timing

            _C = "\033[0m"       # reset
            _Y = "\033[93m"      # yellow — timing
            _D = "\033[90m"      # dark   — detail
            send_label = "SEND~" if send_estimated else "SEND"
            print(
                f"{_Y}⏱️  {send_label}:{image_send_ms:.0f}ms │ VLM:{gpu_vlm_ms:.0f}ms │ "
                f"RECV:{result_recv_ms:.0f}ms │ E2E:{total_call_ms:.0f}ms{_C}",
                flush=True,
            )
            print(
                f"{_D}   SEND detail prep={request_prepare_ms:.1f}ms | "
                f"wire+ingress+multipart={image_send_ms:.0f}ms | "
                f"proxy_pre_vlm={proxy_pre_vlm_ms:.1f}ms "
                f"(step={proxy_step_prepare_ms:.1f}, read={proxy_camera_read_ms:.1f}, "
                f"b64={proxy_base64_ms:.1f}, payload={proxy_payload_build_ms:.1f}) | "
                f"proxy_parse={proxy_parse_ms:.1f}ms | proxy_overhead={gpu_proxy_overhead_ms:.1f}ms | "
                f"send_estimated={'yes' if send_estimated else 'no'}{_C}",
                flush=True,
            )

            pred["reason"] = compact_vlm_text(pred.get("reason", ""), 90)
            pred["feedback"] = compact_vlm_text(pred.get("feedback", ""), 110)

            return pred

        body_preview = pred_resp.text[:300].replace("\n", " ")
        print(
            f"🚨 [MAIN->GPU PREDICT ERROR] step_id={step_id} http={pred_resp.status_code} "
            f"rtt={predict_rtt_ms}ms pre_predict={pre_predict_ms}ms camera={camera_kb}KB | {body_preview}",
            flush=True,
        )
        print(f"🚨 [PREDICT URL] {RUNPOD_PREDICT_URL}", flush=True)
        return {"result": "ERROR", "reason": f"서버 오류: {pred_resp.status_code}"}
    except Exception as e:
        total_call_ms = elapsed_ms(call_start_perf)
        print(
            f"🚨 [MAIN TIMING ERROR] total_call={total_call_ms}ms predict_rtt={predict_rtt_ms}ms "
            f"set_step_rtt={set_step_ms}ms pre_predict={pre_predict_ms}ms "
            f"client_get={client_get_ms}ms camera={camera_kb}KB error={e}",
            flush=True,
        )
        return {"result": "ERROR", "reason": f"통신 장애: {str(e)}"}


def apply_vlm_timing(timing: dict) -> None:
    state["vlm_latency_ms"] = float(timing.get("vlm_inference_ms", timing.get("sglang_latency_ms", 0.0)) or 0.0)
    state["vlm_inference_ms"] = state["vlm_latency_ms"]
    state["vlm_proxy_ms"] = float(timing.get("proxy_total_ms", timing.get("e2e_proxy_ms", 0.0)) or 0.0)
    state["vlm_proxy_overhead_ms"] = float(timing.get("proxy_overhead_ms", 0.0) or 0.0)
    state["vlm_predict_rtt_ms"] = float(timing.get("main_predict_rtt_ms", 0.0) or 0.0)
    state["vlm_transport_ms"] = float(timing.get("main_transport_overhead_ms", 0.0) or 0.0)
    state["vlm_image_send_ms"] = float(timing.get("main_image_send_ms", 0.0) or 0.0)
    state["vlm_result_recv_ms"] = float(timing.get("main_result_recv_ms", 0.0) or 0.0)
    state["vlm_e2e_ms"] = float(timing.get("main_e2e_ms", timing.get("main_total_call_ms", 0.0)) or 0.0)
    state["vlm_set_step_ms"] = float(timing.get("main_set_step_ms", 0.0) or 0.0)
    state["vlm_pre_predict_ms"] = float(timing.get("main_pre_predict_ms", 0.0) or 0.0)
    state["vlm_http_client_get_ms"] = float(timing.get("main_http_client_get_ms", 0.0) or 0.0)
    state["vlm_camera_kb"] = float(timing.get("main_camera_kb", 0.0) or 0.0)
    state["vlm_call_total_ms"] = float(timing.get("main_total_call_ms", 0.0) or 0.0)
    if state["vlm_e2e_ms"] > 0:
        state["vlm_total_s"] = round(state["vlm_e2e_ms"] / 1000.0, 2)


def _vlm_timing_response() -> dict:
    return {
        "vlm_latency_ms": state.get("vlm_latency_ms", 0.0),
        "vlm_inference_ms": state.get("vlm_inference_ms", 0.0),
        "vlm_proxy_ms": state.get("vlm_proxy_ms", 0.0),
        "vlm_total_s": state.get("vlm_total_s", 0.0),
        "vlm_proxy_overhead_ms": state.get("vlm_proxy_overhead_ms", 0.0),
        "vlm_predict_rtt_ms": state.get("vlm_predict_rtt_ms", 0.0),
        "vlm_transport_ms": state.get("vlm_transport_ms", 0.0),
        "vlm_image_send_ms": state.get("vlm_image_send_ms", 0.0),
        "vlm_result_recv_ms": state.get("vlm_result_recv_ms", 0.0),
        "vlm_e2e_ms": state.get("vlm_e2e_ms", 0.0),
        "vlm_set_step_ms": state.get("vlm_set_step_ms", 0.0),
        "vlm_pre_predict_ms": state.get("vlm_pre_predict_ms", 0.0),
        "vlm_http_client_get_ms": state.get("vlm_http_client_get_ms", 0.0),
        "vlm_camera_kb": state.get("vlm_camera_kb", 0.0),
        "vlm_call_total_ms": state.get("vlm_call_total_ms", 0.0),
    }


async def run_vlm_analysis_once(source: str = "manual") -> dict:
    if not state.get("is_analyzed") or not state.get("manual_steps"):
        return {"status": "error", "message": "매뉴얼 준비 안 됨", "source": source}

    vlm_frame = state.get("latest_vlm_frame") or state.get("latest_frame")
    if not vlm_frame:
        return {"status": "error", "message": "카메라 프레임 없음", "source": source}

    manual_steps = state.get("manual_steps", [])
    total = len(manual_steps)
    if total <= 0:
        return {"status": "error", "message": "STEP 목록 없음", "source": source}

    idx = int(state.get("current_step_idx", 0) or 0)
    if idx < 0:
        idx = 0
    if idx >= total:
        idx = total - 1
        state["current_step_idx"] = idx

    current_step = manual_steps[idx]
    if not isinstance(current_step, dict):
        return {"status": "error", "message": "현재 STEP 형식 오류", "source": source}

    image_url = current_step.get("image_url")
    if not image_url:
        return {"status": "error", "message": "현재 STEP 이미지 없음", "source": source}

    desc = current_step.get("desc", "")
    lora_path = get_step_lora_path(current_step)
    start = time.perf_counter()
    state["is_processing"] = True
    try:
        prediction = await call_runpod_inference(image_url, vlm_frame, desc, lora_path)
    finally:
        state["is_processing"] = False

    duration = round(time.perf_counter() - start, 2)
    state["vlm_total_s"] = duration
    if source == "auto" and MANUAL_VLM_PENDING:
        return {"status": "skipped", "reason": "manual_pending_after_call", "source": source}

    if prediction and prediction.get("result") != "ERROR":
        result = prediction.get("result", "UNKNOWN")
        reason = prediction.get("reason", "분석 완료")
        feedback = prediction.get("feedback", "")
        apply_vlm_timing(prediction.get("_timing", {}))

        e2e_ms = float(state.get("vlm_e2e_ms", 0.0) or 0.0)
        e2e_s = (e2e_ms / 1000.0) if e2e_ms > 0 else duration
        e2e_label = f" ({e2e_s:.1f}s)" if e2e_s > 0 else ""
        state["ai_response"] = f"[{result}] {reason}{e2e_label}"
        state["ai_feedback"] = feedback
        state["ai_result"] = result
        if result == "PASS" and idx + 1 < len(state["manual_steps"]):
            state["ai_response"] = f"[{result}] {reason}{e2e_label} - 3초 후 다음 단계로 이동합니다."
            schedule_pass_transition(idx, force=True)
        print(f"[VLM:{source}] {duration}s | {result}")

        return {
            "status": "success",
            "source": source,
            "prediction": prediction,
            "current_step": state["current_step_idx"],
            "duration": duration,
            "timing": _vlm_timing_response(),
        }

    err = prediction.get("reason", "응답 없음") if prediction else "응답 없음"
    state["ai_feedback"] = ""
    print(f"[VLM:{source} ERROR] {err}")
    return {"status": "error", "message": f"VLM 실패: {err}", "source": source}


async def run_manual_vlm_analysis() -> dict:
    global MANUAL_VLM_PENDING
    MANUAL_VLM_PENDING = True
    try:
        async with VLM_ANALYSIS_LOCK:
            return await run_vlm_analysis_once("manual")
    finally:
        MANUAL_VLM_PENDING = False


async def run_auto_vlm_analysis() -> dict:
    if MANUAL_VLM_PENDING or VLM_ANALYSIS_LOCK.locked():
        return {"status": "skipped", "reason": "manual_or_busy"}
    if state.get("pending_step_idx") is not None and state.get("pass_hold_until", 0.0) > time.time():
        return {"status": "skipped", "reason": "pass_hold"}

    async with VLM_ANALYSIS_LOCK:
        if MANUAL_VLM_PENDING:
            return {"status": "skipped", "reason": "manual_pending"}
        return await run_vlm_analysis_once("auto")


async def coaching_loop():
    while True:
        if not state.get("latest_frame") and state["ai_result"] not in ("WAIT",):
            state["ai_result"] = "WAIT"
            state["ai_response"] = "모바일 카메라를 연결해 주세요."
            state["ai_feedback"] = ""
        try:
            interval_s = float(state.get("auto_infer_interval_s", AUTO_VLM_INTERVAL_SECONDS))
        except (TypeError, ValueError):
            interval_s = AUTO_VLM_INTERVAL_SECONDS
        interval_s = max(1.0, min(30.0, interval_s))
        await asyncio.sleep(interval_s)
        if state.get("auto_infer_enabled", False):
            await run_auto_vlm_analysis()
        continue

        vlm_frame = state.get("latest_vlm_frame") or state.get("latest_frame")
        if state["is_analyzed"] and vlm_frame and state["manual_steps"]:
            idx = state["current_step_idx"]
            if idx < len(state["manual_steps"]):
                current_step = state["manual_steps"][idx]
                prediction = await call_runpod_inference(
                    current_step["image_url"],
                    vlm_frame,
                    current_step.get("desc", ""),
                    get_step_lora_path(current_step),
                )
                if prediction:
                    result = prediction.get("result", "UNKNOWN")
                    reason = prediction.get("reason", "분석 중...")
                    feedback = prediction.get("feedback", "")
                    apply_vlm_timing(prediction.get("_timing", {}))

                    e2e_ms = float(state.get("vlm_e2e_ms", 0.0) or 0.0)
                    e2e_s = (e2e_ms / 1000.0) if e2e_ms > 0 else float(state.get("vlm_total_s", 0.0) or 0.0)
                    e2e_label = f" ({e2e_s:.1f}s)" if e2e_s > 0 else ""
                    if result == "PASS" and not state["step_locked"] and idx + 1 < len(state["manual_steps"]):
                        state["ai_response"] = f"[{result}] {reason}{e2e_label} - 3초 후 다음 단계로 이동합니다."
                        state["ai_feedback"] = feedback
                        state["ai_result"] = result
                        schedule_pass_transition(idx)
                    else:
                        state["ai_response"] = f"[{result}] {reason}{e2e_label}"
                        state["ai_feedback"] = feedback
                        state["ai_result"] = result
            await asyncio.sleep(3.0)
        else:
            if not state["latest_frame"] and state["ai_result"] not in ("WAIT",):
                state["ai_result"] = "WAIT"
                state["ai_response"] = "모바일 카메라를 연결해 주세요."
                state["ai_feedback"] = ""
            await asyncio.sleep(1.0)
