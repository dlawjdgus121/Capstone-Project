import asyncio
import os
import re
import socket
import threading
import time
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from config import ESP32_IP, LIVEKIT_URL, MOBILE_TOKEN, PC_TOKEN, UDP_PORT, VLM_JPEG_QUALITY
from manual import get_elapsed_time, preview_event_stream, process_manual_files, reset_preview_state
from state import SESSION_FILE, clear_frame_state, clear_vlm_timing, hw_state, save_session, state
from vlm import (
    REGISTERED_STEP_IDS,
    WARMED_STEP_IDS,
    reset_pass_transition,
    run_manual_vlm_analysis,
)

try:
    import edge_tts
    _EDGE_TTS_OK = True
except ImportError:
    _EDGE_TTS_OK = False
    print("⚠️ edge-tts 없음 — pip install edge-tts")

def query_esp32_servo_angle() -> dict:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(0.5)
        sock.sendto(b"L", (ESP32_IP, UDP_PORT))
        data, addr = sock.recvfrom(128)

    msg = data.decode("utf-8", errors="ignore").strip()
    match = re.match(r"^A([-+]?\d+(?:[.]\d+)?)T([-+]?\d+(?:[.]\d+)?)(?:S([01]))?$", msg)
    if not match:
        raise ValueError(f"unexpected ESP32 servo response: {msg!r}")

    return {
        "current_pan": round(float(match.group(1)), 1),
        "current_tilt": round(float(match.group(2)), 1),
        "is_servo_attached": match.group(3) != "0",
        "raw": msg,
        "addr": addr[0],
    }


def register_routes(app: FastAPI) -> None:
    @app.post("/ping")
    async def ping_check(body: dict):
        return {"pong": True, "client_time": body.get("client_time", 0)}

    @app.get("/preview-steps")
    async def preview_steps():
        return StreamingResponse(
            preview_event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/reset-preview")
    async def reset_preview():
        reset_preview_state()
        return {"status": "ok"}

    @app.get("/config")
    async def get_config():
        return JSONResponse(
            content={
                "livekit_url": LIVEKIT_URL,
                "mobile_token": MOBILE_TOKEN or "",
                "pc_token": PC_TOKEN or "",
                "mobile_url": os.getenv("MOBILE_URL", ""),
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/process-manual")
    async def handle_manual(files: List[UploadFile] = File(...), picture: str = Form("false")):
        return await process_manual_files(files, picture_mode=(picture.lower() == "true"))

    @app.post("/set-step")
    async def set_step(body: dict):
        total = len(state["manual_steps"])
        if total == 0:
            return {"status": "error", "message": "매뉴얼 없음"}
        idx = max(0, min(int(body.get("idx", state["current_step_idx"])), total - 1))
        locked = bool(body.get("locked", True))
        # locked=True(수동 제어)이거나 실제로 스텝이 바뀔 때만 pass transition 취소 및 AI 상태 초기화.
        # locked=False + 동일 idx는 PASS 후 클라이언트가 AI auto 모드를 유지하는 신호이므로 건드리지 않음.
        if locked or idx != state["current_step_idx"]:
            reset_pass_transition()
            state["ai_result"] = "WAIT"
            state["ai_response"] = "대기 중..."
            clear_vlm_timing()
        state["current_step_idx"] = idx
        state["step_locked"] = locked
        save_session()
        return {"status": "ok", "current_step_idx": idx}

    @app.post("/servo-angle")
    async def show_servo_angle():
        try:
            servo = await asyncio.to_thread(query_esp32_servo_angle)
        except Exception as e:
            print(f"[SERVO] ESP32 angle request failed: {type(e).__name__}: {e}")
            return {"status": "error", "message": str(e)}

        pan = servo["current_pan"]
        tilt = servo["current_tilt"]
        attached = servo["is_servo_attached"]
        hw_state["current_pan"] = pan
        hw_state["current_tilt"] = tilt
        hw_state["smooth_pan"] = pan
        hw_state["smooth_tilt"] = tilt
        smooth_pan = pan
        smooth_tilt = tilt
        active = attached
        print(
            f"[SERVO] current pan={pan:.1f}°, tilt={tilt:.1f}° "
            f"| smooth pan={smooth_pan:.1f}°, tilt={smooth_tilt:.1f}° "
            f"| tracking={'ON' if active else 'OFF'}"
        )
        return {
            "status": "ok",
            "current_pan": pan,
            "current_tilt": tilt,
            "smooth_pan": smooth_pan,
            "smooth_tilt": smooth_tilt,
            "is_servo_active": active,
        }

    @app.get("/status")
    async def get_status():
        return {
            "manual_steps": state["manual_steps"],
            "is_analyzed": state["is_analyzed"],
            "current_step_idx": state["current_step_idx"],
            "pending_step_idx": state.get("pending_step_idx"),
            "pass_hold_remaining_s": max(0.0, round(state.get("pass_hold_until", 0.0) - time.time(), 2)),
            "ai_response": state["ai_response"],
            "ai_result": state["ai_result"],
            "analysis_time": state["analysis_time"],
            "progress_step": state["progress_step"],
            "step_locked": state["step_locked"],
            "current_fps": state["current_fps"],
            "stream_level": state["stream_level"],
            "last_rtt": state["last_rtt"],
            "recv_frame_size": state["recv_frame_size"],
            "stream_frame_size": state["stream_frame_size"],
            "vlm_frame_size": state["vlm_frame_size"],
            "hand_frame_size": state["hand_frame_size"],
            "stream_camera_kb": state["stream_camera_kb"],
            "vlm_jpeg_quality": VLM_JPEG_QUALITY,
            "vlm_latency_ms": state["vlm_latency_ms"],
            "vlm_inference_ms": state["vlm_inference_ms"],
            "vlm_proxy_ms": state["vlm_proxy_ms"],
            "vlm_total_s": state["vlm_total_s"],
            "vlm_proxy_overhead_ms": state["vlm_proxy_overhead_ms"],
            "vlm_predict_rtt_ms": state["vlm_predict_rtt_ms"],
            "vlm_transport_ms": state["vlm_transport_ms"],
            "vlm_image_send_ms": state["vlm_image_send_ms"],
            "vlm_result_recv_ms": state["vlm_result_recv_ms"],
            "vlm_e2e_ms": state["vlm_e2e_ms"],
            "vlm_set_step_ms": state["vlm_set_step_ms"],
            "vlm_pre_predict_ms": state["vlm_pre_predict_ms"],
            "vlm_http_client_get_ms": state["vlm_http_client_get_ms"],
            "vlm_camera_kb": state["vlm_camera_kb"],
            "vlm_call_total_ms": state["vlm_call_total_ms"],
            "log": state["log"],
            "has_frame": state["latest_frame"] is not None,
            "has_vlm_frame": state["latest_vlm_frame"] is not None,
            "file_info": state["file_info"],
            "uploaded_preview": state["uploaded_preview"],
            "elapsed_time": get_elapsed_time(),
            "auto_infer_enabled": state.get("auto_infer_enabled", False),
            "gesture": state.get("gesture", "NONE"),
            "gesture_holding_active": state.get("gesture_holding_active", False),
            "gesture_hold_elapsed": state.get("gesture_hold_elapsed", 0.0),
            "gesture_hold_required": state.get("gesture_hold_required", 1.5),
            "gesture_hold_progress": state.get("gesture_hold_progress", 0.0),
            "tracking_active": state.get("tracking_active", True),
            "stepper_state": state.get("stepper_state", "STOP"),
            "servo_pan": round(float(hw_state.get("current_pan", 0.0)), 1),
            "servo_tilt": round(float(hw_state.get("current_tilt", 0.0)), 1),
            "servo_active": bool(hw_state.get("is_servo_active", False)),
            "camera_setup_active": state.get("camera_setup_active", False),
            "camera_setup_phase": state.get("camera_setup_phase", "idle"),
            "camera_setup_done": state.get("camera_setup_done", False),
            "camera_setup_message": state.get("camera_setup_message", ""),
            "camera_setup_countdown": state.get("camera_setup_countdown", 0),
            "camera_setup_countdown_started_at": state.get("camera_setup_countdown_started_at", 0.0),
            "camera_setup_shoulder_y": state.get("camera_setup_shoulder_y"),
            "camera_setup_shoulder_line_y": state.get("camera_setup_shoulder_line_y", 0.48),
        }

    @app.post("/reset")
    async def reset_session():
        REGISTERED_STEP_IDS.clear()
        WARMED_STEP_IDS.clear()
        reset_pass_transition()
        state.update(
            {
                "manual_steps": [],
                "current_step_idx": 0,
                "pending_step_idx": None,
                "pass_hold_until": 0.0,
                "ai_response": "대기 중.",
                "ai_result": "WAIT",
                "is_analyzed": False,
                "analysis_time": 0.0,
                "step_locked": False,
                "progress_step": "upload",
                "file_info": {"name": "", "pages": 0, "steps": 0},
                "uploaded_preview": "",
                "gesture": "NONE",
                "gesture_holding_active": False,
                "gesture_hold_elapsed": 0.0,
                "gesture_hold_required": 1.5,
                "gesture_hold_progress": 0.0,
                "tracking_active": True,
                "stepper_state": "STOP",
                "camera_setup_active": False,
                "camera_setup_phase": "idle",
                "camera_setup_done": False,
                "camera_setup_message": "",
                "camera_setup_countdown": 0,
                "camera_setup_countdown_started_at": 0.0,
                "camera_setup_shoulder_y": None,
                "camera_setup_shoulder_line_y": 0.48,
            }
        )
        clear_vlm_timing()
        clear_frame_state()
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)
        return {"status": "ok"}

    @app.post("/set-auto-infer")
    async def set_auto_infer(body: dict):
        state["auto_infer_enabled"] = bool(body.get("enabled", False))
        return {"status": "ok", "auto_infer_enabled": state["auto_infer_enabled"]}

    @app.post("/start-camera-setup")
    async def start_camera_setup():
        state["camera_setup_active"] = True
        state["camera_setup_phase"] = "linear"
        state["camera_setup_done"] = False
        state["camera_setup_message"] = "어깨 위치를 찾는 중"
        state["camera_setup_message"] = "초기설정을 시작합니다"
        state["camera_setup_countdown"] = 0
        state["camera_setup_countdown_started_at"] = 0.0
        state["camera_setup_shoulder_y"] = None
        state["camera_setup_shoulder_line_y"] = 0.48
        return {"status": "ok"}

    @app.post("/stop-stepper")
    async def stop_stepper():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(b"S", (ESP32_IP, UDP_PORT))
        hw_state["current_stepper_state"] = "STOP"
        state["stepper_state"] = "STOP"
        state["camera_setup_active"] = False
        state["camera_setup_phase"] = "idle"
        state["camera_setup_message"] = "리니어슬라이드 정지"
        print("[HW] STEPPER STOP by keyboard")
        return {"status": "ok"}

    @app.post("/shutdown")
    async def shutdown():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(b"S", (ESP32_IP, UDP_PORT))
        hw_state["current_stepper_state"] = "STOP"
        hw_state["is_servo_active"] = False
        state["stepper_state"] = "STOP"
        state["tracking_active"] = False
        state["camera_setup_active"] = False
        state["camera_setup_phase"] = "idle"
        state["camera_setup_message"] = "브라우저 종료 - 시스템 종료"
        print("[SYS] shutdown request received — stepper stop and tracking disabled")
        threading.Timer(0.25, lambda: os._exit(0)).start()
        return {"status": "ok", "message": "shutdown initiated"}

    @app.post("/reset-camera-setup")
    async def reset_camera_setup():
        state["camera_setup_active"] = False
        state["camera_setup_phase"] = "idle"
        state["camera_setup_message"] = ""
        return {"status": "ok", "message": "camera setup reset"}

    @app.post("/trigger-vlm")
    async def trigger_vlm_analysis():
        try:
            return await run_manual_vlm_analysis()

            if not state.get("is_analyzed") or not state.get("manual_steps"):
                return {"status": "error", "message": "매뉴얼 준비 안 됨"}

            vlm_frame = state.get("latest_vlm_frame") or state.get("latest_frame")
            if not vlm_frame:
                return {"status": "error", "message": "카메라 프레임 없음"}

            manual_steps = state.get("manual_steps", [])
            total = len(manual_steps)
            if total <= 0:
                return {"status": "error", "message": "STEP 목록 없음"}

            idx = int(state.get("current_step_idx", 0) or 0)
            if idx < 0:
                idx = 0
            if idx >= total:
                idx = total - 1
                state["current_step_idx"] = idx

            current_step = manual_steps[idx]
            if not isinstance(current_step, dict):
                return {"status": "error", "message": "현재 STEP 형식 오류"}

            image_url = current_step.get("image_url")
            if not image_url:
                return {"status": "error", "message": "현재 STEP 이미지 없음"}

            desc = current_step.get("desc", "")
            start = asyncio.get_event_loop().time()
            prediction = await call_runpod_inference(image_url, vlm_frame, desc)
            duration = round(asyncio.get_event_loop().time() - start, 2)
            state["vlm_total_s"] = duration

            if prediction and prediction.get("result") != "ERROR":
                result = prediction.get("result", "UNKNOWN")
                reason = prediction.get("reason", "분석 완료")
                apply_vlm_timing(prediction.get("_timing", {}))

                e2e_s = state["vlm_total_s"] or duration
                state["ai_response"] = f"[{result}] {reason} ({e2e_s:.2f}s)"
                state["ai_result"] = result
                if result == "PASS" and not state.get("step_locked") and idx + 1 < len(state["manual_steps"]):
                    state["ai_response"] = f"[{result}] {reason} - 3초 후 다음 단계로 이동합니다. ({e2e_s:.2f}s)"
                    schedule_pass_transition(idx)
                print(f"⏱️ [VLM] {duration}s | {result}")

                return {
                    "status": "success",
                    "prediction": prediction,
                    "current_step": state["current_step_idx"],
                    "duration": duration,
                    "timing": {
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
                    },
                }

            err = prediction.get("reason", "응답 없음") if prediction else "응답 없음"
            print(f"🚨 [VLM ERROR] {err}")
            return {"status": "error", "message": f"VLM 실패: {err}"}
        except Exception as e:
            import traceback

            traceback.print_exc()
            return {"status": "error", "message": f"trigger-vlm 예외: {type(e).__name__}: {str(e)}"}

    @app.get("/tts")
    async def text_to_speech(text: str, voice: str = "ko-KR-InJoonNeural"):
        """edge-tts로 텍스트를 음성 스트림으로 변환 (실패 시 클라이언트가 Web Speech로 폴백)"""
        if not _EDGE_TTS_OK:
            return {"error": "edge-tts 미설치"}
        if not text or not text.strip():
            return {"error": "텍스트 없음"}
        clean = re.sub(r'^\[\w+\]\s*', '', text.strip())
        if not clean:
            return {"error": "빈 텍스트"}
        try:
            communicate = edge_tts.Communicate(clean, voice)

            async def audio_stream():
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        yield chunk["data"]

            return StreamingResponse(
                audio_stream(),
                media_type="audio/mpeg",
                headers={"Cache-Control": "no-cache"},
            )
        except Exception as e:
            return {"error": str(e)}

    @app.get("/")
    @app.get("/mobile")
    async def serve_ui():
        if os.path.exists("index.html"):
            return FileResponse("index.html")
        return {"error": "index.html not found"}
