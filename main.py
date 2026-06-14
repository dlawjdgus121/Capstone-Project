import asyncio
import os
import re as _re
import socket
import subprocess
import threading
import time
from contextlib import asynccontextmanager, suppress

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import CLOUDFLARED_BIN, ESP32_IP, OUTPUT_DIR, PRELOAD_MANUAL_STEPS, RUNPOD_INFERENCE_BASE_URL, UDP_PORT
from routes import register_routes
from state import hw_state, load_session, state
from vlm import (
    REGISTERED_STEP_IDS,
    WARMED_STEP_IDS,
    close_runpod_http_client,
    coaching_loop,
    ensure_runpod_http_client,
    preload_steps_to_gpu,
)


def align_servo_to_center() -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(b"P90.0T90.0", (ESP32_IP, UDP_PORT))
        hw_state["current_pan"] = 90.0
        hw_state["current_tilt"] = 90.0
        hw_state["smooth_pan"] = 90.0
        hw_state["smooth_tilt"] = 90.0
        print("[SERVO] startup align pan=90.0deg, tilt=90.0deg")
    except Exception as e:
        print(f"[SERVO] startup align failed: {type(e).__name__}: {e}")


def start_udp_listener():
    def _listen():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(("0.0.0.0", UDP_PORT))
            print(f"🎧 [UDP] ESP32 상태 수신 리스너 시작 (Port: {UDP_PORT})")
            while True:
                try:
                    data, addr = sock.recvfrom(1024)
                    msg = data.decode('utf-8').strip()
                    
                    if msg == "LIMIT_TOP":
                        if state.get("return_home_active", False):
                            state["hardware_notice_message"] = "카메라 원점 복귀가 끝났습니다"
                            state["hardware_notice_until"] = time.time() + 4.0
                            state["return_home_active"] = False
                            state["return_home_done"] = True
                            state["camera_setup_message"] = "카메라 원점 복귀가 끝났습니다"
                        else:
                            state["hardware_notice_message"] = ""
                            state["hardware_notice_until"] = 0.0
                        state["hardware_limit_gesture"] = "LIMIT_TOP"
                        state["hardware_limit_until"] = time.time() + 2.0
                        state["stepper_state"] = "STOP"
                        state["gesture"] = "NONE"
                        state["gesture_holding_active"] = False
                        state["gesture_hold_elapsed"] = 0.0
                        state["gesture_hold_progress"] = 0.0
                        hw_state["current_stepper_state"] = "STOP"
                        print("🚨 [ESP32] 최상단 리미트 스위치 도달! 상승 강제 정지")
                    elif msg == "LIMIT_BOTTOM":
                        if state.get("return_home_active", False):
                            state["hardware_notice_message"] = "카메라 원점 복귀가 끝났습니다"
                            state["hardware_notice_until"] = time.time() + 4.0
                            state["return_home_active"] = False
                            state["return_home_done"] = True
                            state["camera_setup_message"] = "카메라 원점 복귀가 끝났습니다"
                        else:
                            state["hardware_notice_message"] = ""
                            state["hardware_notice_until"] = 0.0
                        state["hardware_limit_gesture"] = "LIMIT_BOTTOM"
                        state["hardware_limit_until"] = time.time() + 2.0
                        state["stepper_state"] = "STOP"
                        state["gesture"] = "NONE"
                        state["gesture_holding_active"] = False
                        state["gesture_hold_elapsed"] = 0.0
                        state["gesture_hold_progress"] = 0.0
                        hw_state["current_stepper_state"] = "STOP"
                        print("🚨 [ESP32] 최하단 리미트 스위치 도달! 하강 강제 정지")
                    elif msg.startswith("LOG:"):
                        # 🔍 아두이노가 보낸 로그 메시지를 파이썬 터미널에 출력
                        print(f"📟 [ESP32 LOG] {msg[4:]}") 
                except Exception:
                    pass
    threading.Thread(target=_listen, daemon=True).start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_udp_listener() # 🔍 앱 시작 시 리스너 실행 추가
    restored = load_session()
    ensure_runpod_http_client()
    align_servo_to_center()

    if restored and PRELOAD_MANUAL_STEPS and RUNPOD_INFERENCE_BASE_URL and state.get("manual_steps"):
        REGISTERED_STEP_IDS.clear()
        WARMED_STEP_IDS.clear()
        asyncio.create_task(preload_steps_to_gpu(state["manual_steps"]))
        print(f"🚀 [PRELOAD] 세션 복원 후 GPU 서버 등록 시작 — {len(state['manual_steps'])}개 STEP")

    if CLOUDFLARED_BIN and not os.getenv("MOBILE_URL"):
        def run_tunnel():
            try:
                proc = subprocess.Popen(
                    [CLOUDFLARED_BIN, "tunnel", "--url", "http://localhost:8100"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for line in proc.stderr:
                    line = line.strip()
                    if "trycloudflare" in line.lower() or "tunnel" in line.lower():
                        print(f"🔍 [CF 로그] {line}")
                    match = _re.search(r"https://[a-z0-9-]+[.]trycloudflare[.]com", line)
                    if match:
                        os.environ["MOBILE_URL"] = match.group(0)
                        print(f"✅ [Cloudflare] 터널 시작: {match.group(0)}")
                        break
            except Exception as e:
                print(f"⚠️ [Cloudflare] 터널 실패: {e}")

        threading.Thread(target=run_tunnel, daemon=True).start()
        for _ in range(20):
            await asyncio.sleep(0.5)
            if os.getenv("MOBILE_URL"):
                break
        if not os.getenv("MOBILE_URL"):
            print("⚠️ [Cloudflare] URL 감지 실패 — QR은 현재 URL 기반으로 생성됩니다")
    elif not CLOUDFLARED_BIN:
        print("ℹ️ [Cloudflare] cloudflared 바이너리 없음 — 터널 미사용")

    auto_vlm_task = asyncio.create_task(coaching_loop())
    print("⏱️ [VLM] 자동 추론 루프 시작 — 5초 주기")

    livekit_task = None
    try:
        from livekit_client import run_livekit

        livekit_task = asyncio.create_task(run_livekit())
    except Exception as e:
        print(f"⚠️ [LiveKit] 시작 실패: {type(e).__name__}: {e}")
    try:
        yield
    finally:
        if livekit_task:
            livekit_task.cancel()
            with suppress(asyncio.CancelledError):
                await livekit_task
        auto_vlm_task.cancel()
        with suppress(asyncio.CancelledError):
            await auto_vlm_task
        await close_runpod_http_client()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
register_routes(app)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100, access_log=False)
