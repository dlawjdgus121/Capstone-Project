import io
import os
import json
import time
import base64
import asyncio
import httpx
import uvicorn
import glob
import subprocess
import cv2
import numpy as np
import socket
import math
import mediapipe as mp
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from dotenv import load_dotenv

from livekit import rtc

load_dotenv()

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import threading, re as _re

def _find_cloudflared():
    """프로젝트 루트 또는 PATH에서 cloudflared 바이너리 찾기"""
    candidates = [
        os.path.join(BASE_DIR, "cloudflared-linux-amd64"),
        os.path.join(BASE_DIR, "cloudflared"),
        "/usr/local/bin/cloudflared",
        "cloudflared",
    ]
    for c in candidates:
        if os.path.isfile(c):
            os.chmod(c, 0o755)
            return c
    return None

CLOUDFLARED_BIN = _find_cloudflared()

try:
    import opendataloader_pdf
except ImportError:
    print("⚠️ [DEBUG] 'pip install opendataloader-pdf' 라이브러리가 필요합니다.")

# ── 환경 변수 ──────────────────────────────────────────────────────────────
API_KEY    = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

raw_runpod_url = os.getenv("RUNPOD_INFERENCE_URL", "").strip()
if raw_runpod_url:
    if not raw_runpod_url.startswith("http"): raw_runpod_url = f"http://{raw_runpod_url}"
    if not raw_runpod_url.endswith("/predict"): raw_runpod_url = raw_runpod_url.rstrip("/") + "/predict"
    RUNPOD_INFERENCE_URL = raw_runpod_url
else:
    RUNPOD_INFERENCE_URL = None

# ── ESP32 하드웨어 설정 ────────────────────────────────────────────────────
ESP32_IP = "192.168.137.182"
UDP_PORT = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

hw_state = {
    "PAN_MIN_LIMIT": 40.0,  "PAN_MAX_LIMIT": 140.0,
    "TILT_MIN_LIMIT": 50.0, "TILT_MAX_LIMIT": 100.0,
    "current_pan": 90.0,    "current_tilt": 90.0,
    "smooth_pan": 90.0,     "smooth_tilt": 90.0,
    "is_servo_active": True,
    "current_stepper_state": "STOP",
}

# ── MediaPipe ──────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    max_num_hands=1, model_complexity=0,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# ── 적응형 스트리밍 레벨 정의 ────────────────────────────────────────────────
STREAM_LEVELS = {
    0: (426,  240, 40, "저화질 240p"),
    1: (640,  360, 55, "중간 360p"),
    2: (960,  540, 70, "고화질 540p"),
    3: (1280, 720, 85, "최고 720p"),
}
FPS_THRESHOLDS = {
    "down": 15,
    "up":   25,
}
ADAPTIVE_WINDOW = 5 

# ── 최신 프레임 버퍼 ────────────────────────────────────────────────────────
latest_raw_frame = None

# ── WebRTC 렌더링(송출)용 전역 소스 및 트랙 생성 ─────────────────────────────
out_video_source = rtc.VideoSource(1280, 720)
out_video_track = rtc.LocalVideoTrack.create_video_track("processed-video", out_video_source)

# ── 매뉴얼 파싱 프리뷰 ──────────────────────────────────────────────────────
import asyncio as _asyncio
_preview_steps: list = []
_preview_updated: bool = False

# ── 시스템 상태 ────────────────────────────────────────────────────────────
state = {
    "latest_frame":   None,
    "manual_steps":   [],
    "current_step_idx": 0,
    "ai_response":    "대기 중...",
    "ai_result":      "WAIT",
    "is_analyzed":    False,
    "analysis_time":  0.0,
    "step_locked":    False,
    "progress_step":  "upload",
    "file_info":      {"name": "", "pages": 0, "steps": 0},
    "uploaded_preview": "",
    "current_fps":    0.0,
    "last_rtt":       0.0,
    "is_processing":  False,
    "stream_level":   2,
    "fps_history":    [],
    "log":            "모바일 연결 대기 중...",
}

SESSION_FILE    = os.path.join(OUTPUT_DIR, "last_session.json")
prev_frame_time = 0.0
frame_count     = 0

# ── LiveKit 설정 ───────────────────────────────────────────────────────────
LIVEKIT_URL   = os.getenv("LIVEKIT_URL",   "wss://capstoneproject-l2ih740k.livekit.cloud")
LIVEKIT_TOKEN = os.getenv("LIVEKIT_TOKEN")   # 서버용 토큰
MOBILE_TOKEN  = os.getenv("MOBILE_TOKEN")    # 모바일용 토큰
PC_TOKEN      = os.getenv("PC_TOKEN")        # PC 브라우저용 토큰 (추가됨)
MOBILE_URL    = os.getenv("MOBILE_URL", "")

def save_session():
    if not state["is_analyzed"] or not state["manual_steps"]: return
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "current_step_idx": state["current_step_idx"],
                "manual_steps":     state["manual_steps"],
                "analysis_time":    state["analysis_time"],
                "step_locked":      state["step_locked"],
            }, f, ensure_ascii=False, indent=2)
    except Exception as e: print(f"⚠️ 세션 저장 실패: {e}")

def load_session() -> bool:
    if not os.path.exists(SESSION_FILE): return False
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        steps = payload.get("manual_steps", [])
        if not steps: return False
        idx = max(0, min(int(payload.get("current_step_idx", 0)), len(steps) - 1))
        state.update({
            "manual_steps": steps, "current_step_idx": idx,
            "analysis_time": payload.get("analysis_time", 0),
            "step_locked": payload.get("step_locked", False),
            "is_analyzed": True, "progress_step": "done",
        })
        print(f"✅ [SESSION] 복원 완료 — STEP {idx+1}/{len(steps)}")
        return True
    except Exception as e:
        print(f"⚠️ 세션 복원 실패: {e}")
        return False

def calculate_servo_angles(hx, hy):
    dist_x, dist_y = hx - 0.5, hy - 0.5
    if abs(dist_x) < 0.15 and abs(dist_y) < 0.15:
        hw_state["current_pan"]  = hw_state["smooth_pan"]
        hw_state["current_tilt"] = hw_state["smooth_tilt"]
    else:
        hw_state["current_pan"]  += dist_x * 4.0
        hw_state["current_tilt"] += dist_y * 4.0
    hw_state["current_pan"]  = max(hw_state["PAN_MIN_LIMIT"],  min(hw_state["PAN_MAX_LIMIT"],  hw_state["current_pan"]))
    hw_state["current_tilt"] = max(hw_state["TILT_MIN_LIMIT"], min(hw_state["TILT_MAX_LIMIT"], hw_state["current_tilt"]))
    hw_state["smooth_pan"]   = hw_state["smooth_pan"]  * 0.8 + hw_state["current_pan"]  * 0.2
    hw_state["smooth_tilt"]  = hw_state["smooth_tilt"] * 0.8 + hw_state["current_tilt"] * 0.2

def get_finger_status(hand_lms):
    wrist = hand_lms.landmark[0]
    fingers = []
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        dt = math.sqrt((hand_lms.landmark[tip].x - wrist.x)**2 + (hand_lms.landmark[tip].y - wrist.y)**2)
        dp = math.sqrt((hand_lms.landmark[pip].x - wrist.x)**2 + (hand_lms.landmark[pip].y - wrist.y)**2)
        fingers.append(dt > dp)
    return fingers

def heavy_processing(img_bgr, level=2):
    h, w = img_bgr.shape[:2]
    if h > w:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        h, w = img_bgr.shape[:2]

    detected_stepper_cmd = "NONE"
    mp_input = cv2.resize(img_bgr, (160, 120))
    results  = hands.process(cv2.cvtColor(mp_input, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_info in enumerate(results.multi_handedness):
            if hand_info.classification[0].label == "Right":
                hand_lms = results.multi_hand_landmarks[idx]
                hx, hy   = hand_lms.landmark[8].x, hand_lms.landmark[8].y
                if hw_state["is_servo_active"]:
                    calculate_servo_angles(hx, hy)
                    servo_msg = f"P{hw_state['smooth_pan']:.1f}T{hw_state['smooth_tilt']:.1f}"
                    sock.sendto(servo_msg.encode(), (ESP32_IP, UDP_PORT))
                f_status = get_finger_status(hand_lms)
                if f_status == [False, False, False, False]:
                    detected_stepper_cmd = "STOP"
                elif f_status == [True, True, False, False]:
                    detected_stepper_cmd = "DOWN" if hand_lms.landmark[8].y < hand_lms.landmark[0].y else "UP"

    # 브라우저 송출용 원본/화질 최적화 반환
    out_w, out_h, quality, _ = STREAM_LEVELS[level]
    img_out = cv2.resize(img_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    _, buf  = cv2.imencode('.jpg', img_out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes(), detected_stepper_cmd, img_out

async def process_video_track(track: rtc.VideoTrack):
    global latest_raw_frame, prev_frame_time
    loop = asyncio.get_event_loop()
    video_stream = rtc.VideoStream(track)
    try: target_format = rtc.VideoFormatType.FORMAT_RGBA8888
    except AttributeError:
        try: target_format = rtc.VideoBufferType.RGBA
        except AttributeError: target_format = 3

    async def recv_loop():
        global latest_raw_frame, prev_frame_time
        async for event in video_stream:
            try:
                rgba_frame = event.frame.convert(target_format)
                latest_raw_frame = rgba_frame
                curr_time = time.time()
                if prev_frame_time > 0:
                    diff = curr_time - prev_frame_time
                    if diff > 0.01: state["current_fps"] = 1.0 / diff
                prev_frame_time = curr_time
            except Exception as e:
                latest_raw_frame = None
                state["latest_frame"] = None
                break

    async def process_loop():
        global latest_raw_frame
        skip_count = 0
        last_frame_time = time.time()
        while True:
            await asyncio.sleep(0.01)
            frame = latest_raw_frame
            if frame is None:
                if time.time() - last_frame_time > 3.0 and state["latest_frame"] is not None:
                    state["latest_frame"] = None
                    state["log"] = "모바일 연결 끊김"
                continue

            last_frame_time = time.time()
            latest_raw_frame = None
            skip_count += 1
            cur_fps   = state["current_fps"]
            skip_rate = 3 if cur_fps < 15 else 2
            if skip_count % skip_rate != 0: continue

            # 적응형 FPS
            fps = state["current_fps"]
            if fps > 0:
                history = state["fps_history"]
                history.append(fps)
                if len(history) > ADAPTIVE_WINDOW: history.pop(0)
                if len(history) >= 3:
                    avg_fps   = sum(history) / len(history)
                    cur_level = state["stream_level"]
                    if avg_fps < FPS_THRESHOLDS["down"] and cur_level > 0:
                        state["stream_level"] = cur_level - 1
                        history.clear()
                    elif avg_fps > FPS_THRESHOLDS["up"] and cur_level < max(STREAM_LEVELS):
                        state["stream_level"] = cur_level + 1
                        history.clear()

            try:
                img = np.frombuffer(frame.data, dtype=np.uint8).reshape((frame.height, frame.width, 4))
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

                frame_bytes, stepper_cmd, img_out_bgr = await loop.run_in_executor(None, heavy_processing, img_bgr, state["stream_level"])

                if stepper_cmd in ["STOP", "NONE"]:
                    if hw_state["current_stepper_state"] != "STOP":
                        sock.sendto(b'S', (ESP32_IP, UDP_PORT))
                        hw_state["current_stepper_state"] = "STOP"
                elif stepper_cmd in ["UP", "DOWN"]:
                    if stepper_cmd != hw_state["current_stepper_state"]:
                        sock.sendto(stepper_cmd[0].encode(), (ESP32_IP, UDP_PORT))
                        hw_state["current_stepper_state"] = stepper_cmd

                state["latest_frame"] = frame_bytes

                # 🚀 [PC 브라우저 렌더링용 WebRTC 트랙 퍼블리싱]
                rgba_out = cv2.cvtColor(img_out_bgr, cv2.COLOR_BGR2RGBA)
                out_frame = rtc.VideoFrame(
                    rgba_out.shape[1], rgba_out.shape[0], 
                    rtc.VideoBufferType.RGBA, rgba_out.tobytes()
                )
                out_video_source.capture_frame(out_frame)

                state["log"] = f"📡 WebRTC | FPS: {round(state['current_fps'], 1)}"
            except Exception as e:
                print(f"🚨 영상 처리 오류: {e}")

    await asyncio.gather(recv_loop(), process_loop())

async def call_runpod_inference(manual_img_path, camera_frame_bytes, step_desc=""):
    if not RUNPOD_INFERENCE_URL: return {"result": "WAIT", "reason": "카메라 연결 대기 중..."}
    try:
        rel_path        = manual_img_path.lstrip('/')
        full_manual_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(full_manual_path): return {"result": "ERROR", "reason": "이미지 없음"}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                RUNPOD_INFERENCE_URL,
                files={
                    "manual_image": ("manual.jpg", open(full_manual_path, "rb"), "image/jpeg"),
                    "camera_image": ("camera.jpg", io.BytesIO(camera_frame_bytes), "image/jpeg"),
                },
                data={"prompt": f"비교 분석 수행\n{step_desc}", "web_send_time": str(time.time())},
                timeout=15.0
            )
            if response.status_code == 200:
                pred = response.json().get("prediction", {"result": "UNKNOWN", "reason": "분석 오류"})
                reason = pred.get("reason", "")
                if reason:
                    for sep in ['. ', '.\n', '\n']:
                        if sep in reason:
                            first = reason.split(sep)[0].strip()
                            if len(first) > 5:
                                reason = first
                                break
                    if len(reason) > 60: reason = reason[:60] + "..."
                    pred["reason"] = reason
                return pred
            return {"result": "ERROR", "reason": f"서버 오류: {response.status_code}"}
    except Exception as e:
        return {"result": "ERROR", "reason": f"통신 장애: {str(e)}"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_session()
    if CLOUDFLARED_BIN and not os.getenv("MOBILE_URL"):
        def _run_tunnel():
            try:
                proc = subprocess.Popen([CLOUDFLARED_BIN, "tunnel", "--url", "http://localhost:8000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for line in proc.stderr:
                    line = line.strip()
                    match = _re.search(r'https://[a-z0-9-]+[.]trycloudflare[.]com', line)
                    if match:
                        os.environ["MOBILE_URL"] = match.group(0)
                        break
            except Exception: pass
        threading.Thread(target=_run_tunnel, daemon=True).start()
        for _ in range(20):
            await asyncio.sleep(0.5)
            if os.getenv("MOBILE_URL"): break
    livekit_task  = asyncio.create_task(run_livekit())
    yield
    livekit_task.cancel()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

async def run_livekit():
    room = rtc.Room()
    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(process_video_track(track))

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
        if not LIVEKIT_TOKEN: return
        await room.connect(LIVEKIT_URL, LIVEKIT_TOKEN)
        print("✅ [SYSTEM] LiveKit 서버 접속 성공")
        
        # 🚀 [PC 브라우저 렌더링용 트랙 LiveKit에 발행]
        await room.local_participant.publish_track(out_video_track)
    except Exception as e:
        print(f"🚨 [ERROR] LiveKit 접속 실패: {e}")

@app.post("/ping")
async def ping_check(body: dict):
    return {"pong": True, "client_time": body.get("client_time", 0)}

@app.get("/preview-steps")
async def preview_steps():
    async def event_stream():
        last_count = 0
        idle_ticks = 0
        for _ in range(60):
            if state["progress_step"] in ("render", "analyze"): break
            await asyncio.sleep(0.5)
        while True:
            await asyncio.sleep(0.4)
            if len(_preview_steps) > last_count:
                new_steps = _preview_steps[last_count:]
                last_count = len(_preview_steps)
                yield f"data: {json.dumps(new_steps, ensure_ascii=False)}\n\n"
                idle_ticks = 0
            else: idle_ticks += 1
            if state["progress_step"] == "done" and idle_ticks >= 5:
                yield f"data: __done__\n\n"
                break
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/reset-preview")
async def reset_preview():
    global _preview_steps, _preview_updated
    _preview_steps = []
    _preview_updated = False
    return {"status": "ok"}

@app.get("/config")
async def get_config():
    """프론트엔드에 LiveKit 접속 정보 전달 — PC용과 Mobile용 토큰 분리 전송"""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content={
            "livekit_url":  LIVEKIT_URL,
            "mobile_token": MOBILE_TOKEN or "",
            "pc_token":     PC_TOKEN or MOBILE_TOKEN or "", # ⚠️ 충돌 방지를 위해 .env에 반드시 분리할 것
            "mobile_url":   os.getenv("MOBILE_URL", ""), 
        },
        headers={"Cache-Control": "no-store"}
    )

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
{"steps":[{"step_number":int,"title":"STEP N","desc":"원문","box_2d":[int,int,int,int]}]}"""
        res = await client.post(GEMINI_URL, json={"contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}], "generationConfig": {"responseMimeType": "application/json"}}, timeout=60.0)
        if res.status_code != 200: return []
        raw_steps = json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"]).get("steps", [])
        steps = []
        with Image.open(page_img_path) as full_img:
            for s in raw_steps:
                box  = s.get("box_2d")
                desc = s.get("desc", "").strip()
                if not box or not desc: continue
                ymin, xmin, ymax, xmax = box
                l, t = max(0, int(xmin*orig_w/1000)), max(0, int(ymin*orig_h/1000))
                r, b = min(orig_w, int(xmax*orig_w/1000)), min(orig_h, int(ymax*orig_h/1000))
                if r <= l or b <= t: continue
                step_num = s.get("step_number", 0)
                c_path   = os.path.join(job_dir, f"step_p{page_num}_{step_num}.jpg")
                full_img.crop((l, t, r, b)).convert("RGB").save(c_path, "JPEG", quality=92)
                step_data = {"step": step_num, "title": f"STEP {step_num}", "desc": desc, "image_url": f"/outputs/{os.path.relpath(c_path, OUTPUT_DIR)}".replace("\\", "/")}
                steps.append(step_data)
                global _preview_steps, _preview_updated
                _preview_steps.append(step_data)
                _preview_updated = True
                existing_urls = {st["image_url"] for st in state["manual_steps"]}
                if step_data["image_url"] not in existing_urls:
                    state["manual_steps"] = state["manual_steps"] + [step_data]
                    state["file_info"]["steps"] = len(state["manual_steps"])
        return steps
    except: return []

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
{"steps":[{"step_number":int,"title":str,"desc":str,"box_2d":[int,int,int,int]}]}"""
        res = await client.post(GEMINI_URL, json={"contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}], "generationConfig": {"responseMimeType": "application/json"}}, timeout=60.0)
        if res.status_code != 200: return []
        raw_steps = json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"]).get("steps", [])
        job_dir   = os.path.dirname(img_path)
        steps     = []
        with Image.open(img_path) as full_img:
            for s in raw_steps:
                box  = s.get("box_2d")
                desc = s.get("desc", "").strip()
                if not box or not desc: continue
                ymin, xmin, ymax, xmax = box
                l, t = max(0, int(xmin*orig_w/1000)), max(0, int(ymin*orig_h/1000))
                r, b = min(orig_w, int(xmax*orig_w/1000)), min(orig_h, int(ymax*orig_h/1000))
                if r <= l or b <= t: continue
                step_num = s.get("step_number") or (base_idx + len(steps) + 1)
                c_path   = os.path.join(job_dir, f"img_step_{step_num}_{int(time.time()*1000)}.jpg")
                full_img.crop((l, t, r, b)).convert("RGB").save(c_path, "JPEG", quality=92)
                steps.append({"step": step_num, "title": s.get("title", f"STEP {step_num}"), "desc": desc, "image_url": f"/outputs/{os.path.relpath(c_path, OUTPUT_DIR)}".replace("\\", "/")})
        return steps
    except: return []

@app.post("/process-manual")
async def handle_manual(files: List[UploadFile] = File(...)):
    global _analysis_start_time, _preview_steps, _preview_updated
    start_time = time.time()
    _analysis_start_time = start_time
    _preview_steps = []
    _preview_updated = False
    state["file_info"] = {"name": "", "pages": 0, "steps": 0}
    state.update({"is_analyzed": False, "step_locked": False, "progress_step": "upload", "ai_result": "WAIT", "analysis_time": 0.0})
    job_dir   = os.path.join(OUTPUT_DIR, f"sess_{int(start_time)}")
    os.makedirs(job_dir, exist_ok=True)
    all_steps = []
    async with httpx.AsyncClient() as client:
        for f in files:
            file_path = os.path.join(job_dir, f.filename)
            with open(file_path, "wb") as b: b.write(await f.read())
            ext = f.filename.lower()
            state["file_info"]["name"] = f.filename
            state["uploaded_preview"] = "" 
            if ext.endswith(".pdf"):
                thumb_path = os.path.join(job_dir, "preview_thumb.jpg")
                result = subprocess.run(["pdftoppm", "-jpeg", "-r", "72", "-f", "1", "-l", "1", file_path, os.path.join(job_dir, "thumb")], capture_output=True)
                thumb_files = sorted(glob.glob(os.path.join(job_dir, "thumb-*.jpg")))
                if thumb_files: state["uploaded_preview"] = f"/outputs/{os.path.relpath(thumb_files[0], OUTPUT_DIR)}".replace("\\", "/")
            elif ext.endswith((".png", ".jpg", ".jpeg")):
                state["uploaded_preview"] = f"/outputs/{os.path.relpath(file_path, OUTPUT_DIR)}".replace("\\", "/")

            if ext.endswith(".pdf"):
                pages_dir = os.path.join(job_dir, "pages")
                os.makedirs(pages_dir, exist_ok=True)
                state["progress_step"] = "render"
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: subprocess.run(["pdftoppm", "-jpeg", "-r", "200", file_path, os.path.join(pages_dir, "page")], capture_output=True))
                page_imgs = sorted(glob.glob(os.path.join(pages_dir, "page-*.jpg")))
                state["progress_step"] = "analyze"
                results = await asyncio.gather(*[analyze_pdf_page(client, p, i+1, job_dir) for i, p in enumerate(page_imgs)])
                for r in results: all_steps.extend(r)
            elif ext.endswith((".png", ".jpg", ".jpeg")):
                all_steps.extend(await detect_and_crop_image(client, file_path, len(all_steps)))

    unique, seen = [], set()
    for s in all_steps:
        if s["image_url"] not in seen: unique.append(s); seen.add(s["image_url"])
    def step_num(x):
        v = x.get("step")
        return int(v) if v and str(v).isdigit() else 999
    unique.sort(key=step_num)
    filtered = []
    for i, s in enumerate(unique):
        is_boundary = i == 0 or i == len(unique) - 1
        t, d = s.get("title","").lower(), s.get("desc","").lower()
        if is_boundary and any(k in t or k in d for k in ["안내","steam","teaching","copyright"]):
            if step_num(s) in (0, 999): continue
        filtered.append(s)

    with open(os.path.join(job_dir, "instruction.json"), "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=4)
    state.update({"manual_steps": filtered, "is_analyzed": True, "analysis_time": round(time.time()-start_time, 2), "current_step_idx": 0, "progress_step": "done"})
    save_session()
    return {"status": "success", "steps": filtered}

@app.post("/set-step")
async def set_step(body: dict):
    total = len(state["manual_steps"])
    if total == 0: return {"status": "error", "message": "매뉴얼 없음"}
    idx    = max(0, min(int(body.get("idx", state["current_step_idx"])), total-1))
    locked = bool(body.get("locked", True))
    state["current_step_idx"] = idx
    state["step_locked"]      = locked
    save_session()
    return {"status": "ok", "current_step_idx": idx}

@app.get("/status")
async def get_status():
    return {
        "manual_steps":     state["manual_steps"],
        "is_analyzed":      state["is_analyzed"],
        "current_step_idx": state["current_step_idx"],
        "ai_response":      state["ai_response"],
        "ai_result":        state["ai_result"],
        "analysis_time":    state["analysis_time"],
        "progress_step":    state["progress_step"],
        "step_locked":      state["step_locked"],
        "current_fps":      state["current_fps"],
        "stream_level":     state["stream_level"],
        "last_rtt":         state["last_rtt"],
        "log":              state["log"],
        "has_frame":        state["latest_frame"] is not None,
        "file_info":        state["file_info"],
        "uploaded_preview": state["uploaded_preview"],
        "elapsed_time":     round(time.time() - _analysis_start_time, 1) if state["progress_step"] not in ("upload", "done") and _analysis_start_time > 0 else state["analysis_time"],
    }

@app.post("/reset")
async def reset_session():
    state.update({
        "manual_steps": [], "current_step_idx": 0,
        "ai_response": "대기 중...", "ai_result": "WAIT",
        "is_analyzed": False, "analysis_time": 0.0,
        "step_locked": False, "progress_step": "upload",
        "file_info": {"name": "", "pages": 0, "steps": 0},
        "uploaded_preview": "",
    })
    if os.path.exists(SESSION_FILE): os.remove(SESSION_FILE)
    return {"status": "ok"}

@app.post("/trigger-vlm")
async def trigger_vlm_analysis():
    if not state["is_analyzed"] or not state["manual_steps"]: return {"status": "error", "message": "매뉴얼 준비 안 됨"}
    if not state["latest_frame"]: return {"status": "error", "message": "카메라 프레임 없음"}
    idx          = state["current_step_idx"]
    current_step = state["manual_steps"][idx]
    t_start      = time.time()
    prediction   = await call_runpod_inference(current_step["image_url"], state["latest_frame"], current_step.get("desc", ""))
    duration     = round(time.time() - t_start, 2)
    if prediction and prediction.get("result") != "ERROR":
        result = prediction.get("result", "UNKNOWN")
        reason = prediction.get("reason", "분석 완료")
        state["ai_response"] = f"[{result}] {reason}"
        state["ai_result"]   = result
        if result == "PASS" and not state["step_locked"]:
            if idx + 1 < len(state["manual_steps"]):
                state["current_step_idx"] = idx + 1
                save_session()
                async def _reset_after_delay():
                    await asyncio.sleep(2.0)
                    if state["ai_result"] == "PASS":
                        state["ai_result"]   = "WAIT"
                        state["ai_response"] = "대기 중..."
                asyncio.create_task(_reset_after_delay())
        return {"status": "success", "prediction": prediction, "current_step": state["current_step_idx"], "duration": duration}
    else:
        err = prediction.get("reason", "응답 없음") if prediction else "응답 없음"
        return {"status": "error", "message": f"VLM 실패: {err}"}

@app.get("/")
@app.get("/mobile")
async def serve_ui():
    if os.path.exists("nolegercy.html"): return FileResponse("nolegercy.html")
    return {"error": "nolegercy.html not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)