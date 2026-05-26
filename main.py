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
    """Find a cloudflared binary that matches the current OS."""
    import shutil

    if os.name == "nt":
        candidates = [
            os.path.join(BASE_DIR, "cloudflared.exe"),
            shutil.which("cloudflared.exe"),
            shutil.which("cloudflared"),
        ]
    else:
        candidates = [
            os.path.join(BASE_DIR, "cloudflared-linux-amd64"),
            os.path.join(BASE_DIR, "cloudflared"),
            "/usr/local/bin/cloudflared",
            shutil.which("cloudflared"),
        ]
    for c in candidates:
        if not c:
            continue
        if os.path.isfile(c):
            try:
                os.chmod(c, 0o755)
            except OSError:
                pass
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
    if not raw_runpod_url.startswith("http"):
        raw_runpod_url = f"http://{raw_runpod_url}"

    raw_runpod_base = raw_runpod_url.rstrip("/")

    if raw_runpod_base.endswith("/predict"):
        raw_runpod_base = raw_runpod_base[:-len("/predict")]
    if raw_runpod_base.endswith("/set_step"):
        raw_runpod_base = raw_runpod_base[:-len("/set_step")]
    if raw_runpod_base.endswith("/set-step"):
        raw_runpod_base = raw_runpod_base[:-len("/set-step")]

    RUNPOD_INFERENCE_BASE_URL = raw_runpod_base
    RUNPOD_SET_STEP_URL = f"{raw_runpod_base}/set_step"
    RUNPOD_PREDICT_URL = f"{raw_runpod_base}/predict"

    print(f"✅ RUNPOD BASE: {RUNPOD_INFERENCE_BASE_URL}")
    print(f"✅ RUNPOD SET_STEP: {RUNPOD_SET_STEP_URL}")
    print(f"✅ RUNPOD PREDICT: {RUNPOD_PREDICT_URL}")
else:
    RUNPOD_INFERENCE_BASE_URL = None
    RUNPOD_SET_STEP_URL = None
    RUNPOD_PREDICT_URL = None

last_vlm_step_key = None

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── ESP32 하드웨어 설정 ────────────────────────────────────────────────────
ESP32_IP = "192.168.137.227"
UDP_PORT = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

hw_state = {
    "PAN_MIN_LIMIT": 40.0,  "PAN_MAX_LIMIT": 140.0,
    "TILT_MIN_LIMIT": 50.0, "TILT_MAX_LIMIT": 150.0,
    "current_pan": 90.0,    "current_tilt": 90.0,
    "smooth_pan": 90.0,     "smooth_tilt": 90.0,
    "is_servo_active": True,
    "current_stepper_state": "STOP",
    "vlm_latency_ms": 0.0,
    "vlm_proxy_ms": 0.0,
    "vlm_total_s": 0.0,
}

# ── MediaPipe ──────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    max_num_hands=1, model_complexity=0,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# ── 적응형 스트리밍 레벨 정의 ────────────────────────────────────────────────
# level: (width, height, jpeg_quality, label)
STREAM_LEVELS = {
    0: (426,  240, 40, "저화질 240p"),   # 극저대역폭 — ~0.5Mbps
    1: (640,  360, 55, "중간 360p"),     # 저대역폭   — ~1.5Mbps
    2: (960,  540, 70, "고화질 540p"),   # 일반       — ~3Mbps
    3: (1280, 720, 85, "최고 720p"),     # 고대역폭   — ~6Mbps
}
FPS_THRESHOLDS = {
    # (하락 임계값, 상승 임계값) — N초 평균 기준
    "down": 15,   # 평균 FPS가 이 이하면 레벨 낮춤
    "up":   25,   # 평균 FPS가 이 이상이면 레벨 올림
}
ADAPTIVE_WINDOW = 5  # 최근 몇 개의 FPS 샘플로 판단할지

# ── 최신 프레임 버퍼 ────────────────────────────────────────────────────────
latest_raw_frame = None   # VideoFrame 객체

# ── 매뉴얼 파싱 프리뷰 — 추출된 STEP을 실시간으로 프론트에 전달
import asyncio as _asyncio
_preview_steps: list = []       # 추출된 STEP 누적 리스트
_preview_updated: bool = False  # 새 STEP 추가됐을 때 SSE 트리거용

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
    "file_info":      {"name": "", "pages": 0, "steps": 0},  # 업로드된 파일 정보
    "uploaded_preview": "",  # 업로드된 원본 파일 미리보기 URL
    "current_fps":    0.0,
    "last_rtt":       0.0,    # ms 단위 RTT
    "is_processing":  False,

    "vlm_latency_ms": 0.0,   # SGLang 실제 추론 시간
    "vlm_proxy_ms":   0.0,   # GPU proxy 전체 처리 시간
    "vlm_total_s":    0.0,   # Windows main.py 기준 전체 왕복 시간

    # 적응형 스트리밍 — FPS 기반 자동 품질 조절
    "stream_level":   2,      # 0=저화질, 1=중간, 2=고화질
    "fps_history":    [],     # 최근 FPS 기록 (평균 계산용)
    "log":            "모바일 연결 대기 중...",
}

SESSION_FILE   = os.path.join(OUTPUT_DIR, "last_session.json")
prev_frame_time = 0.0
frame_count     = 0

# LiveKit 설정
# LiveKit 설정
LIVEKIT_URL   = os.getenv("LIVEKIT_URL", "wss://capstoneproject-l2ih740k.livekit.cloud")
LIVEKIT_TOKEN = os.getenv("LIVEKIT_TOKEN")   # Python 서버용
MOBILE_TOKEN  = os.getenv("MOBILE_TOKEN")    # 모바일 브라우저용
PC_TOKEN      = os.getenv("PC_TOKEN")        # PC 브라우저용 추가
MOBILE_URL    = os.getenv("MOBILE_URL", "")

# ── 세션 저장/복원 ─────────────────────────────────────────────────────────
def save_session():
    if not state["is_analyzed"] or not state["manual_steps"]:
        return
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "current_step_idx": state["current_step_idx"],
                "manual_steps":     state["manual_steps"],
                "analysis_time":    state["analysis_time"],
                "step_locked":      state["step_locked"],
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 세션 저장 실패: {e}")

def load_session() -> bool:
    if not os.path.exists(SESSION_FILE):
        return False
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        steps = payload.get("manual_steps", [])
        if not steps:
            return False
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

# ── 하드웨어 보조 함수 ─────────────────────────────────────────────────────
def calculate_servo_angles(hx, hy):
    dist_x = hx - 0.5
    dist_y = hy - 0.5

    DEADZONE = 0.10
    GAIN = 4.0
    SMOOTH = 0.2

    move_x = 0.0
    move_y = 0.0

    # x축: 데드존을 벗어난 만큼만 보정
    if dist_x > DEADZONE:
        move_x = dist_x - DEADZONE
    elif dist_x < -DEADZONE:
        move_x = dist_x + DEADZONE

    # y축: 데드존을 벗어난 만큼만 보정
    if dist_y > DEADZONE:
        move_y = dist_y - DEADZONE
    elif dist_y < -DEADZONE:
        move_y = dist_y + DEADZONE

    # 데드존 밖으로 벗어난 양만큼만 서보 목표값 변경
    hw_state["current_pan"]  -= move_x * GAIN
    hw_state["current_tilt"] += move_y * GAIN

    hw_state["current_pan"] = max(
        hw_state["PAN_MIN_LIMIT"],
        min(hw_state["PAN_MAX_LIMIT"], hw_state["current_pan"])
    )

    hw_state["current_tilt"] = max(
        hw_state["TILT_MIN_LIMIT"],
        min(hw_state["TILT_MAX_LIMIT"], hw_state["current_tilt"])
    )

    hw_state["smooth_pan"] = (
        hw_state["smooth_pan"] * (1.0 - SMOOTH)
        + hw_state["current_pan"] * SMOOTH
    )

    hw_state["smooth_tilt"] = (
        hw_state["smooth_tilt"] * (1.0 - SMOOTH)
        + hw_state["current_tilt"] * SMOOTH
    )

def get_finger_status(hand_lms):
    wrist = hand_lms.landmark[0]
    fingers = []
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        dt = math.sqrt((hand_lms.landmark[tip].x - wrist.x)**2 + (hand_lms.landmark[tip].y - wrist.y)**2)
        dp = math.sqrt((hand_lms.landmark[pip].x - wrist.x)**2 + (hand_lms.landmark[pip].y - wrist.y)**2)
        fingers.append(dt > dp)
    return fingers

def hand_area_score(hand_lms):
    """
    손이 화면에서 차지하는 면적을 계산.
    값이 클수록 카메라에 더 가까운 손으로 판단.
    """
    xs = [lm.x for lm in hand_lms.landmark]
    ys = [lm.y for lm in hand_lms.landmark]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    return width * height


def select_closest_hand(results, prefer_label=None):
    """
    검출된 손 중 가장 가까운 손 선택.
    prefer_label='Right'를 주면 오른손 중 가장 가까운 손만 선택.
    prefer_label=None이면 모든 손 중 가장 가까운 손 선택.
    """
    if not results.multi_hand_landmarks:
        return None, None

    candidates = []

    for idx, hand_lms in enumerate(results.multi_hand_landmarks):
        label = None

        if results.multi_handedness and idx < len(results.multi_handedness):
            label = results.multi_handedness[idx].classification[0].label

        if prefer_label is not None and label != prefer_label:
            continue

        score = hand_area_score(hand_lms)
        candidates.append((score, hand_lms, label))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)

    closest_hand_lms = candidates[0][1]
    closest_label = candidates[0][2]

    return closest_hand_lms, closest_label

def heavy_processing(img_bgr, level=2):
    h, w = img_bgr.shape[:2]
    if h > w:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        h, w = img_bgr.shape[:2]

    detected_stepper_cmd = "NONE"
    mp_input = cv2.resize(img_bgr, (160, 120))  # 속도 최적화: 160x120으로 축소
    results  = hands.process(cv2.cvtColor(mp_input, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        hand_lms, hand_label = select_closest_hand(results, prefer_label="Right")

    if hand_lms is not None:
        hx, hy = hand_lms.landmark[8].x, hand_lms.landmark[8].y

        if hw_state["is_servo_active"]:
            calculate_servo_angles(hx, hy)
            servo_msg = f"P{hw_state['smooth_pan']:.1f}T{hw_state['smooth_tilt']:.1f}"
            sock.sendto(servo_msg.encode(), (ESP32_IP, UDP_PORT))

        f_status = get_finger_status(hand_lms)

        if f_status == [False, False, False, False]:
            detected_stepper_cmd = "STOP"

        elif f_status == [True, True, False, False]:
            detected_stepper_cmd = (
                "DOWN"
                if hand_lms.landmark[8].y < hand_lms.landmark[0].y
                else "UP"
            )
    # 적응형 스트리밍 — 현재 레벨에 맞는 해상도/품질로 인코딩
    out_w, out_h, quality, _ = STREAM_LEVELS[level]
    img_out = cv2.resize(img_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    _, buf  = cv2.imencode('.jpg', img_out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes(), detected_stepper_cmd

# ── LiveKit 비디오 트랙 수신 ─────────────────────────────────────────────
async def process_video_track(track: rtc.VideoTrack):
    """
    [최적화] 수신 루프와 처리 루프를 분리.
    - 수신 루프: 최신 프레임만 버퍼에 저장 (블로킹 없음)
    - 처리 루프: 별도 태스크에서 최신 프레임만 골라 처리
    """
    global latest_raw_frame, prev_frame_time
    loop = asyncio.get_event_loop()

    video_stream = rtc.VideoStream(track)
    try:
        target_format = rtc.VideoFormatType.FORMAT_RGBA8888
    except AttributeError:
        try:
            target_format = rtc.VideoBufferType.RGBA
        except AttributeError:
            target_format = 3

    async def recv_loop():
        """프레임 수신 전용 — 최신 프레임만 유지, 처리 대기 없음."""
        global latest_raw_frame, prev_frame_time
        async for event in video_stream:
            try:
                rgba_frame = event.frame.convert(target_format)
                latest_raw_frame = rgba_frame

                curr_time = time.time()
                if prev_frame_time > 0:
                    diff = curr_time - prev_frame_time
                    if diff > 0.01:
                        state["current_fps"] = 1.0 / diff
                prev_frame_time = curr_time
            except Exception as e:
                print(f"🚨 수신 오류: {e}")
                latest_raw_frame = None
                state["latest_frame"] = None
                state["log"] = "모바일 연결 끊김"
                break

    async def process_loop():
        """처리 전용 — 최신 프레임을 가져와 heavy_processing 실행."""
        global latest_raw_frame
        skip_count = 0
        last_frame_time = time.time()  # 마지막 프레임 수신 시각

        while True:
            await asyncio.sleep(0.01)

            frame = latest_raw_frame
            if frame is None:
                # 마지막 프레임으로부터 3초 이상 경과 → 연결 끊김으로 판단
                if time.time() - last_frame_time > 3.0 and state["latest_frame"] is not None:
                    state["latest_frame"] = None
                    state["log"] = "모바일 연결 끊김"
                    print("⚠️ [STREAM] 프레임 타임아웃 — 연결 끊김으로 판단")
                continue

            last_frame_time = time.time()  # 프레임 수신 시각 갱신

            latest_raw_frame = None  # 처리 시작 — 소비

            # FPS 기반 적응형 스킵 (FPS 낮으면 더 많이 스킵)
            skip_count += 1
            cur_fps   = state["current_fps"]
            skip_rate = 3 if cur_fps < 15 else 2  # 저FPS: 3프레임당 1, 고FPS: 2프레임당 1
            if skip_count % skip_rate != 0:
                continue

            # FPS 히스토리 → 적응형 레벨 조절
            fps = state["current_fps"]
            if fps > 0:
                history = state["fps_history"]
                history.append(fps)
                if len(history) > ADAPTIVE_WINDOW:
                    history.pop(0)
                if len(history) >= 3:
                    avg_fps   = sum(history) / len(history)
                    cur_level = state["stream_level"]
                    if avg_fps < FPS_THRESHOLDS["down"] and cur_level > 0:
                        state["stream_level"] = cur_level - 1
                        _, _, _, label = STREAM_LEVELS[state["stream_level"]]
                        print(f"📉 [ADAPTIVE] 평균 {avg_fps:.1f}fps → 레벨 낮춤: {label}")
                        history.clear()
                    elif avg_fps > FPS_THRESHOLDS["up"] and cur_level < max(STREAM_LEVELS):
                        state["stream_level"] = cur_level + 1
                        _, _, _, label = STREAM_LEVELS[state["stream_level"]]
                        print(f"📈 [ADAPTIVE] 평균 {avg_fps:.1f}fps → 레벨 올림: {label}")
                        history.clear()

            try:
                # LiveKit RGBA 프레임 → numpy BGR 변환
                img = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                    (frame.height, frame.width, 4)
                )
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

                frame_bytes, stepper_cmd = await loop.run_in_executor(
                    None, heavy_processing, img_bgr, state["stream_level"]
                )

                # 스테퍼 명령 전송
                if stepper_cmd in ["STOP", "NONE"]:
                    if hw_state["current_stepper_state"] != "STOP":
                        sock.sendto(b'S', (ESP32_IP, UDP_PORT))
                        hw_state["current_stepper_state"] = "STOP"
                elif stepper_cmd in ["UP", "DOWN"]:
                    if stepper_cmd != hw_state["current_stepper_state"]:
                        sock.sendto(stepper_cmd[0].encode(), (ESP32_IP, UDP_PORT))
                        hw_state["current_stepper_state"] = stepper_cmd

                state["latest_frame"] = frame_bytes
                state["log"] = f"📡 WebRTC | FPS: {round(state['current_fps'], 1)}"
            except Exception as e:
                print(f"🚨 영상 처리 오류: {e}")

    # 수신 + 처리 루프 동시 실행 (LiveKit)
    await asyncio.gather(recv_loop(), process_loop())

# ── RunPod 추론 ────────────────────────────────────────────────────────────
async def call_runpod_inference(manual_img_path, camera_frame_bytes, step_desc=""):
    """
    GPU proxy optimized mode:
    - step/manual image가 바뀌면 /set_step 1회 호출
    - 매 프레임 추론은 /predict에 camera_image만 전송
    """
    global last_vlm_step_key

    if not RUNPOD_INFERENCE_BASE_URL:
        return {"result": "WAIT", "reason": "카메라 연결 대기 중..."}

    try:
        rel_path = manual_img_path.lstrip("/")
        full_manual_path = os.path.join(BASE_DIR, rel_path)

        if not os.path.exists(full_manual_path):
            return {"result": "ERROR", "reason": "이미지 없음"}

        # step_id는 이미지 경로 + desc 기준으로 고정
        step_id_raw = f"{manual_img_path}|{step_desc}"
        step_id = str(abs(hash(step_id_raw)))

        prompt = f"""You are a lenient assembly manual inspector. Be generous with PASS judgments.

Current step instruction: {step_desc if step_desc else "No instruction — compare the two images visually."}

Judgment standard (IMPORTANT — be generous):
- PASS: The result roughly matches the instruction. Minor imperfections, slight misalignment, or partial completion are acceptable. If the main action is done, give PASS.
- FAIL: Only if the action is clearly NOT done at all, or the result is completely wrong.
- When in doubt, choose PASS.

Notes:
- Dotted lines = fold lines, arrows = movement direction.
- Do not penalize for camera angle, lighting, or small positional differences.

Response format:
- reason: 1 short sentence in Korean only. Max 30 characters. No English.
- result: PASS or FAIL only.

Example: "접힌 형태가 확인됩니다." → PASS / "접기가 전혀 되지 않았습니다." → FAIL
"""

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            # 1) step이 바뀐 경우에만 /set_step 호출
            if last_vlm_step_key != step_id:
                with open(full_manual_path, "rb") as f:
                    set_resp = await client.post(
                        RUNPOD_SET_STEP_URL,
                        files={
                            "manual_image": ("manual.jpg", f, "image/jpeg"),
                        },
                        data={
                            "step_id": step_id,
                            "prompt": prompt,
                            "warmup": "false",
                        },
                    )

                if set_resp.status_code != 200:
                    body_preview = set_resp.text[:300].replace("\n", " ")
                    print(f"🚨 [SET_STEP ERROR] {set_resp.status_code} | {body_preview}")
                    print(f"🚨 [SET_STEP URL] {RUNPOD_SET_STEP_URL}")
                    return {
                        "result": "ERROR",
                        "reason": f"스텝 등록 실패: {set_resp.status_code}",
                    }

                set_payload = set_resp.json()
                if set_payload.get("status") != "success":
                    return {
                        "result": "ERROR",
                        "reason": f"스텝 등록 오류: {set_payload.get('message', 'unknown')}",
                    }

                last_vlm_step_key = step_id

            # 2) 추론은 camera_image만 전송
            pred_resp = await client.post(
                RUNPOD_PREDICT_URL,
                files={
                    "camera_image": ("camera.jpg", io.BytesIO(camera_frame_bytes), "image/jpeg"),
                },
                data={
                    "step_id": step_id,
                    "web_send_time": str(time.time()),
                },
            )

            if pred_resp.status_code == 200:
                payload = pred_resp.json()

                pred = payload.get(
                    "prediction",
                    {"result": "UNKNOWN", "reason": "분석 오류"},
                )

                pred["_timing"] = payload.get("timing", {})

                reason = pred.get("reason", "")
                if reason:
                    for sep in [". ", ".\n", "\n"]:
                        if sep in reason:
                            first = reason.split(sep)[0].strip()
                            if len(first) > 5:
                                reason = first
                                break
                    if len(reason) > 60:
                        reason = reason[:60] + "..."
                    pred["reason"] = reason

                return pred

            body_preview = pred_resp.text[:300].replace("\n", " ")
            print(f"🚨 [PREDICT ERROR] {pred_resp.status_code} | {body_preview}")
            print(f"🚨 [PREDICT URL] {RUNPOD_PREDICT_URL}")
            return {
                "result": "ERROR",
                "reason": f"서버 오류: {pred_resp.status_code}",
            }
    except Exception as e:
        return {"result": "ERROR", "reason": f"통신 장애: {str(e)}"}

async def coaching_loop():
    while True:
        if state["is_analyzed"] and state["latest_frame"] and state["manual_steps"]:
            idx = state["current_step_idx"]
            if idx < len(state["manual_steps"]):
                current_step = state["manual_steps"][idx]
                prediction   = await call_runpod_inference(current_step["image_url"], state["latest_frame"], current_step.get("desc", ""))
                if prediction:
                    result = prediction.get("result", "UNKNOWN")
                    reason = prediction.get("reason", "분석 중...")
                    
                    timing = prediction.get("_timing", {})
                    state["vlm_latency_ms"] = float(timing.get("sglang_latency_ms", 0.0) or 0.0)
                    state["vlm_proxy_ms"]   = float(timing.get("e2e_proxy_ms", 0.0) or 0.0)

                    # 화면에 바로 보이게 ai_response에도 추가
                    lat_s = state["vlm_latency_ms"] / 1000.0
                    state["ai_response"] = f"[{result}] {reason} ({lat_s:.2f}s)"
                    state["ai_result"]   = result
                    if result == "PASS" and not state["step_locked"]:
                        if idx + 1 < len(state["manual_steps"]):
                            state["current_step_idx"] = idx + 1
                            save_session()
            await asyncio.sleep(3.0)
        else:
            # 카메라 미연결 시 ai_result 초기화
            if not state["latest_frame"] and state["ai_result"] not in ("WAIT",):
                state["ai_result"]   = "WAIT"
                state["ai_response"] = "모바일 카메라를 연결해 주세요."
            await asyncio.sleep(1.0)

# ── FastAPI 앱 ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_session()

    # Cloudflare 터널 자동 시작 — MOBILE_URL이 .env에 없을 때만
    if CLOUDFLARED_BIN and not os.getenv("MOBILE_URL"):
        def _run_tunnel():
            try:
                proc = subprocess.Popen(
                    [CLOUDFLARED_BIN, "tunnel", "--url", "http://localhost:8000"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                # cloudflared는 stderr로 로그 출력
                for line in proc.stderr:
                    line = line.strip()
                    if "trycloudflare" in line.lower() or "tunnel" in line.lower():
                        print(f"🔍 [CF 로그] {line}")
                    match = _re.search(r'https://[a-z0-9-]+[.]trycloudflare[.]com', line)
                    if match:
                        os.environ["MOBILE_URL"] = match.group(0)
                        print(f"✅ [Cloudflare] 터널 시작: {match.group(0)}")
                        break
            except Exception as e:
                print(f"⚠️ [Cloudflare] 터널 실패: {e}")
        threading.Thread(target=_run_tunnel, daemon=True).start()
        # URL이 잡힐 때까지 최대 10초 대기
        for _ in range(20):
            await asyncio.sleep(0.5)
            if os.getenv("MOBILE_URL"):
                break
        if not os.getenv("MOBILE_URL"):
            print("⚠️ [Cloudflare] URL 감지 실패 — QR은 현재 URL 기반으로 생성됩니다")
    elif not CLOUDFLARED_BIN:
        print("ℹ️ [Cloudflare] cloudflared 바이너리 없음 — 터널 미사용")

    # coaching_task = asyncio.create_task(coaching_loop())  # 자동 추론 비활성화 — VLM 버튼 수동 전용
    livekit_task  = asyncio.create_task(run_livekit())
    yield
    # coaching_task.cancel()
    livekit_task.cancel()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ── LiveKit WebRTC ──────────────────────────────────────────────────────────
async def run_livekit():
    room = rtc.Room()

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print("📹 LiveKit 비디오 트랙 수신 시작")
            asyncio.create_task(process_video_track(track))

    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        try:
            payload = json.loads(data.data.decode("utf-8"))
            if payload.get("type") == "ping":
                state["client_send_time"] = payload.get("send_time")
                resp = json.dumps({"type": "pong", "client_time": payload.get("send_time")})
                asyncio.create_task(
                    room.local_participant.publish_data(resp.encode("utf-8"))
                )
            elif payload.get("type") == "metrics":
                state["last_rtt"] = float(payload.get("rtt", 0.0))
        except:
            pass

    try:
        if not LIVEKIT_TOKEN:
            print("🚨 [ERROR] LIVEKIT_TOKEN이 .env 파일에 없습니다!")
            return
        await room.connect(LIVEKIT_URL, LIVEKIT_TOKEN)
        print("✅ [SYSTEM] LiveKit 서버 접속 성공")
    except Exception as e:
        print(f"🚨 [ERROR] LiveKit 접속 실패: {e}")

@app.post("/ping")
async def ping_check(body: dict):
    """클라이언트 RTT 측정용"""
    return {"pong": True, "client_time": body.get("client_time", 0)}

@app.get("/preview-steps")
async def preview_steps():
    """매뉴얼 파싱 중 추출된 STEP을 실시간으로 반환 (SSE)"""
    async def event_stream():
        last_count = 0
        idle_ticks = 0   # 분석 완료 후 대기 카운터

        # 분석 시작 대기 (최대 30초)
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

            # 분석 완료 + 2초 이상 새 STEP 없으면 종료
            if state["progress_step"] == "done" and idle_ticks >= 5:
                yield f"data: __done__\n\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/reset-preview")
async def reset_preview():
    global _preview_steps, _preview_updated
    _preview_steps = []
    _preview_updated = False
    return {"status": "ok"}

@app.get("/config")
async def get_config():
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content={
            "livekit_url":  LIVEKIT_URL,
            "mobile_token": MOBILE_TOKEN or "",
            "pc_token":     PC_TOKEN or "",
            "mobile_url":   os.getenv("MOBILE_URL", ""),
        },
        headers={"Cache-Control": "no-store"}
    )

# ── 매뉴얼 파싱 ───────────────────────────────────────────────────────────
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

        res = await client.post(GEMINI_URL, json={
            "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }, timeout=60.0)
        if res.status_code != 200:
            return []

        raw_steps = json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"]).get("steps", [])
        steps = []
        with Image.open(page_img_path) as full_img:
            for s in raw_steps:
                box  = s.get("box_2d")
                desc = s.get("desc", "").strip()
                if not box or not desc:
                    continue
                ymin, xmin, ymax, xmax = box
                l, t = max(0, int(xmin*orig_w/1000)), max(0, int(ymin*orig_h/1000))
                r, b = min(orig_w, int(xmax*orig_w/1000)), min(orig_h, int(ymax*orig_h/1000))
                if r <= l or b <= t:
                    continue
                step_num = s.get("step_number", 0)
                c_path   = os.path.join(job_dir, f"step_p{page_num}_{step_num}.jpg")
                full_img.crop((l, t, r, b)).convert("RGB").save(c_path, "JPEG", quality=92)
                step_data = {
                    "step": step_num, "title": f"STEP {step_num}",  # 항상 STEP N으로 고정
                    "desc": desc,
                    "image_url": f"/outputs/{os.path.relpath(c_path, OUTPUT_DIR)}".replace("\\", "/")
                }
                steps.append(step_data)
                # 폴링 프리뷰 — 추출 즉시 state에 추가 (중복 제거)
                global _preview_steps, _preview_updated
                _preview_steps.append(step_data)
                _preview_updated = True
                existing_urls = {s["image_url"] for s in state["manual_steps"]}
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

        res = await client.post(GEMINI_URL, json={
            "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }, timeout=60.0)
        if res.status_code != 200:
            return []

        raw_steps = json.loads(res.json()["candidates"][0]["content"]["parts"][0]["text"]).get("steps", [])
        job_dir   = os.path.dirname(img_path)
        steps     = []
        with Image.open(img_path) as full_img:
            orig_w, orig_h = full_img.size
            for s in raw_steps:
                box  = s.get("box_2d")
                desc = s.get("desc", "").strip()
                if not box or not desc:
                    continue
                ymin, xmin, ymax, xmax = box
                l, t = max(0, int(xmin*orig_w/1000)), max(0, int(ymin*orig_h/1000))
                r, b = min(orig_w, int(xmax*orig_w/1000)), min(orig_h, int(ymax*orig_h/1000))
                if r <= l or b <= t:
                    continue
                step_num = s.get("step_number") or (base_idx + len(steps) + 1)
                c_path   = os.path.join(job_dir, f"img_step_{step_num}_{int(time.time()*1000)}.jpg")
                full_img.crop((l, t, r, b)).convert("RGB").save(c_path, "JPEG", quality=92)
                steps.append({
                    "step": step_num, "title": s.get("title", f"STEP {step_num}"),
                    "desc": desc,
                    "image_url": f"/outputs/{os.path.relpath(c_path, OUTPUT_DIR)}".replace("\\", "/")
                })
        return steps
    except Exception as e:
        print(f"🚨 이미지 분석 에러: {e}")
    return []

# ── API 라우팅 ─────────────────────────────────────────────────────────────
@app.post("/process-manual")
async def handle_manual(files: List[UploadFile] = File(...)):
    global _analysis_start_time, _preview_steps, _preview_updated
    start_time = time.time()
    _analysis_start_time = start_time
    _preview_steps = []
    _preview_updated = False
    state["file_info"] = {"name": "", "pages": 0, "steps": 0}
    state.update({"is_analyzed": False, "step_locked": False,
                  "progress_step": "upload", "ai_result": "WAIT",
                  "analysis_time": 0.0})  # 시작 시 즉시 초기화
    job_dir   = os.path.join(OUTPUT_DIR, f"sess_{int(start_time)}")
    os.makedirs(job_dir, exist_ok=True)
    all_steps = []

    async with httpx.AsyncClient() as client:
        for f in files:
            file_path = os.path.join(job_dir, f.filename)
            with open(file_path, "wb") as b:
                b.write(await f.read())
            ext = f.filename.lower()
            state["file_info"]["name"] = f.filename
            state["uploaded_preview"] = ""  # 초기화

            # 업로드 즉시 미리보기 생성
            if ext.endswith(".pdf"):
                # PDF 첫 페이지를 썸네일로 변환
                thumb_path = os.path.join(job_dir, "preview_thumb.jpg")
                result = subprocess.run(
                    ["pdftoppm", "-jpeg", "-r", "72", "-f", "1", "-l", "1",
                     file_path, os.path.join(job_dir, "thumb")],
                    capture_output=True
                )
                thumb_files = sorted(glob.glob(os.path.join(job_dir, "thumb-*.jpg")))
                if thumb_files:
                    state["uploaded_preview"] = f"/outputs/{os.path.relpath(thumb_files[0], OUTPUT_DIR)}".replace("\\", "/")
            elif ext.endswith((".png", ".jpg", ".jpeg")):
                # 이미지는 그대로 사용
                state["uploaded_preview"] = f"/outputs/{os.path.relpath(file_path, OUTPUT_DIR)}".replace("\\", "/")

            if ext.endswith(".pdf"):
                pages_dir = os.path.join(job_dir, "pages")
                os.makedirs(pages_dir, exist_ok=True)
                state["progress_step"] = "render"
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: subprocess.run(
                    ["pdftoppm", "-jpeg", "-r", "200", file_path,
                     os.path.join(pages_dir, "page")], capture_output=True
                ))
                page_imgs = sorted(glob.glob(os.path.join(pages_dir, "page-*.jpg")))
                state["progress_step"] = "analyze"
                results = await asyncio.gather(*[
                    analyze_pdf_page(client, p, i+1, job_dir)
                    for i, p in enumerate(page_imgs)
                ])
                for r in results:
                    all_steps.extend(r)
            elif ext.endswith((".png", ".jpg", ".jpeg")):
                all_steps.extend(await detect_and_crop_image(client, file_path, len(all_steps)))

    unique, seen = [], set()
    for s in all_steps:
        if s["image_url"] not in seen:
            unique.append(s); seen.add(s["image_url"])

    def step_num(x):
        v = x.get("step")
        return int(v) if v and str(v).isdigit() else 999

    unique.sort(key=step_num)

    filtered = []
    for i, s in enumerate(unique):
        is_boundary = i == 0 or i == len(unique) - 1
        t, d = s.get("title","").lower(), s.get("desc","").lower()
        if is_boundary and any(k in t or k in d for k in ["안내","steam","teaching","copyright"]):
            if step_num(s) in (0, 999):
                continue
        filtered.append(s)

    with open(os.path.join(job_dir, "instruction.json"), "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=4)

    state.update({"manual_steps": filtered, "is_analyzed": True,
                  "analysis_time": round(time.time()-start_time, 2),
                  "current_step_idx": 0, "progress_step": "done"})
    save_session()
    print(f"✅ [SUCCESS] 분석 완료 — {len(filtered)}개 STEP, {state['analysis_time']}s")
    return {"status": "success", "steps": filtered}

@app.post("/set-step")
async def set_step(body: dict):
    total = len(state["manual_steps"])
    if total == 0:
        return {"status": "error", "message": "매뉴얼 없음"}
    idx    = max(0, min(int(body.get("idx", state["current_step_idx"])), total-1))
    locked = bool(body.get("locked", True))
    state["current_step_idx"] = idx
    state["step_locked"]      = locked
# ai_result는 즉시 초기화하지 않음 — TTS/UI가 결과를 보여준 후 자연스럽게 사라지도록
    state["ai_result"]        = "WAIT"
    state["ai_response"]      = "대기 중..."
    state["vlm_latency_ms"] = 0.0
    state["vlm_proxy_ms"] = 0.0
    state["vlm_total_s"] = 0.0    
    save_session()
    return {"status": "ok", "current_step_idx": idx}

_analysis_start_time: float = 0.0  # 분석 시작 시각 (전역)

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

        "vlm_latency_ms":   state["vlm_latency_ms"],
        "vlm_proxy_ms":     state["vlm_proxy_ms"],
        "vlm_total_s":      state["vlm_total_s"],

        "log":              state["log"],
        "has_frame":        state["latest_frame"] is not None,
        "file_info":        state["file_info"],
        "uploaded_preview": state["uploaded_preview"],
        "elapsed_time":     round(time.time() - _analysis_start_time, 1)
                            if state["progress_step"] not in ("upload", "done") and _analysis_start_time > 0
                            else state["analysis_time"],
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

        "manual_steps": [],
        "current_step_idx": 0,
        "ai_response": "대기 중.",
        "ai_result": "WAIT",
        "is_analyzed": False,
        "analysis_time": 0.0,
        "step_locked": False,
        "progress_step": "upload",

        # VLM timing reset
        "vlm_latency_ms": 0.0,
        "vlm_proxy_ms": 0.0,
        "vlm_total_s": 0.0,
    })
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
    return {"status": "ok"}

@app.post("/trigger-vlm")
async def trigger_vlm_analysis():
    try:
        if not state.get("is_analyzed") or not state.get("manual_steps"):
            return {"status": "error", "message": "매뉴얼 준비 안 됨"}

        if not state.get("latest_frame"):
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

        t_start = time.time()

        prediction = await call_runpod_inference(
            image_url,
            state["latest_frame"],
            desc,
        )

        duration = round(time.time() - t_start, 2)
        state["vlm_total_s"] = duration

        if prediction and prediction.get("result") != "ERROR":
            result = prediction.get("result", "UNKNOWN")
            reason = prediction.get("reason", "분석 완료")

            timing = prediction.get("_timing", {})
            state["vlm_latency_ms"] = float(timing.get("sglang_latency_ms", 0.0) or 0.0)
            state["vlm_proxy_ms"] = float(timing.get("e2e_proxy_ms", 0.0) or 0.0)

            lat_s = state["vlm_latency_ms"] / 1000.0
            state["ai_response"] = f"[{result}] {reason} ({lat_s:.2f}s)"
            state["ai_result"] = result

            print(f"⏱️ [VLM] {duration}s | {result}")

            if result == "PASS" and not state.get("step_locked"):
                if idx + 1 < len(state["manual_steps"]):
                    state["current_step_idx"] = idx + 1
                    save_session()

                    async def _reset_after_delay():
                        await asyncio.sleep(2.0)
                        if state["ai_result"] == "PASS":
                            state["ai_result"] = "WAIT"
                            state["ai_response"] = "대기 중..."

                    asyncio.create_task(_reset_after_delay())

            return {
                "status": "success",
                "prediction": prediction,
                "current_step": state["current_step_idx"],
                "duration": duration,
                "timing": {
                    "vlm_latency_ms": state.get("vlm_latency_ms", 0.0),
                    "vlm_proxy_ms": state.get("vlm_proxy_ms", 0.0),
                    "vlm_total_s": state.get("vlm_total_s", 0.0),
                },
            }

        err = prediction.get("reason", "응답 없음") if prediction else "응답 없음"
        print(f"🚨 [VLM ERROR] {err}")
        return {"status": "error", "message": f"VLM 실패: {err}"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"trigger-vlm 예외: {type(e).__name__}: {str(e)}",
        }

@app.get("/")
@app.get("/mobile")
async def serve_ui():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "index.html not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)