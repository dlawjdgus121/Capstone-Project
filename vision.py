import asyncio
import math
import socket
import time

import cv2
import mediapipe as mp
import numpy as np
from livekit_client import rtc

from config import (
    ADAPTIVE_WINDOW,
    ESP32_IP,
    FPS_THRESHOLDS,
    HAND_TRACK_HEIGHT,
    HAND_TRACK_WIDTH,
    STREAM_LEVELS,
    UDP_PORT,
    VLM_FRAME_HEIGHT,
    VLM_FRAME_WIDTH,
    VLM_JPEG_QUALITY,
)
from state import clear_frame_state, hw_state, state

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

latest_raw_frame = None
prev_frame_time = 0.0


def calculate_servo_angles(hx, hy):
    dist_x = hx - 0.5
    dist_y = hy - 0.5

    deadzone = 0.10
    gain = 4.0
    smooth = 0.2

    move_x = 0.0
    move_y = 0.0

    if dist_x > deadzone:
        move_x = dist_x - deadzone
    elif dist_x < -deadzone:
        move_x = dist_x + deadzone

    if dist_y > deadzone:
        move_y = dist_y - deadzone
    elif dist_y < -deadzone:
        move_y = dist_y + deadzone

    hw_state["current_pan"] -= move_x * gain
    hw_state["current_tilt"] += move_y * gain

    hw_state["current_pan"] = max(
        hw_state["PAN_MIN_LIMIT"], min(hw_state["PAN_MAX_LIMIT"], hw_state["current_pan"])
    )
    hw_state["current_tilt"] = max(
        hw_state["TILT_MIN_LIMIT"], min(hw_state["TILT_MAX_LIMIT"], hw_state["current_tilt"])
    )

    hw_state["smooth_pan"] = hw_state["smooth_pan"] * (1.0 - smooth) + hw_state["current_pan"] * smooth
    hw_state["smooth_tilt"] = hw_state["smooth_tilt"] * (1.0 - smooth) + hw_state["current_tilt"] * smooth


def get_finger_status(hand_lms):
    wrist = hand_lms.landmark[0]
    fingers = []
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        dt = math.sqrt((hand_lms.landmark[tip].x - wrist.x) ** 2 + (hand_lms.landmark[tip].y - wrist.y) ** 2)
        dp = math.sqrt((hand_lms.landmark[pip].x - wrist.x) ** 2 + (hand_lms.landmark[pip].y - wrist.y) ** 2)
        fingers.append(dt > dp)
    return fingers


def hand_area_score(hand_lms):
    xs = [lm.x for lm in hand_lms.landmark]
    ys = [lm.y for lm in hand_lms.landmark]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def select_closest_hand(results, prefer_label=None):
    if not results.multi_hand_landmarks:
        return None, None

    candidates = []
    for idx, hand_lms in enumerate(results.multi_hand_landmarks):
        label = None
        if results.multi_handedness and idx < len(results.multi_handedness):
            label = results.multi_handedness[idx].classification[0].label
        if prefer_label is not None and label != prefer_label:
            continue
        candidates.append((hand_area_score(hand_lms), hand_lms, label))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def frame_size_label(width: int, height: int) -> str:
    return f"{int(width)}x{int(height)}"


def encode_jpeg(img_bgr, width: int, height: int, quality: int) -> bytes:
    img_out = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    ok, buf = cv2.imencode(".jpg", img_out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def heavy_processing(img_bgr, level=2):
    h, w = img_bgr.shape[:2]
    if h > w:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        h, w = img_bgr.shape[:2]

    source_w, source_h = w, h
    detected_stepper_cmd = "NONE"
    mp_input = cv2.resize(img_bgr, (HAND_TRACK_WIDTH, HAND_TRACK_HEIGHT), interpolation=cv2.INTER_LINEAR)
    results = hands.process(cv2.cvtColor(mp_input, cv2.COLOR_BGR2RGB))

    hand_lms = None
    if results.multi_hand_landmarks:
        hand_lms, _ = select_closest_hand(results, prefer_label="Right")

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
            detected_stepper_cmd = "DOWN" if hand_lms.landmark[8].y < hand_lms.landmark[0].y else "UP"

    out_w, out_h, quality, _ = STREAM_LEVELS[level]
    stream_bytes = encode_jpeg(img_bgr, out_w, out_h, quality)
    vlm_bytes = encode_jpeg(img_bgr, VLM_FRAME_WIDTH, VLM_FRAME_HEIGHT, VLM_JPEG_QUALITY)

    frame_meta = {
        "recv_frame_size": frame_size_label(source_w, source_h),
        "stream_frame_size": frame_size_label(out_w, out_h),
        "vlm_frame_size": frame_size_label(VLM_FRAME_WIDTH, VLM_FRAME_HEIGHT),
        "hand_frame_size": frame_size_label(HAND_TRACK_WIDTH, HAND_TRACK_HEIGHT),
        "stream_jpeg_quality": quality,
        "vlm_jpeg_quality": VLM_JPEG_QUALITY,
    }
    return stream_bytes, vlm_bytes, detected_stepper_cmd, frame_meta


async def process_video_track(track: rtc.VideoTrack):
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
                clear_frame_state()
                state["log"] = "모바일 연결 끊김"
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
                    clear_frame_state()
                    state["log"] = "모바일 연결 끊김"
                    print("⚠️ [STREAM] 프레임 타임아웃 — 연결 끊김으로 판단")
                continue

            last_frame_time = time.time()
            latest_raw_frame = None

            skip_count += 1
            cur_fps = state["current_fps"]
            skip_rate = 3 if cur_fps < 15 else 2
            if skip_count % skip_rate != 0:
                continue

            fps = state["current_fps"]
            if fps > 0:
                history = state["fps_history"]
                history.append(fps)
                if len(history) > ADAPTIVE_WINDOW:
                    history.pop(0)
                if len(history) >= 3:
                    avg_fps = sum(history) / len(history)
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
                img = np.frombuffer(frame.data, dtype=np.uint8).reshape((frame.height, frame.width, 4))
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

                frame_bytes, vlm_frame_bytes, stepper_cmd, frame_meta = await loop.run_in_executor(
                    None, heavy_processing, img_bgr, state["stream_level"]
                )

                if stepper_cmd in ["STOP", "NONE"]:
                    if hw_state["current_stepper_state"] != "STOP":
                        sock.sendto(b"S", (ESP32_IP, UDP_PORT))
                        hw_state["current_stepper_state"] = "STOP"
                elif stepper_cmd in ["UP", "DOWN"]:
                    if stepper_cmd != hw_state["current_stepper_state"]:
                        sock.sendto(stepper_cmd[0].encode(), (ESP32_IP, UDP_PORT))
                        hw_state["current_stepper_state"] = stepper_cmd

                state["latest_frame"] = frame_bytes
                state["latest_vlm_frame"] = vlm_frame_bytes
                state["recv_frame_size"] = frame_meta["recv_frame_size"]
                state["stream_frame_size"] = frame_meta["stream_frame_size"]
                state["vlm_frame_size"] = frame_meta["vlm_frame_size"]
                state["hand_frame_size"] = frame_meta["hand_frame_size"]
                state["stream_camera_kb"] = round(len(frame_bytes) / 1024.0, 2)
                state["vlm_camera_kb"] = round(len(vlm_frame_bytes) / 1024.0, 2)
                state["log"] = (
                    f"📡 WebRTC | FPS: {round(state['current_fps'], 1)} | "
                    f"recv {frame_meta['recv_frame_size']} | "
                    f"stream {frame_meta['stream_frame_size']} | "
                    f"VLM {frame_meta['vlm_frame_size']}"
                )
            except Exception as e:
                print(f"🚨 영상 처리 오류: {e}")

    await asyncio.gather(recv_loop(), process_loop())
