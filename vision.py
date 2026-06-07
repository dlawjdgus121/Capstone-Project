import asyncio
import concurrent.futures
import math
import socket
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from livekit import rtc
from collections import deque
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
thread_local_hands = threading.local()


def get_hands():
    if not hasattr(thread_local_hands, "hands"):
        thread_local_hands.hands = mp_hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return thread_local_hands.hands

last_vlm_encode_time = 0.0
cached_vlm_bytes = None
VLM_ENCODE_INTERVAL = 3.0
LOG_FRAME_INTERVAL = 0.25
DROP_LOG_INTERVAL = 2.0
PROCESS_INTERVAL_SECONDS = 0.08
last_frame_log_time = 0.0
last_drop_log_time = 0.0
latest_raw_frame = None
latest_raw_frame_time = 0.0
prev_frame_time = 0.0
last_servo_calc_time = 0.0
filtered_hand_x = None
filtered_hand_y = None
servo_velocity_pan = 0.0
servo_velocity_tilt = 0.0
hand_target_samples = deque(maxlen=8)
servo_command_was_active = False
last_servo_command_msg = b"V0.000Y0.000"
last_servo_command_active_time = 0.0
last_servo_keepalive_time = 0.0
gesture_state = {
    "swipe_history": deque(maxlen=8),
    "last_swipe_time": 0.0,
    "last_toggle_time": 0.0,
    "last_gesture": "NONE",
    "notice_gesture": "NONE",
    "notice_gesture_until": 0.0,
    "hold_elapsed": 0.0,
    "hold_required": 1.5,
    "hold_progress": 0.0,

    # 두 손 제스처 유지 판정용
    "two_hand_candidate": "NONE",
    "two_hand_candidate_start": 0.0,
    "two_hand_hold_required": 1.5,
    "last_two_hand_hold_log": 0.0,

    # 검지 제스처 유지 판정용 추가
    "linear_candidate": "NONE",
    "linear_candidate_start": 0.0,
    "linear_hold_required": 1.5,
    "last_linear_hold_log": 0.0,
    "linear_armed": True,
    "stop_armed": True,
}


def set_notice_gesture(gesture, duration=1.2):
    gesture_state["notice_gesture"] = gesture
    gesture_state["notice_gesture_until"] = time.time() + duration


def get_display_gesture():
    if time.time() < gesture_state.get("notice_gesture_until", 0.0):
        return gesture_state.get("notice_gesture", "NONE")
    gesture_state["notice_gesture"] = "NONE"
    if gesture_state["last_gesture"] == "STOP":
        gesture_state["last_gesture"] = "TRACKING"
    return gesture_state["last_gesture"]


def set_gesture_hold_progress(elapsed, required):
    gesture_state["hold_elapsed"] = max(0.0, min(float(elapsed), float(required)))
    gesture_state["hold_required"] = float(required)
    gesture_state["hold_progress"] = (
        gesture_state["hold_elapsed"] / gesture_state["hold_required"]
        if gesture_state["hold_required"] > 0
        else 0.0
    )


def reset_gesture_hold_progress():
    gesture_state["hold_elapsed"] = 0.0
    gesture_state["hold_progress"] = 0.0


def is_gesture_holding_active():
    return (
        gesture_state["linear_candidate"] != "NONE"
        or gesture_state["two_hand_candidate"] != "NONE"
    )

last_servo_send_time = 0.0
TRACK_EDGE_ZONE_X = 0.10
TRACK_EDGE_ZONE_Y = 0.10
SERVO_SEND_INTERVAL = 0.02  # Keep commanding continuously while outside deadzone.
SERVO_KEEPALIVE_INTERVAL = 0.05
SERVO_KEEPALIVE_HOLD_SEC = 0.9

def reset_servo_tracking_state():
    global filtered_hand_x, filtered_hand_y
    global servo_velocity_pan, servo_velocity_tilt
    global last_servo_calc_time
    global servo_command_was_active
    global last_servo_command_msg, last_servo_command_active_time
    global last_servo_keepalive_time

    filtered_hand_x = None
    filtered_hand_y = None
    servo_velocity_pan = 0.0
    servo_velocity_tilt = 0.0
    last_servo_calc_time = 0.0
    servo_command_was_active = False
    last_servo_command_msg = b"V0.000Y0.000"
    last_servo_command_active_time = 0.0
    last_servo_keepalive_time = 0.0
    hand_target_samples.clear()


def keep_servo_command_alive():
    global last_servo_send_time, last_servo_keepalive_time

    now = time.time()

    if not servo_command_was_active:
        return

    if now - last_servo_command_active_time > SERVO_KEEPALIVE_HOLD_SEC:
        return

    if now - last_servo_keepalive_time < SERVO_KEEPALIVE_INTERVAL:
        return

    sock.sendto(last_servo_command_msg, (ESP32_IP, UDP_PORT))
    last_servo_keepalive_time = now
    last_servo_send_time = now


def get_stable_hand_target(hx, hy):
    hand_target_samples.append((hx, hy))

    if len(hand_target_samples) < 4:
        return None

    xs = sorted(sample[0] for sample in hand_target_samples)
    ys = sorted(sample[1] for sample in hand_target_samples)

    if len(xs) >= 5:
        xs = xs[1:-1]
        ys = ys[1:-1]

    return sum(xs) / len(xs), sum(ys) / len(ys)


def is_inside_tracking_deadzone(hx, hy):
    return (
        TRACK_EDGE_ZONE_X < hx < 1.0 - TRACK_EDGE_ZONE_X
        and TRACK_EDGE_ZONE_Y < hy < 1.0 - TRACK_EDGE_ZONE_Y
    )


def calculate_servo_velocity_command(hx, hy):
    global last_servo_calc_time
    global filtered_hand_x, filtered_hand_y
    global servo_velocity_pan, servo_velocity_tilt

    now = time.time()

    if last_servo_calc_time <= 0:
        dt = 1.0 / 50.0
    else:
        dt = now - last_servo_calc_time

    last_servo_calc_time = now
    dt = max(0.001, min(dt, 0.08))

    HAND_FILTER_TAU = 0.18
    COMMAND_FILTER_TAU = 0.32

    def exp_alpha(tau):
        return 1.0 - math.exp(-dt / max(tau, 0.001))

    if filtered_hand_x is None or filtered_hand_y is None:
        filtered_hand_x = hx
        filtered_hand_y = hy
    else:
        hand_alpha = exp_alpha(HAND_FILTER_TAU)
        filtered_hand_x += (hx - filtered_hand_x) * hand_alpha
        filtered_hand_y += (hy - filtered_hand_y) * hand_alpha

    def edge_strength(v, edge_zone):
        if v < edge_zone:
            raw = (edge_zone - v) / edge_zone
            sign = 1.0
        elif v > 1.0 - edge_zone:
            raw = (v - (1.0 - edge_zone)) / edge_zone
            sign = -1.0
        else:
            return 0.0

        raw = min(1.0, raw)
        eased = raw * raw * (3.0 - 2.0 * raw)
        return sign * eased

    def vertical_edge_strength(v, edge_zone):
        if v < edge_zone:
            raw = (edge_zone - v) / edge_zone
            sign = -1.0
        elif v > 1.0 - edge_zone:
            raw = (v - (1.0 - edge_zone)) / edge_zone
            sign = 1.0
        else:
            return 0.0

        raw = min(1.0, raw)
        eased = raw * raw * (3.0 - 2.0 * raw)
        return sign * eased

    desired_pan = edge_strength(filtered_hand_x, TRACK_EDGE_ZONE_X)
    desired_tilt = vertical_edge_strength(filtered_hand_y, TRACK_EDGE_ZONE_Y)

    command_alpha = exp_alpha(COMMAND_FILTER_TAU)
    servo_velocity_pan += (desired_pan - servo_velocity_pan) * command_alpha
    servo_velocity_tilt += (desired_tilt - servo_velocity_tilt) * command_alpha

    if abs(servo_velocity_pan) < 0.025:
        servo_velocity_pan = 0.0
    if abs(servo_velocity_tilt) < 0.025:
        servo_velocity_tilt = 0.0

    return servo_velocity_pan, servo_velocity_tilt


def calculate_servo_angles(hx, hy):
    global last_servo_calc_time
    global filtered_hand_x, filtered_hand_y
    global servo_velocity_pan, servo_velocity_tilt

    now = time.time()

    if last_servo_calc_time <= 0:
        dt = 1.0 / 50.0
    else:
        dt = now - last_servo_calc_time

    last_servo_calc_time = now
    dt = max(0.001, min(dt, 0.08))

    PAN_SPEED_DEG_PER_SEC = 58.0
    TILT_SPEED_DEG_PER_SEC = 34.0
    PAN_ACCEL_DEG_PER_SEC2 = 170.0
    TILT_ACCEL_DEG_PER_SEC2 = 115.0

    HAND_FILTER_TAU = 0.15
    TARGET_FILTER_TAU = 0.18
    VELOCITY_DECAY_TAU = 0.16

    def exp_alpha(tau):
        return 1.0 - math.exp(-dt / max(tau, 0.001))

    hand_alpha = exp_alpha(HAND_FILTER_TAU)

    if filtered_hand_x is None or filtered_hand_y is None:
        filtered_hand_x = hx
        filtered_hand_y = hy
    else:
        filtered_hand_x += (hx - filtered_hand_x) * hand_alpha
        filtered_hand_y += (hy - filtered_hand_y) * hand_alpha

    def center_deadzone_strength(v, deadzone_half_width):
        error = v - 0.5
        distance = abs(error)

        if distance <= deadzone_half_width:
            return 0.0

        usable_range = max(0.001, 0.5 - deadzone_half_width)
        raw = min(1.0, (distance - deadzone_half_width) / usable_range)
        eased = raw * raw * (3.0 - 2.0 * raw)
        return math.copysign(eased, error)

    def approach_velocity(current, target, max_delta):
        diff = target - current
        if diff > max_delta:
            return current + max_delta
        if diff < -max_delta:
            return current - max_delta
        return target

    sx = center_deadzone_strength(filtered_hand_x, TRACK_EDGE_ZONE_X)
    sy = center_deadzone_strength(filtered_hand_y, TRACK_EDGE_ZONE_Y)

    desired_pan_velocity = -sx * PAN_SPEED_DEG_PER_SEC
    desired_tilt_velocity = sy * TILT_SPEED_DEG_PER_SEC

    if sx == 0.0:
        desired_pan_velocity = 0.0
    if sy == 0.0:
        desired_tilt_velocity = 0.0

    servo_velocity_pan = approach_velocity(
        servo_velocity_pan,
        desired_pan_velocity,
        PAN_ACCEL_DEG_PER_SEC2 * dt
    )
    servo_velocity_tilt = approach_velocity(
        servo_velocity_tilt,
        desired_tilt_velocity,
        TILT_ACCEL_DEG_PER_SEC2 * dt
    )

    if desired_pan_velocity == 0.0:
        servo_velocity_pan *= 1.0 - exp_alpha(VELOCITY_DECAY_TAU)
    if desired_tilt_velocity == 0.0:
        servo_velocity_tilt *= 1.0 - exp_alpha(VELOCITY_DECAY_TAU)

    hw_state["current_pan"] += servo_velocity_pan * dt
    hw_state["current_tilt"] += servo_velocity_tilt * dt

    hw_state["current_pan"] = max(
        hw_state["PAN_MIN_LIMIT"],
        min(hw_state["PAN_MAX_LIMIT"], hw_state["current_pan"])
    )

    hw_state["current_tilt"] = max(
        hw_state["TILT_MIN_LIMIT"],
        min(hw_state["TILT_MAX_LIMIT"], hw_state["current_tilt"])
    )

    target_alpha = exp_alpha(TARGET_FILTER_TAU)
    hw_state["smooth_pan"] += (
        hw_state["current_pan"] - hw_state["smooth_pan"]
    ) * target_alpha
    hw_state["smooth_tilt"] += (
        hw_state["current_tilt"] - hw_state["smooth_tilt"]
    ) * target_alpha
def get_finger_status(hand_lms):
    wrist = hand_lms.landmark[0]
    fingers = []
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        dt = math.sqrt((hand_lms.landmark[tip].x - wrist.x) ** 2 + (hand_lms.landmark[tip].y - wrist.y) ** 2)
        dp = math.sqrt((hand_lms.landmark[pip].x - wrist.x) ** 2 + (hand_lms.landmark[pip].y - wrist.y) ** 2)
        fingers.append(dt > dp)
    return fingers

def is_fist(hand_lms):
    """
    네 손가락이 모두 접히면 주먹으로 판단.
    엄지는 제외하고 검지/중지/약지/새끼 기준.
    """
    f_status = get_finger_status(hand_lms)
    return f_status == [False, False, False, False]

def reset_linear_hold(rearm=True):
    gesture_state["linear_candidate"] = "NONE"
    gesture_state["linear_candidate_start"] = 0.0
    gesture_state["linear_armed"] = bool(rearm)
    if gesture_state["last_gesture"] in ("HOLDING_UP", "HOLDING_DOWN"):
        gesture_state["last_gesture"] = "TRACKING"
    reset_gesture_hold_progress()


def is_index_only(hand_lms):
    """
    검지만 펴진 상태인지 판단.
    get_finger_status() 반환 순서:
    [검지, 중지, 약지, 새끼]
    """
    f_status = get_finger_status(hand_lms)
    return f_status == [True, False, False, False]


def get_index_direction(hand_lms):
    """
    검지 방향 판단.
    이미지 좌표계에서는 y가 작을수록 위쪽.
    
    반환:
    - UP
    - DOWN
    - NONE
    """
    wrist = hand_lms.landmark[0]
    index_tip = hand_lms.landmark[8]

    dx = index_tip.x - wrist.x
    dy = index_tip.y - wrist.y

    # 검지가 충분히 위/아래로 향하지 않으면 무시
    MIN_VERTICAL = 0.08

    if abs(dy) < MIN_VERTICAL:
        return "NONE"

    # 수평으로 누운 손가락 오인식 방지
    if abs(dy) < abs(dx) * 0.8:
        return "NONE"

    if dy < 0:
        return "UP"
    else:
        return "DOWN"


def detect_index_hold_slide(hand_lms):
    """
    검지만 펴고 위/아래 방향을 1.5초 유지하면
    리니어 슬라이드 UP/DOWN 명령 반환.
    
    반환:
    - UP
    - DOWN
    - HOLDING_UP
    - HOLDING_DOWN
    - NONE
    """
    now = time.time()
    hold_required = gesture_state["linear_hold_required"]

    if not is_index_only(hand_lms):
        reset_linear_hold()
        return "NONE"

    direction = get_index_direction(hand_lms)

    if direction not in ("UP", "DOWN"):
        reset_linear_hold()
        return "NONE"

    if not gesture_state["linear_armed"]:
        return "NONE"

    candidate = direction
    holding_state = f"HOLDING_{direction}"

    # 새 후보 제스처 시작
    if gesture_state["linear_candidate"] != candidate:
        gesture_state["linear_candidate"] = candidate
        gesture_state["linear_candidate_start"] = now
        gesture_state["last_gesture"] = holding_state
        set_gesture_hold_progress(0.0, hold_required)
        return holding_state

    held_time = now - gesture_state["linear_candidate_start"]
    set_gesture_hold_progress(held_time, hold_required)

    # 유지 중 로그, 너무 많이 찍히지 않게 0.3초 간격
    if now - gesture_state["last_linear_hold_log"] > 0.3:
        gesture_state["last_linear_hold_log"] = now
        print(
            f"⏳ [LINEAR HOLD] 검지 {candidate} 유지 중 "
            f"{held_time:.2f}/{hold_required:.1f}s"
        )

    # 1.5초 이상 유지되면 실제 명령 적용
    if held_time >= hold_required:
        gesture_state["last_gesture"] = candidate
        reset_linear_hold(rearm=False)
        return candidate

    gesture_state["last_gesture"] = holding_state
    return holding_state
def is_open_palm(hand_lms):
    """
    네 손가락 중 3개 이상 펴져 있으면 손바닥으로 판단.
    """
    f_status = get_finger_status(hand_lms)
    return sum(f_status) >= 3


def get_hand_center(hand_lms):
    """
    손 전체 중심점.
    스와이프는 검지 끝보다 손 전체 중심으로 보는 게 안정적.
    """
    xs = [lm.x for lm in hand_lms.landmark]
    ys = [lm.y for lm in hand_lms.landmark]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def detect_palm_swipe(hand_lms):
    """
    손바닥을 펼친 상태에서 위/아래 스와이프 감지.
    반환값:
    - UP
    - DOWN
    - NONE
    """
    now = time.time()

    if not is_open_palm(hand_lms):
        gesture_state["swipe_history"].clear()
        return "NONE"

    cx, cy = get_hand_center(hand_lms)
    gesture_state["swipe_history"].append((now, cx, cy))

    if len(gesture_state["swipe_history"]) < 4:
        return "NONE"

    # 중복 인식 방지
    if now - gesture_state["last_swipe_time"] < 0.8:
        return "NONE"

    t0, x0, y0 = gesture_state["swipe_history"][0]
    t1, x1, y1 = gesture_state["swipe_history"][-1]

    dt = t1 - t0
    if dt <= 0:
        return "NONE"

    dx = x1 - x0
    dy = y1 - y0

    MIN_DISTANCE = 0.15
    MAX_DURATION = 0.70
    MIN_SPEED = 0.25

    if dt > MAX_DURATION:
        return "NONE"

    speed_y = abs(dy) / dt

    # 수직 움직임이 수평 움직임보다 클 때만 스와이프 인정
    if abs(dy) > abs(dx) and abs(dy) > MIN_DISTANCE and speed_y > MIN_SPEED:
        gesture_state["last_swipe_time"] = now
        gesture_state["swipe_history"].clear()

        # 이미지 좌표계: y가 작아지면 위로 이동
        if dy < 0:
            return "DOWN"
        else:
            return "UP"

    return "NONE"


def get_detected_hands(results):
    """
    MediaPipe 결과에서 손 목록을 만들고, 가까운 손 순서로 정렬.
    반환 형식:
    [
        {"lms": hand_lms, "label": "Right", "area": 0.04},
        ...
    ]
    """
    if not results.multi_hand_landmarks:
        return []

    detected = []

    for idx, hand_lms in enumerate(results.multi_hand_landmarks):
        label = None

        if results.multi_handedness and idx < len(results.multi_handedness):
            label = results.multi_handedness[idx].classification[0].label

        area = hand_area_score(hand_lms)

        detected.append({
            "lms": hand_lms,
            "label": label,
            "area": area,
        })

    detected.sort(key=lambda h: h["area"], reverse=True)
    return detected

def reset_two_hand_hold():
    gesture_state["two_hand_candidate"] = "NONE"
    gesture_state["two_hand_candidate_start"] = 0.0
    if gesture_state["last_gesture"] in ("HOLDING_TRACKING_OFF", "HOLDING_TRACKING_ON"):
        gesture_state["last_gesture"] = "TRACKING"
    reset_gesture_hold_progress()


def classify_two_hand_command(detected_hands):
    """
    두 손 제스처 명령 판단.
    - 두 손 주먹을 1.5초 유지: TRACKING_OFF
    - 두 손바닥을 1.5초 유지: TRACKING_ON

    반환값:
    - TRACKING_OFF
    - TRACKING_ON
    - HOLDING_TRACKING_OFF
    - HOLDING_TRACKING_ON
    - NONE
    """
    now = time.time()
    hold_required = gesture_state["two_hand_hold_required"]

    if len(detected_hands) < 2:
        reset_two_hand_hold()
        return "NONE"

    h1 = detected_hands[0]["lms"]
    h2 = detected_hands[1]["lms"]

    both_fist = is_fist(h1) and is_fist(h2)
    both_palm = is_open_palm(h1) and is_open_palm(h2)

    if both_fist:
        candidate = "TRACKING_OFF"
        holding_state = "HOLDING_TRACKING_OFF"
    elif both_palm:
        candidate = "TRACKING_ON"
        holding_state = "HOLDING_TRACKING_ON"
    else:
        reset_two_hand_hold()
        return "NONE"

    # 이미 해당 상태라면 재적용하지 않음
    if candidate == "TRACKING_OFF" and not hw_state["is_servo_active"]:
        reset_two_hand_hold()
        return "NONE"

    if candidate == "TRACKING_ON" and hw_state["is_servo_active"]:
        reset_two_hand_hold()
        return "NONE"

    # 새 후보 제스처 시작
    if gesture_state["two_hand_candidate"] != candidate:
        gesture_state["two_hand_candidate"] = candidate
        gesture_state["two_hand_candidate_start"] = now
        gesture_state["last_gesture"] = holding_state
        set_gesture_hold_progress(0.0, hold_required)
        return holding_state

    held_time = now - gesture_state["two_hand_candidate_start"]
    set_gesture_hold_progress(held_time, hold_required)

    if now - gesture_state.get("last_two_hand_hold_log", 0.0) > 0.3:
        gesture_state["last_two_hand_hold_log"] = now
        print(
            f"⏳ [GESTURE HOLD] {candidate} 유지 중 "
            f"{held_time:.2f}/{hold_required:.1f}s"
        )

    if held_time >= hold_required:
        gesture_state["last_toggle_time"] = now
        gesture_state["last_gesture"] = candidate
        gesture_state["swipe_history"].clear()
        reset_two_hand_hold()
        reset_linear_hold()
        return candidate

    gesture_state["last_gesture"] = holding_state
    return holding_state


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
    img_out = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()



def heavy_processing(img_bgr, level=2):
    global last_servo_send_time, last_vlm_encode_time, cached_vlm_bytes
    global filtered_hand_x, filtered_hand_y
    global servo_velocity_pan, servo_velocity_tilt
    global last_servo_calc_time
    global servo_command_was_active
    global last_servo_command_msg, last_servo_command_active_time
    global last_servo_keepalive_time

    h, w = img_bgr.shape[:2]

    if h > w:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        h, w = img_bgr.shape[:2]

    source_w, source_h = w, h
    detected_stepper_cmd = "NONE"

    mp_pre_start = time.perf_counter()
    mp_input = cv2.resize(
        img_bgr,
        (HAND_TRACK_WIDTH, HAND_TRACK_HEIGHT),
        interpolation=cv2.INTER_AREA
    )
    mp_rgb = cv2.cvtColor(mp_input, cv2.COLOR_BGR2RGB)
    heavy_mp_pre_ms = round((time.perf_counter() - mp_pre_start) * 1000.0, 2)

    mp_infer_start = time.perf_counter()
    results = get_hands().process(mp_rgb)
    heavy_mp_infer_ms = round((time.perf_counter() - mp_infer_start) * 1000.0, 2)

    detected_hands = get_detected_hands(results)

    gesture_start = time.perf_counter()
    # 1. 두 손 추적 ON/OFF 제스처 우선 처리
    two_hand_cmd = classify_two_hand_command(detected_hands)

    skip_single_hand_control = two_hand_cmd in (
        "HOLDING_TRACKING_OFF",
        "HOLDING_TRACKING_ON",
        "TRACKING_OFF",
        "TRACKING_ON",
    )

    if two_hand_cmd == "TRACKING_OFF":
        hw_state["is_servo_active"] = False
        detected_stepper_cmd = "STOP"
        reset_linear_hold()
        print("🛑 [GESTURE] 두 손 주먹 1.5초 유지 | 손 추적 중지 + 슬라이드 정지")

    elif two_hand_cmd == "TRACKING_ON":
        hw_state["is_servo_active"] = True
        reset_linear_hold()
        print("🖐️ [GESTURE] 두 손바닥 1.5초 유지 | 손 추적 재개")

    # 2. 한 손 제스처 및 팬/틸트 추적
    if detected_hands and not skip_single_hand_control:
        hand_lms = detected_hands[0]["lms"]

        hx = hand_lms.landmark[8].x
        hy = hand_lms.landmark[8].y

        linear_cmd = "NONE"

        if is_fist(hand_lms):
            detected_stepper_cmd = "STOP"
            reset_linear_hold()
            gesture_state["swipe_history"].clear()
            reset_gesture_hold_progress()
            if gesture_state["stop_armed"] and hw_state["current_stepper_state"] != "STOP":
                gesture_state["last_gesture"] = "STOP"
                set_notice_gesture("STOP", 1.0)
                gesture_state["stop_armed"] = False
            else:
                gesture_state["last_gesture"] = "TRACKING"
            print("✊ [GESTURE] 주먹 | 슬라이드 정지")

        else:
            gesture_state["stop_armed"] = True
            linear_cmd = detect_index_hold_slide(hand_lms)

            if linear_cmd == "UP":
                detected_stepper_cmd = "UP"
                print("⬆️ [GESTURE] 검지 위 1.5초 유지 | 리니어 슬라이드 상승")

            elif linear_cmd == "DOWN":
                detected_stepper_cmd = "DOWN"
                print("⬇️ [GESTURE] 검지 아래 1.5초 유지 | 리니어 슬라이드 하강")

            elif linear_cmd == "NONE":
                if gesture_state["last_gesture"] not in ("UP", "DOWN"):
                    gesture_state["last_gesture"] = "TRACKING"

        # 팬/틸트 손 추적은 리니어 슬라이드 제스처와 별개로 처리
        if hw_state["is_servo_active"]:
            now = time.time()
            stable_target = get_stable_hand_target(hx, hy)

            if stable_target is not None:
                inside_deadzone = is_inside_tracking_deadzone(*stable_target)

                if inside_deadzone:
                    servo_velocity_pan = 0.0
                    servo_velocity_tilt = 0.0
                    last_servo_calc_time = 0.0

                    if servo_command_was_active:
                        sock.sendto(b"V0.000Y0.000", (ESP32_IP, UDP_PORT))
                        last_servo_send_time = now
                        last_servo_keepalive_time = now
                        last_servo_command_msg = b"V0.000Y0.000"
                        last_servo_command_active_time = 0.0
                        servo_command_was_active = False

                elif now - last_servo_send_time >= SERVO_SEND_INTERVAL:
                    pan_cmd, tilt_cmd = calculate_servo_velocity_command(
                        *stable_target
                    )
                    servo_msg = f"V{pan_cmd:.3f}Y{tilt_cmd:.3f}"

                    sock.sendto(servo_msg.encode(), (ESP32_IP, UDP_PORT))
                    last_servo_send_time = now
                    last_servo_keepalive_time = now
                    last_servo_command_msg = servo_msg.encode()
                    last_servo_command_active_time = now
                    servo_command_was_active = True

    else:
        reset_linear_hold()
        reset_servo_tracking_state()

    # PC 화면 렌더링은 WebRTC가 담당하므로 서버 preview JPEG는 낮게 유지
    stream_level_for_server = 0
    out_w, out_h, quality, _ = STREAM_LEVELS[stream_level_for_server]

    stream_encode_start = time.perf_counter()
    stream_bytes = encode_jpeg(img_bgr, out_w, out_h, quality)
    heavy_stream_jpeg_ms = round((time.perf_counter() - stream_encode_start) * 1000.0, 2)

    # VLM용 JPEG는 매 프레임 만들지 않고 일정 주기로만 갱신
    now = time.time()

    if cached_vlm_bytes is None or now - last_vlm_encode_time >= VLM_ENCODE_INTERVAL:
        vlm_encode_start = time.perf_counter()
        cached_vlm_bytes = encode_jpeg(
            img_bgr,
            VLM_FRAME_WIDTH,
            VLM_FRAME_HEIGHT,
            VLM_JPEG_QUALITY
        )
        last_vlm_encode_time = now
        heavy_vlm_jpeg_ms = round((time.perf_counter() - vlm_encode_start) * 1000.0, 2)
    else:
        heavy_vlm_jpeg_ms = 0.0

    vlm_bytes = cached_vlm_bytes

    heavy_gesture_ms = round((time.perf_counter() - gesture_start) * 1000.0, 2)
    heavy_total_ms = round((time.perf_counter() - mp_pre_start) * 1000.0, 2)

    frame_meta = {
        "recv_frame_size": frame_size_label(source_w, source_h),
        "stream_frame_size": frame_size_label(out_w, out_h),
        "vlm_frame_size": frame_size_label(VLM_FRAME_WIDTH, VLM_FRAME_HEIGHT),
        "hand_frame_size": frame_size_label(HAND_TRACK_WIDTH, HAND_TRACK_HEIGHT),
        "stream_jpeg_quality": quality,
        "vlm_jpeg_quality": VLM_JPEG_QUALITY,
        "gesture": get_display_gesture(),
        "gesture_holding_active": is_gesture_holding_active(),
        "gesture_hold_elapsed": round(gesture_state["hold_elapsed"], 2),
        "gesture_hold_required": round(gesture_state["hold_required"], 2),
        "gesture_hold_progress": round(gesture_state["hold_progress"], 3),
        "tracking_active": hw_state["is_servo_active"],
        "stepper_state": hw_state["current_stepper_state"],
        "heavy_mp_pre_ms": heavy_mp_pre_ms,
        "heavy_mp_infer_ms": heavy_mp_infer_ms,
        "heavy_gesture_ms": heavy_gesture_ms,
        "heavy_stream_jpeg_ms": heavy_stream_jpeg_ms,
        "heavy_vlm_jpeg_ms": heavy_vlm_jpeg_ms,
        "heavy_total_ms": heavy_total_ms,
    }

    return stream_bytes, vlm_bytes, detected_stepper_cmd, frame_meta


async def process_video_track(track: rtc.VideoTrack):
    global prev_frame_time
    loop = asyncio.get_event_loop()
    heavy_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    latest_frame = None
    latest_frame_arrival = 0.0
    current_heavy_future = None
    dropped_frames = 0

    video_stream = rtc.VideoStream(track)
    try:
        target_format = rtc.VideoFormatType.FORMAT_RGBA8888
    except AttributeError:
        try:
            target_format = rtc.VideoBufferType.RGBA
        except AttributeError:
            target_format = 3

    def update_state(frame_bytes, vlm_frame_bytes, stepper_cmd, frame_meta, arrival_ms, convert_ms):
        nonlocal current_heavy_future
        try:
            if stepper_cmd == "STOP":
                if hw_state["current_stepper_state"] != "STOP":
                    sock.sendto(b"S", (ESP32_IP, UDP_PORT))
                    hw_state["current_stepper_state"] = "STOP"
                    print("🛑 [HW] STEPPER STOP")
            elif stepper_cmd in ["UP", "DOWN"]:
                actual_stepper_cmd = "DOWN" if stepper_cmd == "UP" else "UP"
                if actual_stepper_cmd != hw_state["current_stepper_state"]:
                    udp_cmd = actual_stepper_cmd[0]
                    sock.sendto(udp_cmd.encode(), (ESP32_IP, UDP_PORT))
                    hw_state["current_stepper_state"] = actual_stepper_cmd
                    print(
                        f"🧭 [HW] STEPPER gesture={stepper_cmd} "
                        f"actual={actual_stepper_cmd}"
                    )

            first_frame = state["latest_frame"] is None
            state_update_start = time.perf_counter()
            state["latest_frame"] = frame_bytes
            state["latest_vlm_frame"] = vlm_frame_bytes
            state["recv_frame_size"] = frame_meta["recv_frame_size"]
            state["stream_frame_size"] = frame_meta["stream_frame_size"]
            state["vlm_frame_size"] = frame_meta["vlm_frame_size"]
            state["hand_frame_size"] = frame_meta["hand_frame_size"]
            state["stream_camera_kb"] = round(len(frame_bytes) / 1024.0, 2)
            state["vlm_camera_kb"] = round(len(vlm_frame_bytes) / 1024.0, 2)
            state["gesture"] = frame_meta.get("gesture", "NONE")
            state["gesture_holding_active"] = frame_meta.get("gesture_holding_active", False)
            state["gesture_hold_elapsed"] = frame_meta.get("gesture_hold_elapsed", 0.0)
            state["gesture_hold_required"] = frame_meta.get("gesture_hold_required", 1.5)
            state["gesture_hold_progress"] = frame_meta.get("gesture_hold_progress", 0.0)
            state["tracking_active"] = frame_meta.get("tracking_active", True)
            state["stepper_state"] = frame_meta.get("stepper_state", "STOP")
            state["log"] = (
                f"📡 WebRTC | FPS: {round(state['current_fps'], 1)} | "
                f"recv {frame_meta['recv_frame_size']} | "
                f"stream {frame_meta['stream_frame_size']} | "
                f"VLM {frame_meta['vlm_frame_size']} | "
                f"gesture {frame_meta.get('gesture', 'NONE')} | "
                f"tracking {'ON' if frame_meta.get('tracking_active') else 'OFF'}"
            )
            state_update_ms = round((time.perf_counter() - state_update_start) * 1000.0, 2)
            total_ms = round(arrival_ms + convert_ms + frame_meta.get('heavy_total_ms', 0.0) + state_update_ms, 2)
            heavy_detail = (
                f"mp={frame_meta['heavy_mp_pre_ms']:.0f}+{frame_meta['heavy_mp_infer_ms']:.0f} "
                f"gest={frame_meta['heavy_gesture_ms']:.0f} "
                f"jpeg={frame_meta['heavy_stream_jpeg_ms']:.0f} "
                f"vlm={frame_meta['heavy_vlm_jpeg_ms']:.0f} "
                f"total={frame_meta['heavy_total_ms']:.0f}"
            )
            print(
                f"🛰️ [FRAME] arrival={arrival_ms:.0f}ms convert={convert_ms:.0f}ms "
                f"heavy={frame_meta['heavy_total_ms']:.0f}ms({heavy_detail}) "
                f"update={state_update_ms:.0f}ms total={total_ms:.0f}ms "
                f"fps={state['current_fps']:.1f} recv={state['recv_frame_size']} "
                f"stream={state['stream_frame_size']} vlm={state['vlm_frame_size']}",
                flush=True,
            )
            if first_frame:
                print(
                    f"✅ [CAM CONNECTED] recv={state['recv_frame_size']} "
                    f"stream={state['stream_frame_size']} vlm={state['vlm_frame_size']} "
                    f"stream_kb={state['stream_camera_kb']} vlm_kb={state['vlm_camera_kb']} "
                    f"fps={state['current_fps']:.1f}",
                    flush=True,
                )
        finally:
            current_heavy_future = None

    def heavy_done(fut):
        nonlocal current_heavy_future
        try:
            frame_bytes, vlm_frame_bytes, stepper_cmd, frame_meta = fut.result()
            meta = getattr(fut, 'meta', {})
            loop.call_soon_threadsafe(
                update_state,
                frame_bytes,
                vlm_frame_bytes,
                stepper_cmd,
                frame_meta,
                meta.get('arrival_ms', 0.0),
                meta.get('convert_ms', 0.0),
            )
        except Exception as e:
            print(f"🚨 heavy task error: {e}")
        finally:
            current_heavy_future = None

    async def recv_loop():
        nonlocal latest_frame, latest_frame_arrival
        global prev_frame_time
        async for event in video_stream:
            try:
                frame_recv_start = time.perf_counter()
                rgba_frame = event.frame.convert(target_format)
                latest_frame = rgba_frame
                latest_frame_arrival = frame_recv_start

                curr_time = frame_recv_start
                if prev_frame_time > 0:
                    diff = curr_time - prev_frame_time
                    if diff > 0.01:
                        state["current_fps"] = 1.0 / diff
                prev_frame_time = curr_time
            except Exception as e:
                print(f"🚨 수신 오류: {e}")
                latest_frame = None
                latest_frame_arrival = 0.0
                clear_frame_state()
                state["log"] = "모바일 연결 끊김"
                break

    async def process_loop():
        nonlocal latest_frame, latest_frame_arrival, current_heavy_future, dropped_frames
        global last_frame_log_time, last_drop_log_time
        last_frame_time = time.perf_counter()

        while True:
            await asyncio.sleep(PROCESS_INTERVAL_SECONDS)

            if latest_frame is None:
                keep_servo_command_alive()
                if time.perf_counter() - last_frame_time > 3.0 and state["latest_frame"] is not None:
                    clear_frame_state()
                    state["log"] = "모바일 연결 끊김"
                    print("⚠️ [STREAM] 프레임 타임아웃 — 연결 끊김으로 판단")
                continue

            if current_heavy_future is not None and not current_heavy_future.done():
                dropped_frames += 1
                if time.perf_counter() - last_drop_log_time >= DROP_LOG_INTERVAL:
                    print(f"⚠️ [DROP] heavy worker busy, dropped {dropped_frames} frames", flush=True)
                    last_drop_log_time = time.perf_counter()
                    dropped_frames = 0
                continue

            frame = latest_frame
            frame_arrival_time = latest_frame_arrival
            latest_frame = None
            latest_frame_arrival = 0.0
            last_frame_time = time.perf_counter()

            process_start = time.perf_counter()
            arrival_ms = round((process_start - frame_arrival_time) * 1000.0, 2) if frame_arrival_time > 0 else 0.0

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
                convert_start = time.perf_counter()
                img = np.frombuffer(frame.data, dtype=np.uint8).reshape((frame.height, frame.width, 4))
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                convert_ms = round((time.perf_counter() - convert_start) * 1000.0, 2)

                current_heavy_future = heavy_executor.submit(heavy_processing, img_bgr, state["stream_level"])
                current_heavy_future.meta = {
                    "arrival_ms": arrival_ms,
                    "convert_ms": convert_ms,
                }
                current_heavy_future.add_done_callback(heavy_done)
            except Exception as e:
                print(f"🚨 영상 처리 오류: {e}")

    await asyncio.gather(recv_loop(), process_loop())
