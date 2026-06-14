import json
import os
import time

from config import BASE_DIR, OUTPUT_DIR


def _current_session_dir() -> str | None:
    """현재 manual_steps의 image_url에서 세션 디렉터리(outputs/sess_X)를 유도."""
    for s in state.get("manual_steps", []):
        url = s.get("image_url", "")
        if "/outputs/" in url:
            seg = url.split("/outputs/", 1)[-1].split("/")[0]
            return os.path.join(OUTPUT_DIR, seg)
    return None

state = {
    "latest_frame": None,
    "latest_vlm_frame": None,
    "manual_steps": [],
    "current_step_idx": 0,
    "ai_response": "대기 중...",
    "ai_feedback": "",
    "ai_result": "WAIT",
    "pending_step_idx": None,
    "pass_hold_until": 0.0,
    "pass_transition_id": 0,
    "is_analyzed": False,
    "analysis_time": 0.0,
    "step_locked": False,
    "progress_step": "upload",
    "file_info": {"name": "", "pages": 0, "steps": 0},
    "uploaded_preview": "",
    "current_fps": 0.0,
    "last_rtt": 0.0,
    "is_processing": False,

    "vlm_latency_ms": 0.0,
    "vlm_inference_ms": 0.0,
    "vlm_proxy_ms": 0.0,
    "vlm_total_s": 0.0,
    "vlm_proxy_overhead_ms": 0.0,
    "vlm_predict_rtt_ms": 0.0,
    "vlm_transport_ms": 0.0,
    "vlm_image_send_ms": 0.0,
    "vlm_result_recv_ms": 0.0,
    "vlm_e2e_ms": 0.0,
    "vlm_set_step_ms": 0.0,
    "vlm_pre_predict_ms": 0.0,
    "vlm_http_client_get_ms": 0.0,
    "vlm_camera_kb": 0.0,
    "vlm_call_total_ms": 0.0,

    "recv_frame_size": "",
    "stream_frame_size": "",
    "vlm_frame_size": "",
    "hand_frame_size": "",
    "stream_camera_kb": 0.0,

    "stream_level": 2,
    "fps_history": [],
    "log": "모바일 연결 대기 중...",
    "auto_infer_enabled": False,
    "auto_infer_interval_s": 5.0,
    "camera_setup_active": False,
    "camera_setup_phase": "idle",
    "camera_setup_done": False,
    "camera_setup_message": "",
    "camera_setup_countdown": 0,
    "camera_setup_countdown_started_at": 0.0,
    "camera_setup_shoulder_y": None,
    "camera_setup_shoulder_line_y": 0.48,
}

hw_state = {
    "PAN_MIN_LIMIT": 40.0,
    "PAN_MAX_LIMIT": 140.0,
    "TILT_MIN_LIMIT": 50.0,
    "TILT_MAX_LIMIT": 140.0,
    "current_pan": 90.0,
    "current_tilt": 90.0,
    "smooth_pan": 90.0,
    "smooth_tilt": 90.0,
    "is_servo_active": True,
    "current_stepper_state": "STOP",
    "vlm_latency_ms": 0.0,
    "vlm_proxy_ms": 0.0,
    "vlm_total_s": 0.0,
}

SESSION_FILE = os.path.join(OUTPUT_DIR, "last_session.json")


def step_image_exists(step: dict) -> bool:
    if not isinstance(step, dict):
        return False
    image_url = step.get("image_url", "")
    if not image_url:
        return False
    image_path = os.path.join(BASE_DIR, image_url.lstrip("/"))
    return os.path.exists(image_path)


def clear_frame_state() -> None:
    state["latest_frame"] = None
    state["latest_vlm_frame"] = None
    state["recv_frame_size"] = ""
    state["stream_frame_size"] = ""
    state["vlm_frame_size"] = ""
    state["hand_frame_size"] = ""
    state["stream_camera_kb"] = 0.0
    state["vlm_camera_kb"] = 0.0


def clear_vlm_timing() -> None:
    state["vlm_latency_ms"] = 0.0
    state["vlm_inference_ms"] = 0.0
    state["vlm_proxy_ms"] = 0.0
    state["vlm_total_s"] = 0.0
    state["vlm_proxy_overhead_ms"] = 0.0
    state["vlm_predict_rtt_ms"] = 0.0
    state["vlm_transport_ms"] = 0.0
    state["vlm_image_send_ms"] = 0.0
    state["vlm_result_recv_ms"] = 0.0
    state["vlm_e2e_ms"] = 0.0
    state["vlm_set_step_ms"] = 0.0
    state["vlm_pre_predict_ms"] = 0.0
    state["vlm_http_client_get_ms"] = 0.0
    state["vlm_camera_kb"] = 0.0
    state["vlm_call_total_ms"] = 0.0


def save_session() -> None:
    if not state["is_analyzed"] or not state["manual_steps"]:
        return
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "current_step_idx": state["current_step_idx"],
                    "manual_steps": state["manual_steps"],
                    "analysis_time": state["analysis_time"],
                    "step_locked": state["step_locked"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        print(f"⚠️ 세션 저장 실패: {e}")

    # 세션별 메타(이어하기 목록·진행도용) — 각 세션 디렉터리에 session.json 저장
    sess = _current_session_dir()
    if sess:
        try:
            with open(os.path.join(sess, "session.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "name": state["file_info"].get("name", ""),
                        "current_step_idx": state["current_step_idx"],
                        "steps": len(state["manual_steps"]),
                        "analysis_time": state["analysis_time"],
                        "updated": time.time(),
                    },
                    f,
                    ensure_ascii=False,
                )
        except Exception:
            pass


def load_session() -> bool:
    if not os.path.exists(SESSION_FILE):
        return False
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        steps = payload.get("manual_steps", [])
        if not steps:
            return False
        missing_count = sum(1 for step in steps if not step_image_exists(step))
        if missing_count:
            print(f"⚠️ [SESSION] 이미지가 없는 이전 세션이라 복원 생략 — missing={missing_count}/{len(steps)}")
            return False
        idx = max(0, min(int(payload.get("current_step_idx", 0)), len(steps) - 1))
        state.update(
            {
                "manual_steps": steps,
                "current_step_idx": idx,
                "analysis_time": payload.get("analysis_time", 0),
                "step_locked": payload.get("step_locked", False),
                "camera_setup_done": False,
                "camera_setup_active": False,
                "camera_setup_phase": "idle",
                "pending_step_idx": None,
                "pass_hold_until": 0.0,
                "pass_transition_id": 0,
                "is_analyzed": True,
                "progress_step": "done",
            }
        )
        print(f"✅ [SESSION] 복원 완료 — STEP {idx + 1}/{len(steps)}")
        return True
    except Exception as e:
        print(f"⚠️ 세션 복원 실패: {e}")
        return False
