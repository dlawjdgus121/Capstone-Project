import os
import shutil

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def find_cloudflared():
    """Find a cloudflared binary that matches the current OS."""
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

    for candidate in candidates:
        if not candidate:
            continue
        if os.path.isfile(candidate):
            try:
                os.chmod(candidate, 0o755)
            except OSError:
                pass
            return candidate
    return None


CLOUDFLARED_BIN = find_cloudflared()

try:
    import opendataloader_pdf  # noqa: F401
except ImportError:
    print("⚠️ [DEBUG] 'pip install opendataloader-pdf' 라이브러리가 필요합니다.")

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

raw_runpod_url = os.getenv("RUNPOD_INFERENCE_URL", "").strip()
if raw_runpod_url:
    if not raw_runpod_url.startswith("http"):
        raw_runpod_url = f"http://{raw_runpod_url}"

    raw_runpod_base = raw_runpod_url.rstrip("/")
    for suffix in ("/predict", "/set_step", "/set-step"):
        if raw_runpod_base.endswith(suffix):
            raw_runpod_base = raw_runpod_base[: -len(suffix)]

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

RUNPOD_HTTP_TIMEOUT = httpx.Timeout(60.0, connect=10.0)

ESP32_IP = "192.168.137.227"
UDP_PORT = 12345

HAND_TRACK_WIDTH = env_int("HAND_TRACK_WIDTH", 96)
HAND_TRACK_HEIGHT = env_int("HAND_TRACK_HEIGHT", 96)
VLM_FRAME_WIDTH = env_int("VLM_FRAME_WIDTH", 640)
VLM_FRAME_HEIGHT = env_int("VLM_FRAME_HEIGHT", 360)
VLM_JPEG_QUALITY = max(1, min(95, env_int("VLM_JPEG_QUALITY", 40)))

STREAM_LEVELS = {
    0: (320, 180, 20, "초저화질 180p"),
    1: (426, 240, 30, "저화질 240p"),
    2: (640, 360, 45, "중간 360p"),
    3: (854, 480, 55, "고화질 480p"),
}
FPS_THRESHOLDS = {
    "down": 15,
    "up": 25,
}
ADAPTIVE_WINDOW = 5

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_TOKEN = os.getenv("LIVEKIT_TOKEN")
MOBILE_TOKEN = os.getenv("MOBILE_TOKEN")
PC_TOKEN = os.getenv("PC_TOKEN")
MOBILE_URL = os.getenv("MOBILE_URL", "")

PRELOAD_MANUAL_STEPS = os.getenv("PRELOAD_MANUAL_STEPS", "1") == "1"
PRELOAD_WARMUP = os.getenv("PRELOAD_WARMUP", "1") == "1"
