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
# 키를 쿼리스트링이 아닌 헤더로 보내 로그/프록시에 키가 찍히는 걸 방지
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
GEMINI_HEADERS = {"x-goog-api-key": API_KEY} if API_KEY else {}

INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")
RUNPOD_AUTH_HEADERS = {"X-API-Key": INFERENCE_API_KEY} if INFERENCE_API_KEY else {}

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

# 서버 SGLANG_TIMEOUT(기본 120s)보다 짧으면 서버가 아직 처리 중인데 클라이언트가 먼저
# 포기해서 다음 요청이 밀리는 문제가 생기므로 여유를 두고 맞춤
RUNPOD_HTTP_TIMEOUT = httpx.Timeout(130.0, connect=10.0)
# httpx 기본 keepalive_expiry는 5.0s로, 자동 추론 간격(5s)과 겹쳐 연결이 매번 만료됐다.
# 서버(uvicorn timeout_keep_alive)와 함께 올려야 효과가 있다 — 한쪽만 올리면 반대쪽이 끊는다.
RUNPOD_KEEPALIVE_EXPIRY = env_int("RUNPOD_KEEPALIVE_EXPIRY", 120)
RUNPOD_HTTP_LIMITS = httpx.Limits(
    max_keepalive_connections=4, max_connections=8, keepalive_expiry=RUNPOD_KEEPALIVE_EXPIRY
)

ESP32_IP = "192.168.137.194"
UDP_PORT = 12346
HAND_TRACK_WIDTH = env_int("HAND_TRACK_WIDTH", 320)
HAND_TRACK_HEIGHT = env_int("HAND_TRACK_HEIGHT", 180)
VLM_FRAME_WIDTH = env_int("VLM_FRAME_WIDTH", 960)
VLM_FRAME_HEIGHT = env_int("VLM_FRAME_HEIGHT", 540)
VLM_JPEG_QUALITY = max(1, min(95, env_int("VLM_JPEG_QUALITY", 70)))

STREAM_LEVELS = {
    0: (426, 240, 40, "저화질 240p"),
    1: (640, 360, 55, "중간 360p"),
    2: (960, 540, 70, "고화질 540p"),
    3: (1280, 720, 85, "최고 720p"),
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
