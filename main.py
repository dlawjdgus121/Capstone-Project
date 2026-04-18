import io
import os
import json
import time
import base64
import asyncio
import httpx
import uvicorn
from typing import List
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()

# 오픈로더 라이브러리 체크
try:
    import opendataloader_pdf
except ImportError:
    print("⚠️ 'pip install opendataloader-pdf'가 필요합니다.")

app = FastAPI()

# CORS 설정: 모든 접속 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- [환경 변수 및 URL 설정] ---
API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-09-2025")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

# 런포드 주소 처리 로직 (오류 방지)
raw_runpod_url = os.getenv("RUNPOD_INFERENCE_URL", "").strip()
if raw_runpod_url:
    # 프로토콜 누락 시 추가
    if not raw_runpod_url.startswith("http"):
        raw_runpod_url = f"http://{raw_runpod_url}"
    # 끝에 /predict가 없다면 추가
    if not raw_runpod_url.endswith("/predict"):
        raw_runpod_url = raw_runpod_url.rstrip("/") + "/predict"
    # 405 에러 방지를 위해 마지막 슬래시 강제 제거 (또는 서버 설정에 맞게 조정)
    RUNPOD_INFERENCE_URL = raw_runpod_url
else:
    RUNPOD_INFERENCE_URL = None

print(f"📡 [SYSTEM] 연결된 학습 모델 주소: {RUNPOD_INFERENCE_URL}")

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# --- [시스템 상태] ---
state = {
    "latest_frame": None,
    "manual_steps": [],
    "current_step_idx": 0,
    "ai_response": "대기 중...",
    "is_analyzed": False,
    "analysis_time": 0.0,
    "is_coaching_active": False
}

# --- [학습 모델 서버 통신 함수] ---
async def call_runpod_inference(manual_img_path, camera_frame_bytes):
    if not RUNPOD_INFERENCE_URL:
        return {"result": "ERROR", "reason": "런포드 주소가 설정되지 않았습니다."}

    try:
        # 매뉴얼 이미지 절대 경로 확보
        rel_path = manual_img_path.lstrip('/')
        full_manual_path = os.path.join(BASE_DIR, rel_path)
        
        if not os.path.exists(full_manual_path):
            return {"result": "ERROR", "reason": "매뉴얼 이미지 파일을 찾을 수 없습니다."}

        async with httpx.AsyncClient() as client:
            files = {
                "manual_image": ("manual.jpg", open(full_manual_path, "rb"), "image/jpeg"),
                "camera_image": ("camera.jpg", io.BytesIO(camera_frame_bytes), "image/jpeg")
            }
            # 학습 모델에 전송할 프롬프트
            data = {"prompt": "현재 조립 단계와 사용자의 화면을 비교해서 PASS 또는 FAIL을 판별해줘."}
            
            # POST 요청 전송
            response = await client.post(RUNPOD_INFERENCE_URL, files=files, data=data, timeout=15.0)
            
            if response.status_code == 200:
                res_data = response.json()
                return res_data.get("prediction", {"result": "UNKNOWN", "reason": "응답 파싱 오류"})
            else:
                return {"result": "ERROR", "reason": f"서버 응답 오류: {response.status_code}"}
    except Exception as e:
        return {"result": "ERROR", "reason": f"통신 장애: {str(e)}"}

# --- [실시간 코칭 워커 루프] ---
async def coaching_loop():
    print("🤖 [SYSTEM] 실시간 코칭 루프 가동")
    while True:
        if state["is_analyzed"] and state["latest_frame"] and len(state["manual_steps"]) > 0:
            idx = state["current_step_idx"]
            if idx < len(state["manual_steps"]):
                current_step = state["manual_steps"][idx]
                
                # 런포드 서버에 실시간 판별 요청
                prediction = await call_runpod_inference(current_step["image_url"], state["latest_frame"])
                
                res_status = prediction.get("result", "UNKNOWN")
                reason = prediction.get("reason", "분석 중...")

                if res_status == "PASS":
                    state["ai_response"] = f"✅ [정확합니다!] {reason}"
                elif res_status == "FAIL":
                    state["ai_response"] = f"❌ [확인이 필요해요] {reason}"
                else:
                    state["ai_response"] = reason

            await asyncio.sleep(3.0)  # 3초마다 체크
        else:
            await asyncio.sleep(1.0)

# --- [매뉴얼 분석 함수 (Gemini)] ---
async def analyze_fragment_metadata(client, img_path, base_idx):
    try:
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            img_rgb.thumbnail((800, 800))
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=80)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        prompt = """이미지 조각을 보고 조립 단계를 추출하세요. 응답 JSON: {"step_number": int, "title": "string", "desc": "string", "is_valid": bool}"""
        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        res = await client.post(GEMINI_URL, json=payload, timeout=30.0)
        if res.status_code == 200:
            data = json.loads(res.json()['candidates'][0]['content']['parts'][0]['text'])
            if data.get("is_valid"):
                return {
                    "step": data.get("step_number") or base_idx,
                    "title": data.get("title", "조립 단계"),
                    "desc": data.get("desc", ""),
                    "image_url": f"/outputs/{os.path.relpath(img_path, OUTPUT_DIR)}".replace("\\", "/")
                }
    except: pass
    return None

async def detect_and_crop_image(client, img_path, base_idx):
    try:
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size
            img_rgb = img.convert("RGB")
            img_rgb.thumbnail((1024, 1024))
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        prompt = """이미지 내의 조립 단계 좌표를 추출하세요. 응답 JSON: {"steps": [{"step_number": int, "box_2d": [int, int, int, int], "title": "string", "desc": "string"}]}"""
        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        res = await client.post(GEMINI_URL, json=payload, timeout=60.0)
        if res.status_code == 200:
            raw = json.loads(res.json()['candidates'][0]['content']['parts'][0]['text'])
            steps = []
            with Image.open(img_path) as full_img:
                for idx, s in enumerate(raw.get("steps", [])):
                    box = s.get("box_2d")
                    if not box: continue
                    left, top = box[1] * orig_w / 1000, box[0] * orig_h / 1000
                    right, bottom = box[3] * orig_w / 1000, box[2] * orig_h / 1000
                    crop = full_img.crop((left, top, right, bottom))
                    c_name = f"ai_crop_{int(time.time())}_{idx}.jpg"
                    c_path = os.path.join(os.path.dirname(img_path), c_name)
                    crop.convert("RGB").save(c_path, "JPEG", quality=90)
                    steps.append({
                        "step": s.get("step_number") or (base_idx + idx),
                        "title": s.get("title", "단계"),
                        "desc": s.get("desc", ""),
                        "image_url": f"/outputs/{os.path.relpath(c_path, OUTPUT_DIR)}".replace("\\", "/")
                    })
            return steps
    except: pass
    return []

# --- [API 엔드포인트] ---

@app.on_event("startup")
async def startup():
    asyncio.create_task(coaching_loop())

@app.post("/process-manual")
async def handle_manual(files: List[UploadFile] = File(...)):
    start_time = time.time()
    sess_id = f"sess_{int(start_time)}"
    job_dir = os.path.join(OUTPUT_DIR, sess_id)
    os.makedirs(job_dir, exist_ok=True)
    all_steps = []
    
    async with httpx.AsyncClient() as client:
        for f in files:
            file_path = os.path.join(job_dir, f.filename)
            content = await f.read()
            with open(file_path, "wb") as b: b.write(content)
            
            ext = f.filename.lower()
            if ext.endswith(".pdf"):
                proc_dir = os.path.join(job_dir, "pdf_parts")
                os.makedirs(proc_dir, exist_ok=True)
                opendataloader_pdf.convert(input_path=[file_path], output_dir=proc_dir, format="json")
                fragments = []
                for root, _, fnames in os.walk(proc_dir):
                    for fn in fnames:
                        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                            fragments.append(os.path.join(root, fn))
                tasks = [analyze_fragment_metadata(client, fp, i) for i, fp in enumerate(fragments)]
                results = await asyncio.gather(*tasks)
                all_steps.extend([r for r in results if r])
            elif ext.endswith((".png", ".jpg", ".jpeg")):
                img_steps = await detect_and_crop_image(client, file_path, len(all_steps))
                all_steps.extend(img_steps)
                
    all_steps.sort(key=lambda x: x['step'])
    state["manual_steps"] = all_steps
    state["is_analyzed"] = True
    state["analysis_time"] = round(time.time() - start_time, 2)
    return {"status": "success", "steps": all_steps}

@app.get("/status")
async def get_status():
    return {
        "ai_response": state["ai_response"],
        "manual_steps": state["manual_steps"],
        "current_step_idx": state["current_step_idx"],
        "is_analyzed": state["is_analyzed"],
        "analysis_time": state["analysis_time"],
        "has_frame": state["latest_frame"] is not None
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            state["latest_frame"] = await websocket.receive_bytes()
    except WebSocketDisconnect:
        state["latest_frame"] = None

@app.get("/stream")
async def stream():
    async def gen():
        while True:
            if state["latest_frame"]:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + state["latest_frame"] + b"\r\n")
            await asyncio.sleep(0.06)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def serve_pc():
    return FileResponse("index.html")

@app.get("/mobile")
async def serve_mobile():
    return FileResponse("index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)