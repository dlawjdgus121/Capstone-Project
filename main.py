import io
import os
import json
import time
import base64
import asyncio
import httpx
import uvicorn
import glob
import re
from typing import List
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()

try:
    import opendataloader_pdf
except ImportError:
    print("⚠️ [DEBUG] 'pip install opendataloader-pdf' 라이브러리가 필요합니다.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- [환경 변수 및 모델 URL 설정] ---
API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-09-2025")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

raw_runpod_url = os.getenv("RUNPOD_INFERENCE_URL", "").strip()
if raw_runpod_url:
    if not raw_runpod_url.startswith("http"): raw_runpod_url = f"http://{raw_runpod_url}"
    if not raw_runpod_url.endswith("/predict"): raw_runpod_url = raw_runpod_url.rstrip("/") + "/predict"
    RUNPOD_INFERENCE_URL = raw_runpod_url
else:
    RUNPOD_INFERENCE_URL = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

state = {
    "latest_frame": None,
    "manual_steps": [],
    "current_step_idx": 0,
    "ai_response": "대기 중...",
    "is_analyzed": False,
    "analysis_time": 0.0,
    "is_coaching_active": False
}

# --- [실시간 코칭 인퍼런스 서버 통신] ---
async def call_runpod_inference(manual_img_path, camera_frame_bytes):
    if not RUNPOD_INFERENCE_URL:
        return {"result": "WAIT", "reason": "카메라 연결 대기 중..."}
    try:
        rel_path = manual_img_path.lstrip('/')
        full_manual_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(full_manual_path): return {"result": "ERROR", "reason": "이미지 없음"}

        async with httpx.AsyncClient() as client:
            files = {
                "manual_image": ("manual.jpg", open(full_manual_path, "rb"), "image/jpeg"),
                "camera_image": ("camera.jpg", io.BytesIO(camera_frame_bytes), "image/jpeg")
            }
            data = {"prompt": "비교 분석 수행"}
            response = await client.post(RUNPOD_INFERENCE_URL, files=files, data=data, timeout=15.0)
            if response.status_code == 200:
                return response.json().get("prediction", {"result": "UNKNOWN", "reason": "분석 오류"})
            return {"result": "ERROR", "reason": f"서버 오류: {response.status_code}"}
    except Exception as e:
        return {"result": "ERROR", "reason": f"통신 장애: {str(e)}"}

async def coaching_loop():
    while True:
        if state["is_analyzed"] and state["latest_frame"] and state["manual_steps"]:
            idx = state["current_step_idx"]
            if idx < len(state["manual_steps"]):
                current_step = state["manual_steps"][idx]
                prediction = await call_runpod_inference(current_step["image_url"], state["latest_frame"])
                if prediction:
                    state["ai_response"] = f"[{prediction.get('result', 'UNKNOWN')}] {prediction.get('reason', '분석 중...')}"
            await asyncio.sleep(3.0)
        else:
            await asyncio.sleep(1.0)

# --- [PDF 조각 분석: OCR 데이터 강제 매핑 및 생성형 복원] ---
async def analyze_with_context(client, img_path, page_context_text):
    """이미지 자체 OCR과 텍스트 컨텍스트를 결합하여 완벽한 지시문을 추출합니다."""
    try:
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            img_rgb.thumbnail((1200, 1200)) # OCR 품질을 위해 해상도 상향
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=90)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # 🌟 지시문 누락 방지를 위한 초강력 프롬프트
        prompt = f"""
        당신은 복잡한 조립 매뉴얼을 완벽하게 디지털화하는 전문가입니다.
        입력된 [이미지 조각]과 [페이지 전체 텍스트 데이터]를 대조하여 가장 풍부한 지시문을 반환하세요.

        [페이지 전체 텍스트 데이터]:
        {page_context_text}

        [분석 지침]:
        1. **지시문(desc) 추출 규칙 (필독)**:
           - 제공된 [페이지 전체 텍스트 데이터]에서 이 단계와 관련된 한국어 설명을 찾아 '토씨 하나 틀리지 말고 그대로' 가져오세요.
           - 만약 텍스트 데이터에 설명이 부족하다면, **[이미지 조각]을 직접 정밀 OCR 분석**하여 이미지에 적힌 모든 한국어 문장을 빠짐없이 추출하세요.
           - "STEP 1" 같은 짧은 문구만 적는 것은 실패입니다. 이미지 내의 모든 지시문이나 텍스트 데이터의 상세 설명을 우선순위로 두세요.
        2. **텍스트가 없는 경우**: 이미지 내에 글자가 전혀 없다면, 시각적 동작(예: 화살표 방향, 부품 위치)을 해석하여 한국어로 친절한 지시문을 직접 작성하세요.
        3. **단계 번호**: 이미지 속의 번호를 추출하여 정수로 반환하세요.

        반드시 아래 JSON 형식으로만 응답하세요:
        {{"step_number": int, "title": "string", "desc": "string", "is_step": bool}}
        """
        
        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        res = await client.post(GEMINI_URL, json=payload, timeout=30.0)
        if res.status_code == 200:
            data = json.loads(res.json()['candidates'][0]['content']['parts'][0]['text'])
            if isinstance(data, list): data = data[0]
            
            # desc가 너무 짧으면 context의 전체 텍스트 중 일부를 백업으로 사용 시도
            if len(data.get("desc", "")) < 5 and page_context_text:
                print(f"⚠️ [WARN] {os.path.basename(img_path)}의 지시문이 너무 짧아 컨텍스트 보강을 시도합니다.")

            if data.get("is_step", True):
                return {
                    "step": data.get("step_number"),
                    "title": data.get("title", "조립 단계"),
                    "desc": data.get("desc", "이미지 내용을 확인해주세요."),
                    "image_url": f"/outputs/{os.path.relpath(img_path, OUTPUT_DIR)}".replace("\\", "/")
                }
    except Exception as e:
        print(f"🚨 분석 에러 ({os.path.basename(img_path)}): {e}")
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

        prompt = "이미지에서 조립 단계 영역을 탐지하고 상세 지시문을 추출해라. 글자가 없으면 동작을 해석해라. JSON: {\"steps\": [{\"step_number\": int, \"box_2d\": [ymin, xmin, ymax, xmax], \"title\": \"string\", \"desc\": \"string\"}]}"
        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        res = await client.post(GEMINI_URL, json=payload, timeout=60.0)
        if res.status_code == 200:
            raw_res = json.loads(res.json()['candidates'][0]['content']['parts'][0]['text'])
            raw_steps = raw_res.get("steps", [])
            
            steps = []
            with Image.open(img_path) as full_img:
                for idx, s in enumerate(raw_steps):
                    box = s.get("box_2d")
                    if not box: continue
                    left, top, right, bottom = box[1]*orig_w/1000, box[0]*orig_h/1000, box[3]*orig_w/1000, box[2]*orig_h/1000
                    crop = full_img.crop((left, top, right, bottom))
                    c_path = os.path.join(os.path.dirname(img_path), f"crop_{int(time.time())}_{idx}.jpg")
                    crop.convert("RGB").save(c_path, "JPEG", quality=90)
                    steps.append({
                        "step": s.get("step_number") or (base_idx + idx + 1),
                        "title": s.get("title", f"단계"),
                        "desc": s.get("desc", ""),
                        "image_url": f"/outputs/{os.path.relpath(c_path, OUTPUT_DIR)}".replace("\\", "/")
                    })
            return steps
    except: pass
    return []

# --- [FastAPI 라우팅] ---

@app.on_event("startup")
async def startup(): asyncio.create_task(coaching_loop())

@app.post("/process-manual")
async def handle_manual(files: List[UploadFile] = File(...)):
    start_time = time.time()
    state["is_analyzed"] = False
    sess_id = f"sess_{int(start_time)}"
    job_dir = os.path.join(OUTPUT_DIR, sess_id)
    os.makedirs(job_dir, exist_ok=True)
    all_steps = []
    
    async with httpx.AsyncClient() as client:
        for f in files:
            file_path = os.path.join(job_dir, f.filename)
            with open(file_path, "wb") as b: b.write(await f.read())
            
            ext = f.filename.lower()
            if ext.endswith(".pdf"):
                proc_dir = os.path.join(job_dir, "pdf_parts")
                os.makedirs(proc_dir, exist_ok=True)
                print(f"📄 [SYSTEM] PDF 변환 시작: {f.filename}")
                opendataloader_pdf.convert(input_path=[file_path], output_dir=proc_dir, format="json")
                
                # 🌟 [개선] OCR 텍스트 추출 정확도 향상
                page_texts = {} 
                img_to_page = {} 
                
                json_files = glob.glob(os.path.join(proc_dir, "**", "*.json"), recursive=True)
                for jf in json_files:
                    try:
                        with open(jf, 'r', encoding='utf-8') as j:
                            data = json.load(j)
                            items = data if isinstance(data, list) else [data]
                            for item in items:
                                p_num = item.get('page_num', 0)
                                # 텍스트 데이터 누락 방지: 가능한 모든 필드 수집
                                txt = item.get('text') or item.get('content') or item.get('ocr_text') or ""
                                page_texts[p_num] = page_texts.get(p_num, "") + "\n" + txt.strip()
                                
                                img_key = item.get('image_path') or item.get('filename') or item.get('image')
                                if img_key:
                                    img_to_page[os.path.basename(img_key)] = p_num
                    except Exception as e:
                        print(f"⚠️ [DEBUG] JSON 파싱 에러: {e}")

                fragments = glob.glob(os.path.join(proc_dir, "**", "*.png"), recursive=True) + \
                            glob.glob(os.path.join(proc_dir, "**", "*.jpg"), recursive=True)
                
                print(f"🔍 [SYSTEM] {len(fragments)}개의 이미지 조각 분석 시작...")
                
                tasks = []
                for fp in fragments:
                    fname = os.path.basename(fp)
                    p_num = img_to_page.get(fname, 0)
                    context = page_texts.get(p_num, "")
                    tasks.append(analyze_with_context(client, fp, context))
                
                results = await asyncio.gather(*tasks)
                for res in results:
                    if res: all_steps.append(res)
                
            elif ext.endswith((".png", ".jpg", ".jpeg")):
                img_steps = await detect_and_crop_image(client, file_path, len(all_steps))
                all_steps.extend(img_steps)
                
    # 중복 제거 및 정렬
    unique_steps = []
    seen_urls = set()
    for s in all_steps:
        if s['image_url'] not in seen_urls:
            unique_steps.append(s)
            seen_urls.add(s['image_url'])
            
    unique_steps.sort(key=lambda x: int(x.get('step') or 0) if str(x.get('step')).isdigit() else 999)
    
    with open(os.path.join(job_dir, "instruction.json"), "w", encoding="utf-8") as f:
        json.dump(unique_steps, f, ensure_ascii=False, indent=4)
    
    state.update({"manual_steps": unique_steps, "is_analyzed": True, "analysis_time": round(time.time() - start_time, 2)})
    print(f"✅ [SUCCESS] 분석 완료. {len(unique_steps)}개 단계 추출됨.")
    return {"status": "success", "steps": unique_steps}

@app.get("/status")
async def get_status():
    return {**state, "has_frame": state["latest_frame"] is not None}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True: state["latest_frame"] = await websocket.receive_bytes()
    except WebSocketDisconnect: state["latest_frame"] = None

@app.get("/stream")
async def stream():
    async def gen():
        while True:
            if state["latest_frame"]:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + state["latest_frame"] + b"\r\n")
            await asyncio.sleep(0.06)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
@app.get("/mobile")
async def serve_ui():
    if os.path.exists("index.html"): return FileResponse("index.html")
    return {"error": "index.html not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)