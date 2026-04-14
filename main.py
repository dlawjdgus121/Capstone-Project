import io
import os
import json
import time
import base64
import asyncio
import httpx
import uvicorn
from typing import List
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from dotenv import load_dotenv

# 오픈로더 라이브러리 임포트
try:
    import opendataloader_pdf
except ImportError:
    print("⚠️ 'pip install opendataloader-pdf'가 필요합니다.")

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 설정
API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

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
    "analysis_time": 0.0
}

# --- AI 도우미 1: 이미 자투리난 조각 분석 (PDF용) ---
async def analyze_fragment_metadata(client, img_path, base_idx):
    """
    OpenLoader가 이미 잘라놓은 이미지 조각을 분석하여 텍스트 정보를 입힙니다.
    """
    try:
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            img_rgb.thumbnail((800, 800))
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=80)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        prompt = """
        이미지 조각을 보고 조립 단계를 추출하세요.
        1. 'step_number': 이미지 내 숫자(1, 2, 3...) 추출. 없으면 0.
        2. 'title': 짧은 제목 (한국어).
        3. 'desc': 상세 조립 방법 (한국어).
        4. 'is_valid': 실제 조립 동작이면 true, 단순 로고나 무의미한 조각이면 false.

        응답 JSON:
        {"step_number": int, "title": "string", "desc": "string", "is_valid": bool}
        """
        
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
                    "image_url": f"/outputs/{os.path.relpath(img_path, OUTPUT_DIR)}"
                }
    except: pass
    return None

# --- AI 도우미 2: 통이미지 분석 후 직접 크롭 (이미지용) ---
async def detect_and_crop_image(client, img_path, base_idx):
    """
    이미지 한 장을 통째로 분석하여 좌표를 따고 자릅니다.
    """
    try:
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size
            img_rgb = img.convert("RGB")
            img_rgb.thumbnail((1024, 1024))
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        prompt = """
        이미지 내의 조립 단계들을 찾아 좌표[ymin, xmin, ymax, xmax] (0~1000 정규화)와 내용을 추출하세요.
        응답 JSON:
        {"steps": [{"step_number": int, "box_2d": [int, int, int, int], "title": "string", "desc": "string"}]}
        """

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
                        "image_url": f"/outputs/{os.path.relpath(c_path, OUTPUT_DIR)}"
                    })
            return steps
    except: pass
    return []

async def process_manual_task(files: List[UploadFile]):
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
            
            # 1. PDF인 경우: 오픈로더(OpenLoader) 사용
            if ext.endswith(".pdf"):
                print(f"📄 [PDF] OpenLoader 가동: {f.filename}")
                proc_dir = os.path.join(job_dir, "pdf_parts")
                os.makedirs(proc_dir, exist_ok=True)
                
                # 오픈로더를 통해 이미지 조각 추출
                opendataloader_pdf.convert(input_path=[file_path], output_dir=proc_dir, format="json")
                
                # 추출된 조각들 수집 및 분석
                fragments = []
                for root, _, fnames in os.walk(proc_dir):
                    for fn in fnames:
                        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                            fragments.append(os.path.join(root, fn))
                
                tasks = [analyze_fragment_metadata(client, fp, i) for i, fp in enumerate(fragments)]
                results = await asyncio.gather(*tasks)
                all_steps.extend([r for r in results if r])

            # 2. 이미지인 경우: AI 지능형 크롭 사용
            elif ext.endswith((".png", ".jpg", ".jpeg")):
                print(f"🖼️ [IMG] AI 지능형 크롭 가동: {f.filename}")
                img_steps = await detect_and_crop_image(client, file_path, len(all_steps))
                all_steps.extend(img_steps)

    # 전체 순서 정렬
    all_steps.sort(key=lambda x: x['step'])

    state["manual_steps"] = all_steps
    state["is_analyzed"] = True
    state["analysis_time"] = round(time.time() - start_time, 2)
    return all_steps

@app.post("/process-manual")
async def handle_manual(files: List[UploadFile] = File(...)):
    steps = await process_manual_task(files)
    return {"status": "success", "steps": steps}

@app.get("/status")
async def get_status():
    return state

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
async def serve(): return FileResponse("index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)