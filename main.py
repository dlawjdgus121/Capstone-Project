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

# --- [PDF 조각 분석: 한국어 강제 출력 및 정밀 필터링] ---
async def analyze_with_context(client, img_path, page_context_text):
    """이미지 조각을 분석하여 한국어로 번역된 조립 단계를 추출합니다."""
    try:
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            img_rgb.thumbnail((1200, 1200))
            buf = io.BytesIO()
            img_rgb.save(buf, format="JPEG", quality=90)
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # 🌟 한국어 출력을 강제하고 영어를 번역하도록 지시하는 프롬프트
        prompt = f"""
        당신은 조립 매뉴얼 정밀 디지털화 및 번역 전문가입니다. 
        매뉴얼 내용이 영어로 되어 있더라도, 결과는 반드시 '자연스러운 한국어'로 작성하세요.

        [판단 지침]
        1. **언어 규칙 (필독)**: 
           - 'title'과 'desc'는 반드시 한국어로 작성하세요. 
           - 영문 매뉴얼인 경우, 기술적인 용어를 고려하여 한국 사용자가 이해하기 쉽게 번역하세요.
        2. **is_step 판별**:
           - 구체적인 조립 동작, 부품 확인, 준비물 단계만 'is_step': true로 하세요.
           - 브랜드 로고(Teaching STEAM), 섹션 제목만 있는 조각, 주의사항 없는 일반 안내문은 'is_step': false로 하세요.
        3. **내용 추출**:
           - 이미지 내 텍스트와 아래 제공된 [전체 페이지 텍스트]를 대조하여 가장 정확한 설명을 생성하세요.

        [전체 페이지 텍스트 문맥]:
        {page_context_text}

        반드시 아래 JSON 형식으로 응답하세요:
        {{"step_number": int, "title": "한국어 제목", "desc": "한국어 상세 설명", "is_step": bool}}
        """
        
        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        res = await client.post(GEMINI_URL, json=payload, timeout=30.0)
        if res.status_code == 200:
            text_res = res.json()['candidates'][0]['content']['parts'][0]['text']
            data = json.loads(text_res)
            if isinstance(data, list): data = data[0]
            
            if not data.get("is_step", False):
                return None
            
            title_val = data.get("title", "").lower()
            desc_val = data.get("desc", "").lower()
            
            # 오인식 키워드 차단
            invalid_keywords = ["안내", "공지", "steam", "teaching", "copyright", "정렬", "주의하여", "문의"]
            valid_action_keywords = ["나사", "결합", "연결", "끼웁니다", "조입니다", "부품", "step", "수량", "설치", "고정"]
            
            has_invalid = any(k in title_val or k in desc_val for k in invalid_keywords)
            has_valid = any(k in title_val or k in desc_val for k in valid_action_keywords)
            
            if has_invalid and not (has_valid and len(desc_val) > 25):
                return None

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

        prompt = """
        이미지에서 실제 조립 단계 영역만 탐지하세요. 
        결과(title, desc)는 반드시 한국어로 작성하고, 영문은 번역하세요.
        JSON: {"steps": [{"step_number": int, "box_2d": [ymin, xmin, ymax, xmax], "title": "한국어 제목", "desc": "한국어 설명", "is_step": bool}]}
        """
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
                    if not s.get("is_step", True): continue
                    
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
                
    unique_steps = []
    seen_urls = set()
    for s in all_steps:
        if s['image_url'] not in seen_urls:
            unique_steps.append(s)
            seen_urls.add(s['image_url'])
            
    def get_step_num(x):
        val = x.get('step')
        if val is None or not str(val).isdigit(): return 999
        return int(val)

    unique_steps.sort(key=get_step_num)
    
    final_filtered_steps = []
    for i, s in enumerate(unique_steps):
        is_boundary = (i == 0 or i == len(unique_steps) - 1)
        title = s.get("title", "").lower()
        desc = s.get("desc", "").lower()
        
        if is_boundary:
            if any(k in title or k in desc for k in ["안내", "steam", "teaching", "copyright"]):
                if get_step_num(s) == 999 or get_step_num(s) == 0:
                    continue
        
        final_filtered_steps.append(s)

    with open(os.path.join(job_dir, "instruction.json"), "w", encoding="utf-8") as f:
        json.dump(final_filtered_steps, f, ensure_ascii=False, indent=4)
    
    state.update({
        "manual_steps": final_filtered_steps, 
        "is_analyzed": True, 
        "analysis_time": round(time.time() - start_time, 2),
        "current_step_idx": 0
    })
    print(f"✅ [SUCCESS] 분석 완료. 모든 가이드가 한국어로 생성되었습니다.")
    return {"status": "success", "steps": final_filtered_steps}

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