import io, os, re, json, time, base64, asyncio, tempfile, httpx, uvicorn
from typing import List
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# PDF 지원 체크
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# API 설정
const_apiKey = ""
MODEL_NAME = "gemini-3-flash-preview"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={const_apiKey}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# 전역 상태 관리
state = {
    "latest_frame": None,
    "manual_steps": [],
    "current_step_idx": 0,
    "ai_feedback": "시스템 준비 완료. 매뉴얼을 업로드하세요.",
    "is_analyzed": False,
    "is_coaching_active": False,
    "processing_tasks": 0,
    "analysis_time": 0.0
}

MAX_CONCURRENT_TASKS = 2

# --- Gemini API 호출 유틸리티 ---
async def call_gemini(prompt, pil_image=None):
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    if pil_image:
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        payload["contents"][0]["parts"].append({"inlineData": {"mimeType": "image/jpeg", "data": img_b64}})
    
    payload["generationConfig"] = {"responseMimeType": "application/json"}

    async with httpx.AsyncClient() as client:
        for i in range(3):
            try:
                res = await client.post(GEMINI_URL, json=payload, timeout=40.0)
                if res.status_code == 200:
                    return res.json()
            except: pass
            await asyncio.sleep(1)
    return None

# --- 병렬 코칭 루프 ---
async def run_parallel_analysis(frame_data, steps, current_idx):
    state["processing_tasks"] += 1
    try:
        current_step = steps[current_idx]
        prompt = f"""너는 조립 전문가야. 현재 단계: {current_step['title']} ({current_step['desc']}). 
        카메라를 보고 피드백을 한국어로 짧게 줘. 완료했다면 'is_completed'를 true로 해.
        JSON 응답: {{"feedback": "메시지", "is_completed": bool}}"""
        
        pil_frame = Image.open(io.BytesIO(frame_data))
        result = await call_gemini(prompt, pil_frame)
        if result:
            data = json.loads(result['candidates'][0]['content']['parts'][0]['text'])
            state["ai_feedback"] = data.get("feedback", state["ai_feedback"])
            if data.get("is_completed") and state["current_step_idx"] < len(steps)-1:
                state["current_step_idx"] += 1
    except: pass
    finally:
        state["processing_tasks"] -= 1

async def coaching_manager():
    while True:
        if state["is_coaching_active"] and state["latest_frame"] and state["manual_steps"]:
            if state["processing_tasks"] < MAX_CONCURRENT_TASKS:
                asyncio.create_task(run_parallel_analysis(state["latest_frame"], state["manual_steps"], state["current_step_idx"]))
        await asyncio.sleep(1.5)

# --- 매뉴얼 크롭 로직 통합 ---
async def process_manual_logic(files: List[UploadFile]):
    start_time = time.time()
    session_id = f"sess_{int(start_time)}"
    sess_dir = os.path.join(OUTPUT_DIR, session_id)
    os.makedirs(sess_dir, exist_ok=True)
    
    all_steps = []
    for f_idx, file in enumerate(files):
        content = await file.read()
        imgs = convert_from_bytes(content, dpi=150) if file.filename.lower().endswith(".pdf") and PDF_SUPPORT else [Image.open(io.BytesIO(content)).convert("RGB")]
        
        for p_idx, img in enumerate(imgs):
            box_prompt = "Identify assembly steps. Return JSON: {'1': [ymin, xmin, ymax, xmax], ...} (0-1000)"
            box_res = await call_gemini(box_prompt, img)
            if not box_res: continue
            
            boxes = json.loads(box_res['candidates'][0]['content']['parts'][0]['text'])
            w, h = img.size
            for sid in sorted(boxes.keys(), key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 999):
                ymin, xmin, ymax, xmax = boxes[sid]
                crop = img.crop(((xmin*w)/1000, (ymin*h)/1000, (xmax*w)/1000, (ymax*h)/1000))
                fname = f"f{f_idx}_p{p_idx}_s{sid}.jpg"
                crop.save(os.path.join(sess_dir, fname), "JPEG", quality=85)
                
                detail_prompt = "이 단계 분석해서 JSON: {'title': '제목', 'desc': '설명'}"
                detail_res = await call_gemini(detail_prompt, crop)
                if detail_res:
                    detail = json.loads(detail_res['candidates'][0]['content']['parts'][0]['text'])
                    all_steps.append({
                        "title": detail.get("title", f"Step {sid}"),
                        "desc": detail.get("desc", ""),
                        "image_url": f"/outputs/{session_id}/{fname}"
                    })
    
    state["manual_steps"] = all_steps
    state["is_analyzed"] = True
    state["analysis_time"] = round(time.time() - start_time, 2)
    print(f"✅ 분석 완료: {state['analysis_time']}초")
    return all_steps

@app.on_event("startup")
async def startup(): asyncio.create_task(coaching_manager())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state["is_coaching_active"] = True
    try:
        while True:
            state["latest_frame"] = await websocket.receive_bytes()
    except WebSocketDisconnect: state["is_coaching_active"] = False

@app.post("/process-manual")
async def handle_manual(files: List[UploadFile] = File(...)):
    steps = await process_manual_logic(files)
    return {"status": "success", "steps": steps}

@app.get("/status")
async def get_status():
    # 프론트엔드와 이름 통일 (steps, current_step_idx, ai_feedback)
    return {
        "ai_feedback": state["ai_feedback"],
        "steps": state["manual_steps"],
        "current_step_idx": state["current_step_idx"],
        "is_analyzed": state["is_analyzed"],
        "has_frame": state["latest_frame"] is not None,
        "processing_tasks": state["processing_tasks"],
        "analysis_time": state["analysis_time"]
    }

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
async def serve(): return FileResponse(os.path.join(BASE_DIR, "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)