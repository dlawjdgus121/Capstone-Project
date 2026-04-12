import io, os, re, json, time, base64, asyncio, tempfile, httpx, uvicorn
from typing import List
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

try:
    import opendataloader_pdf
except ImportError:
    print("⚠️ 'pip install opendataloader-pdf'가 필요합니다.")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-09-2025")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

state = {
    "latest_frame": None,
    "manual_steps": [],
    "current_step_idx": 0,
    "ai_feedback": "시스템 준비 완료.",
    "is_analyzed": False,
    "analysis_time": 0.0
}

# --- 유틸리티: 자연어 정렬 (파일명 순서 기초 정렬용) ---
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# --- AI 지능형 이미지 분석 및 순서 판별 ---
async def analyze_step_from_image(client, img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((1024, 1024))
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        # [AI 미션] 이미지 속 숫자를 읽어 순서를 결정하고 설명을 작성
        prompt = """
        당신은 매뉴얼 시각 분석 전문가입니다. 전송된 이미지를 보고 조립 단계를 추출하세요.

        [수행 작업]
        1. 이미지 내 텍스트 판독: 이미지 안에 'Step 1', '①', '1.' 처럼 단계를 나타내는 숫자가 있는지 확인하세요.
        2. 유효성 검사: 실제 조립 동작이 포함된 그림인가요? 로고나 빈 페이지라면 'is_valid': false로 답하세요.
        3. 순서 결정: 이미지 속 숫자 정보를 바탕으로 'step_number'를 숫자로 입력하세요. (예: Step 1 -> 1)
        4. 내용 요약: 그림에 나타난 조립 동작과 지시사항 텍스트를 분석하여 제목과 설명을 한국어로 작성하세요.

        반드시 JSON으로 응답하세요:
        {
          "is_valid": bool,
          "step_number": int,
          "title": "동작 제목",
          "desc": "상세 조립 가이드"
        }
        """
        
        payload = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}
            ]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }

        res = await client.post(GEMINI_URL, json=payload, timeout=40.0)
        if res.status_code == 200:
            data = json.loads(res.json()['candidates'][0]['content']['parts'][0]['text'])
            if data.get("is_valid") is True:
                rel_path = os.path.relpath(img_path, OUTPUT_DIR)
                return {
                    "step_number": data.get("step_number", 999), # 숫자가 없으면 뒤로 보냄
                    "title": data.get("title", "조립 단계"),
                    "desc": data.get("desc", ""),
                    "image_url": f"/outputs/{rel_path}"
                }
    except: pass
    return None

async def process_manual_logic(files: List[UploadFile]):
    start_time = time.time()
    state["manual_steps"] = []
    state["is_analyzed"] = False
    
    sess_id = f"sess_{int(start_time)}"
    sess_dir = os.path.join(OUTPUT_DIR, sess_id)
    raw_dir = os.path.join(sess_dir, "raw"); proc_dir = os.path.join(sess_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True); os.makedirs(proc_dir, exist_ok=True)
    
    paths = []
    for f in files:
        p = os.path.join(raw_dir, f.filename)
        with open(p, "wb") as b: b.write(await f.read())
        paths.append(p)

    # 1. 오픈로더 전처리 (이미지 조각 추출)
    print("🚀 [Loader] 이미지 조각 추출 중...")
    opendataloader_pdf.convert(input_path=paths, output_dir=proc_dir, format="json")

    img_files = []
    for root, _, fnames in os.walk(proc_dir):
        for fn in fnames:
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                img_files.append(os.path.join(root, fn))
    
    # 2. 제미나이 병렬 분석 (이미지 내 숫자로 순서 찾기)
    print(f"🧠 [AI] {len(img_files)}개 조각 분석 및 이미지 기반 순서 정렬 시작...")
    async with httpx.AsyncClient() as client:
        tasks = [analyze_step_from_image(client, path) for path in img_files]
        results = await asyncio.gather(*tasks)

    # 3. 유효한 단계만 필터링 및 AI가 판별한 step_number 기준으로 최종 정렬
    valid_steps = [r for r in results if r is not None]
    # step_number 기준으로 정렬 (이미지 속에 써있는 숫자 순서대로)
    valid_steps.sort(key=lambda x: x['step_number'])

    state["manual_steps"] = valid_steps
    state["is_analyzed"] = True
    state["analysis_time"] = round(time.time() - start_time, 2)
    print(f"✅ 분석 완료! 총 {len(valid_steps)}개 단계가 이미지 내 번호 순으로 정렬되었습니다.")
    return valid_steps

@app.post("/process-manual")
async def handle_manual(files: List[UploadFile] = File(...)):
    steps = await process_manual_logic(files)
    return {"status": "success", "steps": steps}

@app.get("/status")
async def get_status():
    return {
        "current_idx": state["current_step_idx"],
        "ai_response": state.get("last_ai_feedback", "대기 중..."),
        "steps": state["manual_steps"],
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
async def serve(): return FileResponse("index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)