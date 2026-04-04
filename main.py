import os
import asyncio
import uvicorn
import json
import base64
import httpx
import time
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# API 설정 (지침에 따라 키는 빈 문자열로 설정, 실행 환경에서 주입됨)
API_KEY = ""
MODEL_NAME = "gemini-3-flash-preview"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 전역 상태 관리
state = {
    "latest_frame": None,
    "manual_steps": [],
    "current_step_idx": 0,
    "ai_feedback": "매뉴얼을 업로드하면 코칭이 시작됩니다.",
    "is_analyzed": False,
    "is_coaching_active": False,
    "last_analysis_time": 0
}

# --- Gemini API 호출 유틸리티 (지수 백오프 적용) ---
async def call_gemini(payload: dict):
    retries = 5
    for i in range(retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(GEMINI_URL, json=payload, timeout=30.0)
                if response.status_code == 200:
                    return response.json()
        except Exception:
            pass
        await asyncio.sleep(2 ** i) # 1s, 2s, 4s, 8s, 16s
    return None

# --- 실시간 코칭 루프 (백그라운드) ---
async def coaching_loop():
    while True:
        if state["is_coaching_active"] and state["latest_frame"] and state["manual_steps"]:
            # 3초마다 한 번씩 AI 분석 (비용 및 부하 절감)
            current_time = time.time()
            if current_time - state["last_analysis_time"] > 3.0:
                state["last_analysis_time"] = current_time
                
                current_step = state["manual_steps"][state["current_step_idx"]]
                frame_b64 = base64.b64encode(state["latest_frame"]).decode('utf-8')

                prompt = f"""
                You are a professional assembly coach. 
                Current Goal: {current_step['title']} - {current_step['desc']}
                Look at the user's live camera feed and provide feedback in Korean.
                If the user completed the step, set 'completed' to true.
                Respond ONLY in JSON format: {{"feedback": "string", "completed": boolean}}
                """

                payload = {
                    "contents": [{
                        "parts": [
                            {"text": prompt},
                            {"inlineData": {"mimeType": "image/jpeg", "data": frame_b64}}
                        ]
                    }],
                    "generationConfig": {
                        "responseMimeType": "application/json"
                    }
                }

                result = await call_gemini(payload)
                if result:
                    try:
                        text = result['candidates'][0]['content']['parts'][0]['text']
                        res_json = json.loads(text)
                        state["ai_feedback"] = res_json.get("feedback", "")
                        if res_json.get("completed") and state["current_step_idx"] < len(state["manual_steps"]) - 1:
                            state["current_step_idx"] += 1
                            state["ai_feedback"] = f"축하합니다! 다음 단계로 넘어갑니다: {state['manual_steps'][state['current_step_idx']]['title']}"
                    except:
                        pass
        await asyncio.sleep(0.5)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(coaching_loop())

# --- API 엔드포인트 ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            state["latest_frame"] = data
            state["is_coaching_active"] = True
    except WebSocketDisconnect:
        state["latest_frame"] = None
        state["is_coaching_active"] = False

@app.post("/process-manual")
async def process_manual(files: List[UploadFile] = File(...)):
    image_parts = []
    for file in files:
        content = await file.read()
        image_parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": base64.b64encode(content).decode('utf-8')
            }
        })

    prompt = """
    Analyze these assembly manual images. 
    Extract a logical sequence of steps for the user to follow.
    Respond ONLY in JSON format: {"steps": [{"title": "string", "desc": "string"}]}
    The response must be in Korean.
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}] + image_parts}],
        "generationConfig": {"responseMimeType": "application/json"}
    }

    result = await call_gemini(payload)
    if result:
        try:
            text = result['candidates'][0]['content']['parts'][0]['text']
            data = json.loads(text)
            state["manual_steps"] = data.get("steps", [])
            state["is_analyzed"] = True
            state["current_step_idx"] = 0
            state["ai_feedback"] = "분석 완료! 조립을 시작하세요."
            return {"status": "success", "steps": state["manual_steps"]}
        except:
            return {"status": "error", "message": "JSON 파싱 실패"}
    
    return {"status": "error", "message": "AI 응답 실패"}

@app.get("/status")
async def get_status():
    return {
        "ai_response": state["ai_feedback"],
        "has_frame": state["latest_frame"] is not None,
        "steps": state["manual_steps"],
        "current_idx": state["current_step_idx"],
        "is_analyzed": state["is_analyzed"]
    }

@app.get("/stream")
async def stream():
    async def gen():
        while True:
            frame = state["latest_frame"]
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.06)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
@app.get("/mobile")
async def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

if os.path.exists(os.path.join(BASE_DIR, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(BASE_DIR, "assets")), name="assets")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)