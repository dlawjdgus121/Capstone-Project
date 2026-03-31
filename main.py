import io, os, re, json, time, base64, asyncio, tempfile, httpx, uvicorn
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 정적 파일 설정
if os.path.exists("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

BASE_TEMP_DIR = os.path.join(tempfile.gettempdir(), "ai_manual_system")
os.makedirs(BASE_TEMP_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=BASE_TEMP_DIR), name="outputs")

# 시스템 전역 상태
state = {
    "latest_frame": None,
    "is_vlm_active": False,
    "manual_steps": [],
    "current_step_idx": 0,
    "last_ai_response": "카메라 연결 대기 중...",
}

# --- 경로 설정 ---

@app.get("/")
async def serve_pc():
    """PC용 대시보드 (index.html)"""
    return FileResponse("index.html")

@app.get("/mobile")
async def serve_mobile():
    """모바일용 카메라 (index.html - 프론트엔드에서 경로 분기 처리)"""
    return FileResponse("index.html")

# --- 통신 엔드포인트 ---

@app.get("/status")
async def get_status():
    return {
        "current_idx": state["current_step_idx"],
        "ai_response": state["last_ai_response"],
        "is_active": state["is_vlm_active"],
        "has_frame": state["latest_frame"] is not None
    }

@app.post("/vlm/toggle")
async def vlm_toggle(active: bool):
    state["is_vlm_active"] = active
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 [LOG] 모바일 카메라 장치 연결됨")
    try:
        while True:
            data = await websocket.receive_bytes()
            state["latest_frame"] = data
    except WebSocketDisconnect:
        print("🔴 [LOG] 모바일 카메라 연결 끊김")
        state["latest_frame"] = None

@app.get("/stream")
async def stream():
    async def gen():
        while True:
            frame = state["latest_frame"]
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.04)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/process-manual")
async def process_manual(files: List[UploadFile] = File(...)):
    # 테스트용 데이터 (실제 로직은 Gemini 분석 결과를 리스트로 저장)
    state["manual_steps"] = [
        {"display_id": "Step 1", "action": "본체 준비", "expected": "본체가 수평으로 놓임"},
        {"display_id": "Step 2", "action": "나사 체결", "expected": "나사 4개가 모두 박힘"}
    ]
    state["current_step_idx"] = 0
    return {"status": "success", "steps": state["manual_steps"]}

# VLM 분석 워커 (가상 로직)
async def vlm_worker():
    while True:
        if state["is_vlm_active"] and state["latest_frame"]:
            # 여기서 실제 GPU VLM 서버와 통신하여 결과를 state["last_ai_response"]에 저장
            pass
        await asyncio.sleep(2.0)

@app.on_event("startup")
async def startup():
    asyncio.create_task(vlm_worker())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)