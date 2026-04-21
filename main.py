import io
import os
import json
import time
import base64
import asyncio
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

state = {
    "latest_frame": None,
    "ai_feedback": "모바일 연결 대기 중",
    "has_frame": False
}

frame_count = 0
test_count = 0
baseline_offset = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global frame_count, test_count, baseline_offset
    log_file_path = os.path.join(OUTPUT_DIR, "latency_logs.txt")
    
    try:
        while True:
            # 1. 메시지 수신
            message = await websocket.receive_text()
            data = json.loads(message)
            
            # 🌟 [즉시 응답] 받은 즉시 보낸 시각을 다시 돌려줌 (모바일에서 RTT 계산용)
            await websocket.send_text(json.dumps({"pong": data.get("send_time")}))
            
            recv_time = time.time()
            raw_latency = recv_time - (data.get("send_time") / 1000)
            mobile_rtt = data.get("rtt", 0) # 모바일이 계산해서 보낸 이전 프레임의 RTT
            
            # 이미지 갱신
            header, encoded = data["data"].split(",", 1)
            frame_bytes = base64.b64decode(encoded)
            state["latest_frame"] = frame_bytes
            state["has_frame"] = True
            frame_count += 1

            if frame_count >= 50 and test_count < 100:
                if baseline_offset is None: baseline_offset = raw_latency
                adj_latency = raw_latency - baseline_offset
                scale = 100 - (test_count // 10) * 10
                test_count += 1
                
                # 🌟 Absolute Latency 추출: RTT의 절반을 대략적인 편도 지연으로 간주
                absolute_latency = mobile_rtt / 2
                
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img_cv is not None:
                    h, w, _ = img_cv.shape
                    cur_w, cur_h = int(w * scale / 100), int(h * scale / 100)
                    
                    # 🌟 로그 메시지에 절대 지연 시간(ABS) 추가
                    res_msg = f"Test #{test_count:3} | Scale: {scale:3}% | Res: {cur_w:4}x{cur_h:4} | ABS: {round(absolute_latency, 2):5}ms | Adj: {round(adj_latency * 1000, 2):6}ms"
                    state["ai_feedback"] = res_msg
                    
                    with open(log_file_path, "a") as f:
                        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {res_msg}\n")
                    print(f"🚀 {res_msg}")

                # Buffer Drain
                try:
                    while True: await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                except asyncio.TimeoutError: pass

    except WebSocketDisconnect:
        state["has_frame"] = False
        state["ai_feedback"] = "연결 종료됨"

@app.get("/status")
async def get_status(): return state

@app.get("/stream")
async def stream():
    async def gen():
        while True:
            if state["latest_frame"]:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + state["latest_frame"] + b"\r\n")
            await asyncio.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
@app.get("/mobile")
async def serve(): return FileResponse(os.path.join(BASE_DIR, "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)