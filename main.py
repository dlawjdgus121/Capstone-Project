import io
import os
import json
import time
import asyncio
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
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

# 시스템 상태 및 실험 변수
state = {
    "latest_frame": None,
    "log": "모바일 연결 대기 중",
    "has_frame": False
}

frame_count = 0
test_count = 0
baseline_offset = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global frame_count, test_count, baseline_offset
    log_path = os.path.join(OUTPUT_DIR, "binary_fhd_abs_metrics.txt")
    
    try:
        while True:
            # 1. Binary 수신 (16바이트 헤더 + 이미지 데이터)
            data = await websocket.receive_bytes()
            recv_time = time.time() * 1000 
            
            # 헤더 파싱 (앞 8B: 발송시각, 뒤 8B: 클라이언트 측정 이전 RTT)
            header = np.frombuffer(data[:16], dtype=np.float64)
            client_send_time = header[0]
            last_rtt = header[1]
            
            image_bytes = data[16:]
            byte_size = len(data)

            # 2. RTT 측정을 위한 즉시 응답 (클라이언트가 보낸 시각만 반환)
            await websocket.send_bytes(data[:8])

            # 3. 이미지 디코딩 및 상태 업데이트
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if img is not None:
                h, w, _ = img.shape
                state["latest_frame"] = image_bytes
                state["has_frame"] = True
                frame_count += 1

                # 4. 🔥 100회 실험 로직 (50프레임 워밍업 후 10%씩 해상도 축소)
                if frame_count >= 50 and test_count < 100:
                    raw_latency = recv_time - client_send_time
                    if baseline_offset is None:
                        baseline_offset = raw_latency
                    
                    scale = 100 - (test_count // 10) * 10
                    adj_latency = raw_latency - baseline_offset
                    test_count += 1
                    
                    # 🌟 ABS 계산: 클라이언트가 보고한 RTT / 2 (가장 정확한 편도 지연)
                    abs_latency = last_rtt / 2
                    
                    # 목표 해상도 계산 및 리사이징 성능 측정
                    cur_w, cur_h = int(w * scale / 100), int(h * scale / 100)
                    start_resize = time.time()
                    if scale < 100:
                        cv2.resize(img, (cur_w, cur_h))
                    process_ms = (time.time() - start_resize) * 1000

                    # 로그 생성 및 저장
                    log_entry = (
                        f"Test #{test_count:3} | Scale: {scale:3}% | Res: {cur_w:4}x{cur_h:4} | "
                        f"ABS: {round(abs_latency, 1):5}ms | Adj: {round(adj_latency, 1):6}ms | Bytes: {byte_size:7}"
                    )
                    state["log"] = log_entry
                    
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {log_entry}\n")
                    print(f"🚀 {log_entry}")

                # 5. Buffer Drain: 네트워크 밀림 방지
                try:
                    while True: await asyncio.wait_for(websocket.receive_bytes(), timeout=0.001)
                except: pass

    except WebSocketDisconnect:
        state["has_frame"] = False
        state["log"] = "연결 종료됨"

@app.get("/status")
async def get_status():
    return {"log": state["log"]}

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
async def serve():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)