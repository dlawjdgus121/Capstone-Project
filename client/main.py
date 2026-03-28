import asyncio
import cv2
import time
import datetime
import urllib3
import base64
import numpy as np
import httpx
import uvicorn
import os
import webbrowser
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 🚨 반드시 본인의 Cloudflare URL로 맞춰주세요!
GPU_VLM_URL = "https://brothers-reporters-sitemap-logs.trycloudflare.com/vlm"
HEADERS = {"ngrok-skip-browser-warning": "true", "Connection": "close"}

CAPTURE_DIR = "captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates")

latest_frame_bytes = None
is_vlm_streaming = False
fps_start_time = time.time()
fps_counter = 0

blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(blank_img, "Waiting for mobile camera...", (70, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
_, blank_buffer = cv2.imencode('.jpg', blank_img)
blank_frame_bytes = blank_buffer.tobytes()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global latest_frame_bytes, fps_counter, fps_start_time
    await websocket.accept()
    print("\n🟢 [LOG] 모바일 카메라 연결 완료! (버퍼링 제로 모드)")
    try:
        while True:
            data = await websocket.receive_bytes()
            latest_frame_bytes = data
            
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed > 2.0:
                nparr = np.frombuffer(data, np.uint8)
                temp_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                h, w = (temp_frame.shape[0], temp_frame.shape[1]) if temp_frame is not None else (0,0)
                mb_size = len(data) / (1024 * 1024)
                
                print(f"📡 [캠 수신] {w}x{h} | 전송속도: {fps_counter/elapsed:.1f} 장/초 | {mb_size:.2f} MB")
                fps_counter = 0
                fps_start_time = time.time()
    except WebSocketDisconnect:
        print("🔴 [LOG] 모바일 연결 끊김.")
        latest_frame_bytes = None

async def vlm_background_worker():
    global is_vlm_streaming, latest_frame_bytes
    print("🤖 [SYSTEM] VLM 분석 엔진 대기 중...")
    
    current_step = 1
    total_steps = 5

    while True:
        if is_vlm_streaming and latest_frame_bytes is not None:
            try:
                web_start_time = time.time()
                current_bytes = latest_frame_bytes 
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(CAPTURE_DIR, f"vlm_capture_{timestamp}.jpg")
                with open(filename, "wb") as f:
                    f.write(current_bytes)

                img_b64 = base64.b64encode(current_bytes).decode('utf-8')
                payload = {"message": "현재 상황을 분석해줘", "image": img_b64}
                print(f"\n🚀 [분석 요청] GPU 서버로 이미지 전송 중...")

                async with httpx.AsyncClient() as client:
                    net_start_time = time.time()
                    resp = await client.post(GPU_VLM_URL, json=payload, headers=HEADERS, timeout=60.0)
                    net_end_time = time.time()
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        response_text = result.get("gpu_response", "응답 없음")
                        gpu_internal_time = result.get("gpu_time", 0.0) 
                        
                        rtt = net_end_time - net_start_time
                        pure_network_delay = rtt - gpu_internal_time
                        total_elapsed = time.time() - web_start_time
                        
                        if "[성공]" in response_text:
                            status_msg = "✅ 단계 통과"
                            current_step += 1
                        else:
                            status_msg = "❌ 코칭 진행 중"

                        print("-" * 55)
                        print(f"📊 [시스템 지연 시간 분석 보고서]")
                        print(f"  └ 🧠 GPU AI 연산 시간 : {gpu_internal_time:.2f}초")
                        print(f"  └ 🌐 순수 네트워크 지연 : {max(0, pure_network_delay):.2f}초 (인터넷 왕복)")
                        print(f"  └ ⌛ 메인 서버 총 처리 : {total_elapsed:.2f}초")
                        print(f"🏁 결과: {status_msg} | 단계: ({current_step}/{total_steps})")
                        print(f"💬 AI 응답: {response_text}")
                        print("-" * 55)
                    else:
                        print(f"❌ [GPU 에러] 상태 코드: {resp.status_code}")
            except Exception as e:
                print(f"🔥 [연결 에러] 통신 실패: {e}")
            
            # ⏱️ [핵심] VLM 처리 시간을 고려하여 다음 요청까지 5초 대기
            await asyncio.sleep(5.0) 
        else:
            await asyncio.sleep(1.0)

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(vlm_background_worker())

if os.path.isdir("dist"):
    app.mount("/assets", StaticFiles(directory="dist/assets"), name="assets")

@app.get("/")
async def serve_index():
    return FileResponse("dist/index.html") if os.path.exists("dist/index.html") else {"error": "dist 폴더 없음"}

@app.get("/mobile")
async def serve_mobile(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/stream")
async def stream():
    async def gen():
        while True:
            frame = latest_frame_bytes if latest_frame_bytes is not None else blank_frame_bytes
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.01)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/vlm/start")
def vlm_on():
    global is_vlm_streaming
    is_vlm_streaming = True
    print("\n▶️ [명령] 실시간 VLM 분석 시작")
    return {"status": "ok"}

@app.post("/vlm/stop")
def vlm_off():
    global is_vlm_streaming
    is_vlm_streaming = False
    print("\n⏹️ [명령] 실시간 VLM 분석 중지")
    return {"status": "ok"}

@app.post("/start")
def start_app(): return {"ok": True}

@app.post("/stop")
def stop_app(): return {"ok": True}

if __name__ == "__main__":
    def open_browser():
        time.sleep(2)
        webbrowser.open("http://127.0.0.1:8001")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)