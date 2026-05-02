import io, os, re, json, time, base64, asyncio, httpx, uvicorn
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import subprocess
import re
import qrcode
import threading
import time
import base64  # 추가: 이미지를 텍스트로 변환
from io import BytesIO # 추가: 메모리 상에서 이미지 처리

state = {
    "latest_frame": None,
    "manual_steps": [],
    "current_step_idx": 0,
    "ai_feedback": "시스템 준비 완료. 매뉴얼을 업로드하세요.",
    "is_analyzed": False,
    "is_coaching_active": False,
    "processing_tasks": 0,
    "analysis_time": 0.0,
    "mobile_w": 0, "mobile_h": 0, "mobile_fps": 0.0,
    "pc_fps": 0.0,
    "tunnel_url": None,
    "mobile_qr": None  # 추가: UI에 보여줄 QR 이미지 데이터를 담을 곳
}

# 1. 클라우드플레어 실행 (동일)
def start_cloudflare_and_qr():
    cmd = ["cloudflared-windows-amd64.exe", "tunnel", "--url", "http://localhost:8000"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')

    print("☁️ 클라우드플레어 터널 연결 중...")
    
    for line in process.stdout:
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            state["tunnel_url"] = match.group()
            print(f"\n✅ 터널 준비 완료: {state['tunnel_url']}")
            break

threading.Thread(target=start_cloudflare_and_qr, daemon=True).start()

# 2. [수정] 터미널 출력 대신 '이미지 데이터'를 생성하는 함수
def generate_qr_base64(url):
    mobile_url = f"{url}/mobile"
    qr = qrcode.QRCode()
    qr.add_data(mobile_url)
    qr.make()
    
    # QR을 이미지로 생성
    img = qr.make_image(fill_color="black", back_color="white")
    
    # 이미지를 텍스트(Base64)로 변환하는 과정
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

# WebRTC 관련 임포트 추가
from aiortc import RTCPeerConnection, RTCSessionDescription

# PDF 지원 체크
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# API 설정
const_apiKey = ""
MODEL_NAME = "gemini-3-flash-preview"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={const_apiKey}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 전역 상태 관리
state = {
    "latest_frame": None,
    "manual_steps": [],
    "current_step_idx": 0,
    "ai_feedback": "시스템 준비 완료. 매뉴얼을 업로드하세요.",
    "is_analyzed": False,
    "is_coaching_active": False,
    "processing_tasks": 0,
    "analysis_time": 0.0,
    "mobile_w": 0, "mobile_h": 0, "mobile_fps": 0.0,
    "pc_fps": 0.0
}

MAX_CONCURRENT_TASKS = 2

# 활성화된 WebRTC 커넥션 관리
pcs = set()

# --- Gemini API & 코칭 루프 ---
async def call_gemini(prompt, pil_image=None):
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    if pil_image:
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        payload["contents"][0]["parts"].append({"inlineData": {"mimeType": "image/jpeg", "data": img_b64}})
    
    payload["generationConfig"] = {"responseMimeType": "application/json"}
    async with httpx.AsyncClient() as client:
        for _ in range(3):
            try:
                res = await client.post(GEMINI_URL, json=payload, timeout=40.0)
                if res.status_code == 200: return res.json()
            except: pass
            await asyncio.sleep(1)
    return None

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
    finally: state["processing_tasks"] -= 1

async def coaching_manager():
    while True:
        if state["is_coaching_active"] and state["latest_frame"] and state["manual_steps"]:
            if state["processing_tasks"] < MAX_CONCURRENT_TASKS:
                asyncio.create_task(run_parallel_analysis(state["latest_frame"], state["manual_steps"], state["current_step_idx"]))
        await asyncio.sleep(1.5)

# --- FastAPI 수명주기 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(coaching_manager())
    yield
    # 서버 종료 시 커넥션 정리
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# --- 매뉴얼 크롭 로직 ---
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
                
                detail_res = await call_gemini("이 단계 분석해서 JSON: {'title': '제목', 'desc': '설명'}", crop)
                if detail_res:
                    detail = json.loads(detail_res['candidates'][0]['content']['parts'][0]['text'])
                    all_steps.append({"title": detail.get("title", f"Step {sid}"), "desc": detail.get("desc", ""), "image_url": f"/outputs/{session_id}/{fname}"})
    
    state["manual_steps"], state["is_analyzed"], state["analysis_time"] = all_steps, True, round(time.time() - start_time, 2)

    if state["tunnel_url"]:
    # 터미널에 찍는 대신 state에 QR 이미지 데이터를 저장!
        state["mobile_qr"] = generate_qr_base64(state["tunnel_url"])

    return all_steps  # 함수의 마지막은 항상 return이어야 합니다.

# --- WebRTC 시그널링 엔드포인트 ---
@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("WebRTC Connection State:", pc.connectionState)
        if pc.connectionState in ["failed", "closed"]:
            pcs.discard(pc)
            state["is_coaching_active"] = False

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            state["is_coaching_active"] = True
            asyncio.create_task(process_webrtc_track(track))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

async def process_webrtc_track(track):
    frame_count = 0
    start_time = time.time()
    
    while True:
        try:
            # WebRTC로 넘어온 프레임 수신
            frame = await track.recv()
            
            # numpy array로 변환 후 JPEG로 압축하여 글로벌 상태에 저장 (PC 뷰어 및 VLM 용)
            img = frame.to_ndarray(format="rgb24")
            pil_img = Image.fromarray(img)
            
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=60)
            state["latest_frame"] = buffered.getvalue()
            
            # FPS 및 해상도 통계 계산
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                state["mobile_fps"] = round(frame_count / elapsed, 1)
                state["mobile_w"] = frame.width
                state["mobile_h"] = frame.height
                frame_count = 0
                start_time = time.time()
                
        except Exception as e:
            print("WebRTC Track Ended:", e)
            break

# --- 기타 엔드포인트 ---
@app.post("/process-manual")
async def handle_manual(files: List[UploadFile] = File(...)):
    steps = await process_manual_logic(files)
    return {"status": "success", "steps": steps}

@app.get("/status")
async def get_status():
    return {
        "ai_feedback": state["ai_feedback"],
        "steps": state["manual_steps"],
        "current_step_idx": state["current_step_idx"],
        "is_analyzed": state["is_analyzed"],
        "mobile_qr": state["mobile_qr"],
        "has_frame": state["latest_frame"] is not None,
        "processing_tasks": state["processing_tasks"],
        "mobile_stats": {"w": state["mobile_w"], "h": state["mobile_h"], "fps": state["mobile_fps"]},
        "pc_fps": state["pc_fps"]
    }

@app.get("/stream")
async def stream():
    async def gen():
        last_frame_id = None
        frame_count = 0
        start_time = time.time()
        
        while True:
            current_frame = state["latest_frame"]
            if current_frame and id(current_frame) != last_frame_id:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + current_frame + b"\r\n")
                last_frame_id = id(current_frame)
                frame_count += 1
                
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                state["pc_fps"] = round(frame_count / elapsed, 1)
                frame_count = 0
                start_time = time.time()
                
            await asyncio.sleep(0.01)
            
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
@app.get("/mobile")
async def serve(): return FileResponse(os.path.join(BASE_DIR, "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)