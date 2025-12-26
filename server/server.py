from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLOWorld
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import io
import torch
from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLOWorld
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import io
import torch
import time

app = FastAPI()

print("=========================================")
print("⏳ [1/2] YOLO-World 로딩 중... (손, 핸드폰)")
yolo_model = YOLOWorld('yolov8s-world.pt')
yolo_model.set_classes(["hand", "cell phone"])

print("⏳ [2/2] VLM (Qwen2-VL) 로딩 중... (시간 소요됨)")
vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    torch_dtype="auto", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("✅ 서버 준비 완료! (8000번 포트 대기 중)")
print("=========================================")

@app.post("/detect")
async def detect_object(user_id: str = "Unknown", file: UploadFile = File(...)):
    # 1. 이미지 읽기
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. 추론 및 시간 측정
    start = time.time()
    results = yolo_model.predict(img, conf=0.1, verbose=False)
    duration = (time.time() - start) * 1000
    
    # 로그 출력 (사용자 ID 포함)
    print(f"⚡ [{user_id}] YOLO 요청 처리: {duration:.2f} ms")

    # 3. 결과 포장
    detections = []
    for box in results[0].boxes:
        bbox = box.xywh[0].tolist() # 중심x, 중심y, 너비, 높이
        label = yolo_model.names[int(box.cls[0])]
        detections.append({"label": label, "bbox": bbox})
    
    return {"detections": detections}

@app.post("/analyze")
async def analyze_image(user_id: str = "Unknown", file: UploadFile = File(...)):
    print(f"\n🤖 [{user_id}] VLM 정밀 분석 요청 도착!")
    try:
        image = Image.open(io.BytesIO(await file.read()))
        prompt = "Describe this scene in detail, focusing on what the person is doing with their hands."
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=process_vision_info(messages)[0], padding=True, return_tensors="pt").to(vlm_model.device)
        
        start = time.time()
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=100)
        duration = (time.time() - start) * 1000
        
        result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
        
        print(f"📝 [{user_id}] 분석 완료 ({duration:.2f} ms): {result[:30]}...")
        return {"result": result}
    except Exception as e:
        print(f"❌ 오류: {e}")
        return {"result": "분석 실패"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
app = FastAPI()

print("=========================================")
print("⏳ [1/3] YOLO-World 모델 로딩 중...")
# 1. YOLO-World 로드
yolo_model = YOLOWorld('yolov8s-world.pt')

# ★ 다연님이 원하시는 탐지 객체 설정
target_classes = ["hand", "cell phone"]
yolo_model.set_classes(target_classes)
print(f"✅ YOLO 설정 완료: {target_classes} 탐지 모드")

print("⏳ [2/3] VLM (Qwen2-VL) 모델 로딩 중... (시간 좀 걸림)")
# 2. VLM 로드 (GPU 가속 활성화)
vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", 
    torch_dtype="auto", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("✅ VLM 로딩 완료!")
print("🚀 [3/3] 서버 가동 시작! (8000번 포트)")
print("=========================================")

@app.get("/")
def read_root():
    return {"status": "RunPod Server is Running!"}

@app.post("/detect")
async def detect_object(file: UploadFile = File(...)):
    # 1. 이미지 읽기
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. 추론 수행 (YOLO)
    # conf=0.1: 확신도 10% 이상이면 탐지
    results = yolo_model.predict(img, conf=0.1, verbose=False)
    
    # 3. 결과 포장 (Client가 원하는 포맷으로 변환)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id] # hand, cell phone 등
        
        # xywh: 중심x, 중심y, 너비, 높이 (Client에서 계산하기 편하게)
        bbox = box.xywh[0].tolist() 
        
        detections.append({
            "label": label,
            "bbox": bbox,
            "conf": float(box.conf[0])
        })
    
    return {"status": "success", "detections": detections}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    print("🤖 VLM 분석 요청 수신!")
    try:
        # 1. 이미지 변환 (OpenCV -> PIL)
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # 2. 질문 설정
        prompt = "Describe this scene in detail, focusing on what the person is doing with their hands."
        
        messages = [{
            "role": "user", 
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        }]

        # 3. 모델 입력 준비
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], 
            images=image_inputs, 
            videos=video_inputs, 
            padding=True, 
            return_tensors="pt"
        ).to(vlm_model.device)

        # 4. 답변 생성
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=150)
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
        
        print(f"✅ 분석 완료: {output_text[:50]}...")
        return {"status": "success", "result": output_text}
        
    except Exception as e:
        print(f"❌ VLM 에러: {e}")
        return {"status": "error", "result": "분석 중 오류가 발생했습니다."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)