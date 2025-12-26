from fastapi import FastAPI, UploadFile, File
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLOWorld
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import io

app = FastAPI()

# ==========================================
# [1] 모델 설정 (다연님 코드 반영)
# ==========================================
print("⏳ 모델을 로딩 중입니다...")

# 1. YOLO-World 로드
yolo_model = YOLOWorld('yolov8s-world.pt')

# ★ 다연님이 원하시는 탐지 객체 설정
target_classes = ["hand", "cell phone"]
yolo_model.set_classes(target_classes)
print(f"✅ 설정 완료: {target_classes}만 집중적으로 탐지합니다.")

# 2. VLM 로드 (나중을 위해 유지)
print("⏳ VLM 모델 로딩 중...")
vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
print("✅ 서버 준비 완료!")

# ==========================================
# [2] API 정의
# ==========================================
@app.post("/detect")
async def detect_object(file: UploadFile = File(...)):
    # 이미지 읽기
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 추론 수행
    results = yolo_model.predict(img, conf=0.1, verbose=False)
    
    # 결과 포장 (좌표 및 라벨)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id] # hand, cell phone 등
        bbox = box.xywh[0].tolist() # x, y, w, h
        
        detections.append({
            "label": label,
            "bbox": bbox
        })
    
    return {"status": "success", "detections": detections}

# VLM용 API (유지)
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this scene detail."}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=process_vision_info(messages)[0], padding=True, return_tensors="pt").to(vlm_model.device)
    generated_ids = vlm_model.generate(**inputs, max_new_tokens=100)
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
    return {"status": "success", "result": output_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)