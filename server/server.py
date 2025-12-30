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
def detect_object(user_id: str = "Unknown", file: UploadFile = File(...)):
    # 1. 이미지 읽기
    file_bytes = file.file.read()
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
    
    return {
        "detections": detections, 
        "server_time": duration 
    }

@app.post("/analyze")
def analyze_image(user_id: str = "Unknown", file: UploadFile = File(...)):
    print(f"\n🤖 [{user_id}] VLM 정밀 분석 요청 도착!")
    try:
        image = Image.open(io.BytesIO(file.file.read()))
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
