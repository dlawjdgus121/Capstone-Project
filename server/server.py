from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import io

app = FastAPI()
#http://127.0.0.1:8000/docs
# ==========================================
# [1] AI 모델 로드 (서버 켤 때 딱 한 번 실행)
# ==========================================
print("⏳ AI 모델을 다운로드/로드 중입니다... (처음엔 오래 걸림)")

# 사용할 모델 이름 (2B는 가볍고 빠름, 성능을 원하면 7B로 변경 가능)
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

try:
    # 모델 불러오기 (GPU가 있으면 자동으로 GPU 사용)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto" 
    )
    
    # 프로세서(이미지 처리기) 불러오기
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("✅ 모델 로드 완료! 이제 똑똑해졌습니다.")
    
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    print("혹시 메모리가 부족하거나 라이브러리가 꼬였을 수 있습니다.")

# ==========================================
# [2] 통신 API 정의
# ==========================================

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    print(f"📸 이미지 수신: {file.filename} -> 분석 시작")

    # 1. 이미지 파일 읽기
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # 2. AI에게 질문할 내용 (Prompt)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in detail within 2 sentences."} 
                # (해석: 이 이미지를 2문장 이내로 자세히 설명해줘)
            ],
        }
    ]

    # 3. 전처리 (Preprocessing)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # 4. 추론 (Inference) - AI가 생각하는 시간
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    # 5. 결과 해석 (Decoding)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(f"🤖 AI 답변: {output_text}")

    return {
        "status": "success",
        "result": output_text
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)