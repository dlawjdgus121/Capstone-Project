import os
import httpx
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

print("\n=== 🔍 현재 사용 가능한 Gemini 모델 조회 ===")
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

try:
    res = httpx.get(url, timeout=10.0)
    if res.status_code == 200:
        models = res.json().get("models", [])
        print("✅ 텍스트/이미지 분석(generateContent) 지원 모델 목록:\n")
        
        available_models = []
        for m in models:
            if "generateContent" in m.get("supportedGenerationMethods", []):
                # 모델 이름에서 'models/' 부분 제거 후 출력
                clean_name = m['name'].replace("models/", "")
                available_models.append(clean_name)
                print(f"- {clean_name}")
                
        if not available_models:
            print("🚨 앗, 현재 API 키로 텍스트 생성을 지원하는 모델이 하나도 없다고 나옵니다.")
    else:
        print(f"🚨 목록 조회 실패 ({res.status_code}): {res.text}")
except Exception as e:
    print(f"🚨 통신 에러: {e}")
print("\n============================================\n")