import cv2
import requests
import threading
import time
import urllib3
from ultralytics import YOLOWorld

# 보안 경고 끄기
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SERVER_URL = "https://milton-nitrogen-asp-anthropology.trycloudflare.com"
USER_NAME = "Dayeon"

VLM_URL = f"{SERVER_URL}/analyze"
HEADERS = {"ngrok-skip-browser-warning": "true"}

# 세션 유지 (연결 속도 향상)
session = requests.Session()

# 로컬 AI 모델 로드
print(f"⏳ [{USER_NAME}] 로컬 AI 모델(YOLO-World) 로딩 중... (손, 사람, 폰)")
local_model = YOLOWorld('yolov8s-world.pt')
local_model.set_classes(["person", "hand", "cell phone", "cup"])

is_vlm_running = False

def request_vlm(frame):
    global is_vlm_running
    is_vlm_running = True
    print(f"\n🚀 [{USER_NAME}] VLM 분석 요청 중... (서버로 전송)")
    
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        _, img_encoded = cv2.imencode('.jpg', frame, encode_param)
        
        response = session.post(
            VLM_URL,
            files={"file": ("capture.jpg", img_encoded.tobytes(), "image/jpeg")},
            params={"user_id": USER_NAME},
            headers=HEADERS,
            verify=False,
            timeout=10 
        )
        
        if response.status_code == 200:
            result = response.json().get("result", "분석 실패")
            print(f"✅ VLM 분석 결과: {result}")
        else:
            print(f"⚠️ 서버 응답 코드: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 전송 오류: {e}")
    finally:
        is_vlm_running = False

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return
    cap.set(3, 640); cap.set(4, 480)
    
    print("=========================================")
    print(f"📡 서버 연결: {SERVER_URL}")
    print("⚡ 하이브리드 모드 실행 중")
    print("=========================================")

    prev_time = 0 

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # 1. 추론 및 속도 측정
        t0 = time.time()
        results = local_model.predict(frame, conf=0.3, verbose=False)
        t1 = time.time()
        
        inference_time = (t1 - t0) * 1000 
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != curr_time else 0
        prev_time = curr_time
        
        # ★ [여기!] 로그 출력을 다시 켰습니다.
        print(f"⚡ YOLO: {inference_time:.1f}ms | 📺 FPS: {fps:.1f}")

        # 2. 결과 그리기
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = local_model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            caption = f"{label} {conf:.2f}"
            cv2.putText(frame, caption, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 3. 화면 정보 표시
        cv2.putText(frame, f"YOLO: {inference_time:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if is_vlm_running:
            cv2.putText(frame, "Analyzing...", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(frame, (30, 390), 10, (0, 0, 255), -1)

        cv2.imshow("Client", frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'): break
        elif key == 32 and not is_vlm_running:
            threading.Thread(target=request_vlm, args=(frame.copy(),), daemon=True).start()
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()