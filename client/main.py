import cv2
import requests
import threading
import time

# SSH 터널링 주소
SERVER_URL = "http://127.0.0.1:8000"
YOLO_URL = f"{SERVER_URL}/detect"
VLM_URL = f"{SERVER_URL}/analyze"

latest_detections = []
is_analyzing = False

def request_yolo(frame):
    global latest_detections
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(
            YOLO_URL,
            files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
            timeout=1
        )
        if response.status_code == 200:
            latest_detections = response.json().get("detections", [])
    except:
        pass

def request_vlm(frame):
    global is_analyzing
    is_analyzing = True
    print("\n🚀 분석 요청 중...")
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(
            VLM_URL,
            files={"file": ("capture.jpg", img_encoded.tobytes(), "image/jpeg")}
        )
        result = response.json().get("result", "실패")
        print(f"✅ AI 분석 결과: {result}")
    except Exception as e:
        print(f"❌ 오류: {e}")
    finally:
        is_analyzing = False

def main():
    # 2. 카메라 연결
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # 해상도 조절 (전송 속도 최적화)
    cap.set(3, 640)
    cap.set(4, 480)
    
    frame_count = 0
    print(f"서버 연결: {SERVER_URL}")
    print("탐지 시작! 종료하려면 화면을 클릭하고 'q', VLM 분석은 'Spacebar'")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 거울 모드 (좌우 반전)
        frame = cv2.flip(frame, 1)

        # 1. 추론 요청 (서버로 전송, 3프레임마다)
        if frame_count % 3 == 0:
            threading.Thread(target=request_yolo, args=(frame.copy(),), daemon=True).start()

        # 2. 결과 시각화 및 좌표 출력
        for det in latest_detections:
            label = det['label']
            # 서버에서 xywh(중심x, 중심y, 너비, 높이)로 보냄
            cx, cy, w, h = map(int, det['bbox'])
            
            # 그리기 좌표 계산 (좌상단, 우하단)
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)

            # 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # ★ 다연님이 원하시던 좌표 출력 로그
            # (너무 빠르게 출력되면 정신없으니 프레임 카운트로 조절)
            if frame_count % 30 == 0: 
                 print(f"감지됨: {label} -> 위치: ({cx}, {cy})")

        # 분석 중 표시
        if is_analyzing:
            cv2.putText(frame, "Analyzing...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 화면 출력
        cv2.imshow("Hand & Phone Detector (Client)", frame)

        key = cv2.waitKey(1)
        if key == ord('q'): break
        elif key == 32 and not is_analyzing: # Spacebar
            threading.Thread(target=request_vlm, args=(frame.copy(),), daemon=True).start()
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()