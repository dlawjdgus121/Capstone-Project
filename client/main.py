import cv2
import requests
import time

# ★ 중요: 서버 주소 (로컬에서 테스트할 땐 localhost)
# 만약 팀원이 서로 다른 집에서 한다면 ngrok 주소나 공인 IP를 적어야 함
SERVER_URL = "http://127.0.0.1:8000/analyze"

def send_frame_to_server(frame):
    # 1. 이미지를 메모리 상에서 인코딩 (파일로 저장 안 하고 바로 보냄)
    _, img_encoded = cv2.imencode('.jpg', frame)
    
    # 2. 서버로 전송 (POST 요청)
    print("🚀 서버로 이미지 전송 중...")
    try:
        response = requests.post(
            SERVER_URL,
            files={"file": ("capture.jpg", img_encoded.tobytes(), "image/jpeg")}
        )
        # 3. 응답 출력
        print(f"✅ 서버 응답: {response.json()}")
    except Exception as e:
        print(f"❌ 연결 실패: {e}")

def main():
    cap = cv2.VideoCapture(0) # 웹캠 켜기
    print("캠이 켜졌습니다. 스페이스바를 누르면 서버로 전송합니다. (q는 종료)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 거울 모드
        frame = cv2.flip(frame, 1)

        # 화면에 안내 문구 띄우기
        cv2.putText(frame, "Press SPACE to Send", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Client Cam", frame)

        key = cv2.waitKey(1)
        if key == ord('q'): # q 누르면 종료
            break
        elif key == 32: # 스페이스바(Space) 누르면 전송
            send_frame_to_server(frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()