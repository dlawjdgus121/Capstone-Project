import cv2
import requests
import threading
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
#안녕

# =========================================================
# [설정] 여기만 확인하세요!
# =========================================================
USER_NAME = "Dayeon"  # ★ 팀원은 여기 이름만 바꾸면 됩니다!
SERVER_URL = "https://milton-nitrogen-asp-anthropology.trycloudflare.com"

YOLO_URL = f"{SERVER_URL}/detect"
VLM_URL = f"{SERVER_URL}/analyze"
HEADERS = {"Connection": "keep-alive"}

session = requests.Session()

latest_detections = []
is_vlm_running = False
is_yolo_running = False # ★ 끊김 방지용 깃발

def request_yolo(frame):
    global latest_detections, is_yolo_running
    
    # 이미 분석 중이면 중복 요청 안 함 (끊김 방지 핵심!)
    if is_vlm_running: 
        is_yolo_running = False
        return

    try:
        # [참고] 화질 100은 용량이 커서 네트워크 지연이 높게 나올 겁니다.
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100] 
        _, img_encoded = cv2.imencode('.jpg', frame, encode_param)

        # ★ 1. [수정] 출발 시간 기록 (택배 보내기 전)
        start_time = time.time()

        response = session.post(
            YOLO_URL,
            files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
            params={"user_id": USER_NAME}, 
            headers=HEADERS, 
            timeout=2, 
            verify=False
        )
        
        # ★ 2. [수정] 도착 시간 기록 (택배 받은 후)
        end_time = time.time()

        if response.status_code == 200:
            data = response.json()
            latest_detections = data.get("detections", [])
            
            # ★ 3. [수정] 네트워크 지연 시간 계산 및 로그 출력
            # (총 왕복 시간) - (서버가 일한 시간) = (순수 네트워크 시간)
            total_rtt = (end_time - start_time) * 1000      # ms 단위 변환
            server_proc = data.get("server_time", 0)        # 서버에서 받아온 시간
            network_delay = total_rtt - server_proc         # 계산된 네트워크 지연
            
            # 터미널에 로그 찍기
            print(f"⚡ YOLO: 총 {total_rtt:.0f}ms (서버 {server_proc:.0f}ms + 🌐네트워크 {network_delay:.0f}ms)")

    except Exception as e:
        pass # 에러 나도 조용히 넘어감
    finally:
        is_yolo_running = False # 작업 끝!

def request_vlm(frame):
    global is_vlm_running, latest_detections
    is_vlm_running = True
    latest_detections = [] 
    
    print(f"\n🚀 [{USER_NAME}] VLM 분석 요청 중... (YOLO 일시정지)")
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(
            VLM_URL,
            files={"file": ("capture.jpg", img_encoded.tobytes(), "image/jpeg")},
            params={"user_id": USER_NAME}, # ★ 내 이름표 붙이기
            headers=HEADERS, verify=False
        )
        print(f"✅ 결과: {response.json().get('result', '실패')}")
    except Exception as e:
        print(f"❌ 오류: {e}")
    finally:
        print("🔄 분석 완료. YOLO 재시작.")
        is_vlm_running = False

def main():
    global is_yolo_running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠 실패")
        return
    cap.set(3, 640); cap.set(4, 480)
    
    frame_count = 0
    print(f"📡 서버 연결: {SERVER_URL} (User: {USER_NAME})")
    print("💡 [SPACE]: VLM 분석 | [Q]: 종료")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        # 1. YOLO 요청 (3프레임마다 + ★이전 요청이 끝났을 때만★)
        if frame_count % 3 == 0 and not is_vlm_running and not is_yolo_running:
            is_yolo_running = True
            threading.Thread(target=request_yolo, args=(frame.copy(),), daemon=True).start()

        # 2. 결과 그리기
        for det in latest_detections:
            label = det['label']
            cx, cy, w, h = map(int, det['bbox'])
            x1, y1 = int(cx - w/2), int(cy - h/2)
            x2, y2 = int(cx + w/2), int(cy + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if is_vlm_running:
            cv2.putText(frame, "Analyzing...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Client", frame)
        key = cv2.waitKey(1)
        if key == ord('q'): break
        elif key == 32 and not is_vlm_running:
            threading.Thread(target=request_vlm, args=(frame.copy(),), daemon=True).start()
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()