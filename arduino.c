#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESP32Servo.h>
#include <AccelStepper.h>
#include <ArduinoOTA.h>
#include <math.h>

// --- [1. 네트워크 및 통신 설정] ---
const char* ssid = "Lnet";
const char* password = "123456788";
unsigned int localPort = 12345;

WiFiUDP udp;
char packetBuffer[255];

// --- [2. 서보모터(팬-틸트) 설정] ---
Servo servoPan;
Servo servoTilt;

const int panPin = 18;
const int tiltPin = 19;

bool isServoAttached = false;

// 실제 서보 현재 위치
float currentPan = 90.0;
float currentTilt = 90.0;

// ESP32 내부에서 부드럽게 만든 목표 위치
float targetPan = 90.0;
float targetTilt = 90.0;

// Python에서 받은 원본 목표 위치
float rawTargetPan = 90.0;
float rawTargetTilt = 90.0;

// --- [서보 부드러운 이동 튜닝값] ---
const unsigned long SERVO_UPDATE_MS = 20;   // 20ms = 50Hz

const float PAN_MAX_DEG_PER_SEC = 8.0;
const float TILT_MAX_DEG_PER_SEC = 4.0;

const float PAN_TARGET_FILTER = 0.015;
const float TILT_TARGET_FILTER = 0.005;

const float SERVO_DEADBAND = 0.35;

unsigned long lastServoUpdate = 0;

// --- [3. 스테퍼(리니어 슬라이드) 설정] ---
const int stepPin = 12;
const int dirPin = 14;

AccelStepper stepper(AccelStepper::DRIVER, stepPin, dirPin);

float max_speed = 6000.0;
float acceleration = 700.0;

// --- [4. 유틸 함수] ---
float moveToward(float current, float target, float maxStep) {
  float diff = target - current;

  if (fabsf(diff) < SERVO_DEADBAND) {
    return target;
  }

  if (diff > maxStep) {
    return current + maxStep;
  }

  if (diff < -maxStep) {
    return current - maxStep;
  }

  return target;
}

float lowPass(float current, float target, float alpha) {
  return current * (1.0 - alpha) + target * alpha;
}

int angleToMicroseconds(float angle) {
  angle = constrain(angle, 0.0, 180.0);
  return 500 + (angle / 180.0) * 1900;
}

void attachServosIfNeeded() {
  if (!isServoAttached) {
    servoPan.attach(panPin, 500, 2400);
    servoTilt.attach(tiltPin, 500, 2400);

    lastServoUpdate = millis();

    servoPan.writeMicroseconds(angleToMicroseconds(currentPan));
    servoTilt.writeMicroseconds(angleToMicroseconds(currentTilt));

    isServoAttached = true;
  }
}

void setup() {
  Serial.begin(115200);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);

  servoPan.setPeriodHertz(50);
  servoTilt.setPeriodHertz(50);

  stepper.setMaxSpeed(max_speed);
  stepper.setAcceleration(acceleration);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  Serial.print("WiFi 연결 중");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("WiFi 연결 완료. IP 주소: ");
  Serial.println(WiFi.localIP());

  // --- OTA 설정 ---
  ArduinoOTA.setHostname("ESP32-3Axis-Controller");
  ArduinoOTA.setPassword("1234");

  ArduinoOTA.onStart([]() {
    Serial.println("\nOTA 업데이트 시작");

    stepper.stop();
    stepper.setSpeed(0);

    if (isServoAttached) {
      servoPan.detach();
      servoTilt.detach();
      isServoAttached = false;
    }
  });

  ArduinoOTA.onEnd([]() {
    Serial.println("\nOTA 업데이트 완료");
  });

  ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
    Serial.printf("OTA 진행률: %u%%\r", (progress * 100) / total);
  });

  ArduinoOTA.onError([](ota_error_t error) {
    Serial.printf("\nOTA 오류[%u]: ", error);

    if (error == OTA_AUTH_ERROR) {
      Serial.println("인증 실패");
    } else if (error == OTA_BEGIN_ERROR) {
      Serial.println("시작 실패");
    } else if (error == OTA_CONNECT_ERROR) {
      Serial.println("연결 실패");
    } else if (error == OTA_RECEIVE_ERROR) {
      Serial.println("수신 실패");
    } else if (error == OTA_END_ERROR) {
      Serial.println("종료 실패");
    }
  });

  ArduinoOTA.begin();

  udp.begin(localPort);

  Serial.println("✅ OTA 준비 완료");
  Serial.println("✅ UDP 준비 완료");
  Serial.println("✅ 시스템 준비 완료");
}

void loop() {
  ArduinoOTA.handle();

  int packetSize = udp.parsePacket();

  if (packetSize) {
    int len = udp.read(packetBuffer, 255);

    if (len > 0) {
      packetBuffer[len] = 0;

      String input = String(packetBuffer);
      char firstChar = packetBuffer[0];

      // 1. 팬-틸트 목표값 업데이트
      if (firstChar == 'P') {
        int tIndex = input.indexOf('T');

        if (tIndex != -1) {
          float newPan = input.substring(1, tIndex).toFloat();
          float newTilt = input.substring(tIndex + 1).toFloat();

          // Python에서 온 원본 목표값 저장
          // 바로 targetPan/targetTilt에 넣지 않고 rawTarget에만 저장
          rawTargetPan = constrain(newPan, 0.0, 180.0);
          rawTargetTilt = constrain(newTilt, 0.0, 180.0);

          attachServosIfNeeded();
        }
      }

      // 2. 스테퍼 명령
      else if (firstChar == 'U') {
        stepper.move(-2000000);
      }

      else if (firstChar == 'D') {
        stepper.move(2000000);
      }

      else if (firstChar == 'S') {
        stepper.stop();
        stepper.setCurrentPosition(stepper.currentPosition());
        stepper.setSpeed(0);
      }
    }
  }

  // --- [실시간 부드러운 팬-틸트 제어] ---
  if (isServoAttached) {
    unsigned long now = millis();

    if (now - lastServoUpdate >= SERVO_UPDATE_MS) {
      float dt = (now - lastServoUpdate) / 1000.0;
      lastServoUpdate = now;

      if (dt <= 0) {
        dt = SERVO_UPDATE_MS / 1000.0;
      }

      // 1. Python에서 받은 raw target을 바로 따라가지 않고 내부 target을 천천히 갱신
      targetPan = lowPass(targetPan, rawTargetPan, PAN_TARGET_FILTER);
      targetTilt = lowPass(targetTilt, rawTargetTilt, TILT_TARGET_FILTER);

      // 2. 실제 서보 current 위치는 초당 최대 이동량 제한
      float maxPanStep = PAN_MAX_DEG_PER_SEC * dt;
      float maxTiltStep = TILT_MAX_DEG_PER_SEC * dt;

      currentPan = moveToward(currentPan, targetPan, maxPanStep);
      currentTilt = moveToward(currentTilt, targetTilt, maxTiltStep);

      // 3. 서보 출력
      servoPan.writeMicroseconds(angleToMicroseconds(currentPan));
      servoTilt.writeMicroseconds(angleToMicroseconds(currentTilt));
    }
  }

  stepper.run();
}