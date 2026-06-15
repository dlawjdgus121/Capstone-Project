//리미트 스위치 내용 추가된 버전
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESP32Servo.h>
#include <AccelStepper.h>
#include <ArduinoOTA.h>
#include <math.h>

// --- [1. 네트워크 및 통신 설정] ---
const char* ssid = "Lnet";
const char* password = "123456788";
unsigned int localPort = 12346;

WiFiUDP udp;
char packetBuffer[255];

// --- [2. 서보모터(팬-틸트) 설정] ---
Servo servoPan;
Servo servoTilt;

const int panPin = 18;
const int tiltPin = 19;
//tilt=상하
const int upperLimitPin = 13; // 상단 스위치
const int lowerLimitPin = 15; // 하단 스위치

IPAddress lastRemoteIP;       // UI에 알림을 보내기 위한 목적지 저장용
const int UPPER_LIMIT_PRESSED_LEVEL = LOW;  // NO + INPUT_PULLUP: unpressed=HIGH, pressed=LOW
const int LOWER_LIMIT_PRESSED_LEVEL = LOW;  // NO + INPUT_PULLUP: unpressed=HIGH, pressed=LOW

uint16_t lastRemotePort = 0;  
bool isShuttingDown = false;
bool isServoAttached = false;

// 실제 서보 현재 위치
float currentPan = 90.0;
float currentTilt = 90.0;

const float PAN_MIN_LIMIT = 40.0;
const float PAN_MAX_LIMIT = 140.0;
const float TILT_MIN_LIMIT = 50.0;
const float TILT_MAX_LIMIT = 120.0;

// ESP32 내부에서 부드럽게 만든 목표 위치
float targetPan = 90.0;
float targetTilt = 90.0;

// Python에서 받은 원본 목표 위치
float rawTargetPan = 90.0;
float rawTargetTilt = 90.0;

// --- [🔥 부드러운 이동을 위한 튜닝값 수정 완료] ---
const unsigned long SERVO_UPDATE_MS = 20;   // 50Hz PWM 주기에 맞춰 10ms -> 20ms 동기화

// 파이썬 속도 명령(Pan: 58.0 / Tilt: 34.0)을 온전히 수용하도록 한계 상향
const float PAN_MAX_DEG_PER_SEC = 28.0;
const float TILT_MAX_DEG_PER_SEC = 20.0;

// 파이썬 제어 궤적과 충돌하지 않도록 하위 가속도 제한 대폭 상향 (추종성 확보)
const float PAN_ACCEL_DEG_PER_SEC2 = 90.0;
const float TILT_ACCEL_DEG_PER_SEC2 = 70.0;

const float PAN_TARGET_TAU = 0.22;
const float TILT_TARGET_TAU = 0.28;
const float PAN_TRACK_GAIN = 3.0;
const float TILT_TRACK_GAIN = 2.6;

// 미세 제어 시 굳는 현상을 막기 위해 데드밴드 최소화 (노이즈 컷은 파이썬이 전담)
const float SERVO_DEADBAND = 0.08;
const float VELOCITY_COMMAND_DEADBAND = 0.01;
const unsigned long SERVO_COMMAND_TIMEOUT_MS = 250;

float panVelocity = 0.0;
float tiltVelocity = 0.0;
float commandPanNorm = 0.0;
float commandTiltNorm = 0.0;
bool velocityCommandMode = false;

unsigned long lastServoUpdate = 0;
unsigned long lastPanTiltPacketTime = 0;

// --- [3. 스테퍼(리니어 슬라이드) 설정] ---
const int stepPin = 12;
const int dirPin = 14;

AccelStepper stepper(AccelStepper::DRIVER, stepPin, dirPin);

const unsigned long LIMIT_DEBOUNCE_MS = 50;
bool upperLimitPressed = false;
bool lowerLimitPressed = false;
bool upperLimitRawLast = false;
bool lowerLimitRawLast = false;
unsigned long upperLimitRawChangedAt = 0;
unsigned long lowerLimitRawChangedAt = 0;

// 🔧 현재 0.5A + 1/16 스텝 상태에서 더 빠르게 동작하도록 조정
float max_speed = 10000.0;       // 더 빠른 최대 속도
float acceleration = 5000.0;     // 처음부터 빠르게 올라가고 내려가도록 큰 가속도로 설정

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

float alphaFromTau(float tau, float dt) {
  if (tau <= 0.001) {
    return 1.0;
  }

  return 1.0 - expf(-dt / tau);
}

float approachFloat(float current, float target, float maxDelta) {
  float diff = target - current;

  if (diff > maxDelta) {
    return current + maxDelta;
  }

  if (diff < -maxDelta) {
    return current - maxDelta;
  }

  return target;
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

// 🔍 파이썬으로 로그 메시지를 보내는 전용 함수
void sendLogToPython(const char* msg) {
  if (lastRemoteIP[0] != 0) { // 파이썬 IP가 확보되었을 때만
    udp.beginPacket(lastRemoteIP, 12346); // 파이썬의 수신 전용 포트로 
    udp.print("LOG:");
    udp.print(msg);
    udp.endPacket();
  }
}

bool readUpperLimitRaw() {
  return digitalRead(upperLimitPin) == UPPER_LIMIT_PRESSED_LEVEL;
}

bool readLowerLimitRaw() {
  return digitalRead(lowerLimitPin) == LOWER_LIMIT_PRESSED_LEVEL;
}

void initLimitStates() {
  upperLimitRawLast = readUpperLimitRaw();
  lowerLimitRawLast = readLowerLimitRaw();
  upperLimitPressed = upperLimitRawLast;
  lowerLimitPressed = lowerLimitRawLast;
  upperLimitRawChangedAt = millis();
  lowerLimitRawChangedAt = millis();
}

void updateLimitStates() {
  unsigned long now = millis();
  bool upperRaw = readUpperLimitRaw();
  bool lowerRaw = readLowerLimitRaw();

  if (upperRaw != upperLimitRawLast) {
    upperLimitRawLast = upperRaw;
    upperLimitRawChangedAt = now;
  } else if (now - upperLimitRawChangedAt >= LIMIT_DEBOUNCE_MS) {
    upperLimitPressed = upperRaw;
  }

  if (lowerRaw != lowerLimitRawLast) {
    lowerLimitRawLast = lowerRaw;
    lowerLimitRawChangedAt = now;
  } else if (now - lowerLimitRawChangedAt >= LIMIT_DEBOUNCE_MS) {
    lowerLimitPressed = lowerRaw;
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
  stepper.setMinPulseWidth(5);

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

  pinMode(upperLimitPin, INPUT_PULLUP);
  pinMode(lowerLimitPin, INPUT_PULLUP);
  initLimitStates();

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
  updateLimitStates();

  int packetSize = udp.parsePacket();

  if (packetSize) {
    int len = udp.read(packetBuffer, 255);

    if (len > 0) {
      packetBuffer[len] = 0;

      String input = String(packetBuffer);
      char firstChar = packetBuffer[0];

      // 🔍 파이썬/UI 측 목적지 정보 저장 (역송신용)
      lastRemoteIP = udp.remoteIP();
      lastRemotePort = udp.remotePort();

      // 1. 팬-틸트 목표값 업데이트
      if (firstChar == 'P') {
        int tIndex = input.indexOf('T');

        if (tIndex != -1) {
          float newPan = input.substring(1, tIndex).toFloat();
          float newTilt = input.substring(tIndex + 1).toFloat();

          rawTargetPan = constrain(newPan, PAN_MIN_LIMIT, PAN_MAX_LIMIT);
          rawTargetTilt = constrain(newTilt, TILT_MIN_LIMIT, TILT_MAX_LIMIT);
          lastPanTiltPacketTime = millis();
          velocityCommandMode = false;

          attachServosIfNeeded();
        }
      }

      // 2. 속도기반 제어 명령
      else if (firstChar == 'V') {
        int yIndex = input.indexOf('Y');

        if (yIndex != -1) {
          float newPanNorm = input.substring(1, yIndex).toFloat();
          float newTiltNorm = input.substring(yIndex + 1).toFloat();

          commandPanNorm = constrain(newPanNorm, -1.0, 1.0);
          commandTiltNorm = constrain(newTiltNorm, -1.0, 1.0);
          
          if (fabsf(commandPanNorm) < VELOCITY_COMMAND_DEADBAND) {
            commandPanNorm = 0.0;
          }
          if (fabsf(commandTiltNorm) < VELOCITY_COMMAND_DEADBAND) {
            commandTiltNorm = 0.0;
          }
          lastPanTiltPacketTime = millis();
          velocityCommandMode = true;

          attachServosIfNeeded();
        }
      }

      // 3. 현재 각도 기준 1회 보정 명령
      else if (firstChar == 'C') {
        int yIndex = input.indexOf('Y');

        if (yIndex != -1) {
          float panDelta = input.substring(1, yIndex).toFloat();
          float tiltDelta = input.substring(yIndex + 1).toFloat();

          rawTargetPan = constrain(currentPan + panDelta, PAN_MIN_LIMIT, PAN_MAX_LIMIT);
          rawTargetTilt = constrain(currentTilt + tiltDelta, TILT_MIN_LIMIT, TILT_MAX_LIMIT);
          targetPan = currentPan;
          targetTilt = currentTilt;
          panVelocity = 0.0;
          tiltVelocity = 0.0;
          lastPanTiltPacketTime = millis();
          velocityCommandMode = false;

          attachServosIfNeeded();
        }
      }

      // 4. 상태 요청
      else if (firstChar == 'L') {
        char reply[64];
        snprintf(
          reply,
          sizeof(reply),
          "A%.1fT%.1fS%d",
          currentPan,
          currentTilt,
          isServoAttached ? 1 : 0
        );

        udp.beginPacket(udp.remoteIP(), udp.remotePort());
        udp.print(reply);
        udp.endPacket();

        Serial.printf(
          "[SERVO] current pan=%.1fdeg, tilt=%.1fdeg, attached=%d\n",
          currentPan,
          currentTilt,
          isServoAttached ? 1 : 0
        );
      }

      // 5. 상승 명령 (상단 스위치가 눌리지 않았을 때만 작동)
      else if (firstChar == 'U') {
        if (!upperLimitPressed) {
          stepper.move(2000000);
          sendLogToPython("⬆️ 리니어 슬라이드 상승 시작"); // 🔍 파이썬으로 전송 
        } else {
          sendLogToPython("⚠️ 상단 리미트 감지로 상승 차단");
        } // <--- 이 괄호가 주석에 가려져 있던 것을 수정했습니다!
      }

      // 6. 하강 명령 (하단 스위치가 눌리지 않았을 때만 작동)
      else if (firstChar == 'D') {
        if (!lowerLimitPressed) {
          stepper.move(-2000000);
          sendLogToPython("⬇️ 리니어 슬라이드 하강 시작"); // 🔍 파이썬으로 전송 
        } // <--- 이 괄호가 주석에 가려져 있던 것을 수정했습니다!
      }

      // 7. 정지 명령
      else if (firstChar == 'S') {
        stepper.stop();
        stepper.setCurrentPosition(stepper.currentPosition());
        stepper.setSpeed(0);
        sendLogToPython("🛑 리니어 슬라이드 강제 정지"); // 🔍 파이썬으로 전송
      }

      // 8. 시스템 종료 시퀀스 명령
      else if (firstChar == 'X') {
        isShuttingDown = true;
        if (!upperLimitPressed) {
          stepper.move(2000000); 
        } else {
          sendLogToPython("⚠️ 상단 리미트 감지로 종료 상승 차단");
        }
        sendLogToPython("🛑 종료 시퀀스 개시: 슬라이드 상승"); // 🔍 파이썬으로 전송
      }
    }
  }

  bool currentUpperLimit = upperLimitPressed;
  bool currentLowerLimit = lowerLimitPressed;
  static bool prevUpperLimit = false;
  static bool prevLowerLimit = false;

  // 1. 상단 스위치가 눌려있는 "모든 순간"에 작동 (철벽 방어)
  if (currentUpperLimit) {
    if (stepper.distanceToGo() > 0) { 
      stepper.setSpeed(0);            
      stepper.setCurrentPosition(stepper.currentPosition()); 
    }
    if (!prevUpperLimit && lastRemotePort != 0) {
      udp.beginPacket(lastRemoteIP, 12346);
      udp.print("LIMIT_TOP");
      udp.endPacket();
    }
  }

  // 2. 하단 스위치가 눌려있는 "모든 순간"에 작동 (철벽 방어)
  if (currentLowerLimit) {
    if (stepper.distanceToGo() < 0) { // 모터가 아래로 가려고 시도하면
      stepper.setSpeed(0);
      stepper.setCurrentPosition(stepper.currentPosition());
    }
    if (!prevLowerLimit && lastRemotePort != 0) {
      udp.beginPacket(lastRemoteIP, 12346);
      udp.print("LIMIT_BOTTOM");
      udp.endPacket();
    }
  }

  // 3. 시스템 종료 시퀀스 완료 처리
  if (isShuttingDown && currentUpperLimit) {
    stepper.setSpeed(0);
    stepper.setCurrentPosition(0); 
    isShuttingDown = false;
    Serial.println("[SYSTEM] 상단 리미트 스위치 접촉 - 상승 정지 및 종료 완료");
  }

  // 이전 상태 업데이트
  prevUpperLimit = currentUpperLimit;
  prevLowerLimit = currentLowerLimit;


  // --- [실시간 부드러운 팬-틸트 제어] ---
  if (isServoAttached) {
    unsigned long now = millis();

    if (now - lastServoUpdate >= SERVO_UPDATE_MS) {
      float dt = (now - lastServoUpdate) / 1000.0;
      lastServoUpdate = now;

      if (dt <= 0) {
        dt = SERVO_UPDATE_MS / 1000.0;
      }

      float panAlpha = alphaFromTau(PAN_TARGET_TAU, dt);
      float tiltAlpha = alphaFromTau(TILT_TARGET_TAU, dt);

      if (velocityCommandMode && lastPanTiltPacketTime > 0 && now - lastPanTiltPacketTime > SERVO_COMMAND_TIMEOUT_MS) {
        rawTargetPan = currentPan;
        rawTargetTilt = currentTilt;
        targetPan = currentPan;
        targetTilt = currentTilt;
        panVelocity = 0.0;
        tiltVelocity = 0.0;
        commandPanNorm = 0.0;
        commandTiltNorm = 0.0;
      }

      float panError = 0.0;
      float tiltError = 0.0;
      float desiredPanVelocity = 0.0;
      float desiredTiltVelocity = 0.0;

      if (velocityCommandMode) {
        desiredPanVelocity = commandPanNorm * PAN_MAX_DEG_PER_SEC;
        desiredTiltVelocity = commandTiltNorm * TILT_MAX_DEG_PER_SEC;

        rawTargetPan = currentPan;
        rawTargetTilt = currentTilt;
        targetPan = currentPan;
        targetTilt = currentTilt;
      } else {
        targetPan = lowPass(targetPan, rawTargetPan, panAlpha);
        targetTilt = lowPass(targetTilt, rawTargetTilt, tiltAlpha);

        panError = targetPan - currentPan;
        tiltError = targetTilt - currentTilt;

        desiredPanVelocity = constrain(
          panError * PAN_TRACK_GAIN,
          -PAN_MAX_DEG_PER_SEC,
          PAN_MAX_DEG_PER_SEC
        );
        desiredTiltVelocity = constrain(
          tiltError * TILT_TRACK_GAIN,
          -TILT_MAX_DEG_PER_SEC,
          TILT_MAX_DEG_PER_SEC
        );

        if (fabsf(panError) < SERVO_DEADBAND) {
          desiredPanVelocity = 0.0;
        }

        if (fabsf(tiltError) < SERVO_DEADBAND) {
          desiredTiltVelocity = 0.0;
        }
      }

      panVelocity = approachFloat(
        panVelocity,
        desiredPanVelocity,
        PAN_ACCEL_DEG_PER_SEC2 * dt
      );
      tiltVelocity = approachFloat(
        tiltVelocity,
        desiredTiltVelocity,
        TILT_ACCEL_DEG_PER_SEC2 * dt
      );

      currentPan += panVelocity * dt;
      currentTilt += tiltVelocity * dt;

      currentPan = constrain(currentPan, PAN_MIN_LIMIT, PAN_MAX_LIMIT);
      currentTilt = constrain(currentTilt, TILT_MIN_LIMIT, TILT_MAX_LIMIT);

      if (fabsf(panError) < SERVO_DEADBAND && fabsf(panVelocity) < 0.05) {
        currentPan = targetPan;
        panVelocity = 0.0;
      }

      if (fabsf(tiltError) < SERVO_DEADBAND && fabsf(tiltVelocity) < 0.05) {
        currentTilt = targetTilt;
        tiltVelocity = 0.0;
      }

      // 3. 서보 출력
      servoPan.writeMicroseconds(angleToMicroseconds(currentPan));
      servoTilt.writeMicroseconds(angleToMicroseconds(currentTilt));
    }
  }

  stepper.run();
}
