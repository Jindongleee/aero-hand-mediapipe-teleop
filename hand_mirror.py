"""Step 3: 카메라 손 → Aero Hand 실시간 미러링.

매핑:
  INDEX bend  → 채널 3 (검지)
  MIDDLE bend → 채널 4 (중지)
  RING bend   → 채널 5 (약지)
  PINKY bend  → 채널 6 (소지)
  THUMB bend  → 채널 1, 2 (엄지 굽힘 + 텐던)
  채널 0 (엄지 벌림) → 고정 (PAPER 기본값)

안전 장치:
  - EMA 필터  : 떨림 제거
  - Slew limit : 프레임 간 변화율 제한 (텐던 보호)
  - Clamp 0..1 : 범위 초과 방지
  - SPACE 키   : 토크 해제 (긴급 정지처럼 사용)

종료: 'q' 키
"""

import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from aero_open_sdk.aero_hand import AeroHand, CTRL_TOR
from poses import send_normalized, PAPER

# ---------- 설정 ----------
PORT = "/dev/cu.usbmodem101"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

EMA_ALPHA = 0.7         # 0~1, 낮을수록 부드럽고 느림
SLEW_LIMIT = 0.15       # 프레임 간 최대 변화 (~30fps에서 0.15 → 0.2초에 풀스윙)
SEND_HZ = 30            # 전송 주파수 상한

# 엄지 매핑 — 사람 손과 구조 차이가 있어서 별도 스케일/캘리브
THUMB_FLEX_SCALE = 0.85
THUMB_TENDON_SCALE = 0.85
# 엄지 벌림: thumb_TIP ↔ pinky_MCP 거리를 손 크기로 정규화
# 값 범위 (사람마다 약간 다름) — 좁히고 싶으면 ABD_MIN을 올림
THUMB_ABD_MIN = 0.55  # 엄지가 손바닥에 붙은 상태(주먹) 근처 거리/손크기
THUMB_ABD_MAX = 1.40  # 엄지가 옆으로 활짝 벌어진 상태

# ---------- MediaPipe ----------
FINGERS = [
    ("INDEX", 5, 6, 8),
    ("MIDDLE", 9, 10, 12),
    ("RING", 13, 14, 16),
    ("PINKY", 17, 18, 20),
]
THUMB_IDX = (2, 3, 4)  # MCP, IP, TIP

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def to_xyz(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def bend_ratio(pts, mcp, pip, tip) -> float:
    v1 = pts[pip] - pts[mcp]
    v2 = pts[tip] - pts[pip]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    c = float(np.dot(v1, v2) / (n1 * n2))
    return max(0.0, min(1.0, (1.0 - max(-1.0, min(1.0, c))) / 2.0))


def draw_hand(frame, lms):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def clamp01(x):
    return max(0.0, min(1.0, x))


# ---------- 상태 변수 ----------
ema = [PAPER[i] for i in range(7)]   # EMA 결과 (현재 부드럽게 추정된 값)
sent = list(ema)                     # 직전 송신값 (slew limit 기준)
torque_on = True

# ---------- 로봇 연결 ----------
hand = AeroHand(port=PORT)
print(f"Aero Hand connected: {PORT}")
send_normalized(hand, PAPER)         # 시작 자세
time.sleep(0.5)


def release_torque():
    """모든 서보 토크 0 — 손 자유 상태."""
    import struct
    msg = struct.pack("<2B7H", CTRL_TOR & 0xFF, 0x00, *[0] * 7)
    hand.ser.write(msg)
    hand.ser.flush()


# ---------- MediaPipe ----------
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    hand.close()
    raise RuntimeError("카메라를 열 수 없음.")

start_t = time.time()
last_send = 0.0
fps_t = time.time()
fps_n = 0
fps = 0.0

try:
    with mp_vision.HandLandmarker.create_from_options(options) as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - start_t) * 1000)
            result = lm.detect_for_video(mp_img, ts_ms)

            target = list(ema)  # 손 못 찾으면 이전 값 유지
            if result.hand_landmarks:
                hand_lms = result.hand_landmarks[0]
                draw_hand(frame, hand_lms)
                pts = to_xyz(hand_lms)

                # 4 손가락
                target[3] = bend_ratio(pts, *FINGERS[0][1:])
                target[4] = bend_ratio(pts, *FINGERS[1][1:])
                target[5] = bend_ratio(pts, *FINGERS[2][1:])
                target[6] = bend_ratio(pts, *FINGERS[3][1:])

                # 엄지 굽힘 (channel 1, 2)
                t_bend = bend_ratio(pts, *THUMB_IDX)
                target[1] = clamp01(t_bend * THUMB_FLEX_SCALE)
                target[2] = clamp01(t_bend * THUMB_TENDON_SCALE)

                # 엄지 벌림 (channel 0): thumb_TIP ↔ pinky_MCP 거리
                hand_size = float(np.linalg.norm(pts[9] - pts[0]) + 1e-6)
                spread = float(np.linalg.norm(pts[4] - pts[17])) / hand_size
                target[0] = clamp01((spread - THUMB_ABD_MIN) /
                                    (THUMB_ABD_MAX - THUMB_ABD_MIN))

            # EMA 필터
            for i in range(7):
                ema[i] = EMA_ALPHA * target[i] + (1 - EMA_ALPHA) * ema[i]

            # 전송 (Slew limit + Hz 제한)
            now = time.time()
            if torque_on and (now - last_send) >= 1.0 / SEND_HZ:
                limited = []
                for i in range(7):
                    delta = ema[i] - sent[i]
                    delta = max(-SLEW_LIMIT, min(SLEW_LIMIT, delta))
                    limited.append(clamp01(sent[i] + delta))
                send_normalized(hand, limited)
                sent = limited
                last_send = now

            # FPS
            fps_n += 1
            if now - fps_t >= 1.0:
                fps = fps_n / (now - fps_t)
                fps_n = 0
                fps_t = now

            # HUD
            cv2.putText(frame, f"FPS:{fps:.1f}  torque:{'ON' if torque_on else 'OFF'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y = 60
            ch_names = ["TH-ABD", "TH-FLX", "TH-TEN", "INDEX", "MIDDLE", "RING", "PINKY"]
            for i in range(7):
                v = sent[i]
                cv2.putText(frame, f"{ch_names[i]:7s} {v:.2f}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.rectangle(frame, (180, y - 14), (180 + 200, y - 2), (60, 60, 60), 1)
                cv2.rectangle(frame, (180, y - 14), (180 + int(v * 200), y - 2),
                              (0, 200, 255), -1)
                y += 22
            cv2.putText(frame, "[q]uit  [SPACE]toggle torque",
                        (10, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow("Aero Hand mirror - q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                torque_on = not torque_on
                if not torque_on:
                    release_torque()
                else:
                    sent = list(ema)
finally:
    cap.release()
    cv2.destroyAllWindows()
    release_torque()
    hand.close()
    print("done")
