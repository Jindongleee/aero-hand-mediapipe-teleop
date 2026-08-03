"""Step 2: 손가락 굽힘 비율 계산.

각 손가락의 굽힘 정도를 0~1 값으로 출력.
  0 = 완전히 펴짐 (직선)
  1 = 완전히 굽힘 (90도 이상)

방법: MCP→PIP 벡터와 PIP→TIP 벡터 사이 각도의 코사인.
  cos = 1  (벡터 같은 방향, 펴짐)  → bend = 0
  cos = -1 (벡터 반대 방향, 완전 굽힘) → bend = 1
  bend = (1 - cos) / 2

엄지는 구조가 달라 별도 처리 필요 (Step 4에서).

종료: 'q' 키
"""

import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# (이름, MCP idx, PIP idx, TIP idx)
FINGERS = [
    ("INDEX",  5, 6, 8),
    ("MIDDLE", 9, 10, 12),
    ("RING",   13, 14, 16),
    ("PINKY",  17, 18, 20),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def to_xyz(landmarks):
    """랜드마크 리스트 → (21, 3) numpy 배열 (정규화 좌표)."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def bend_ratio(pts: np.ndarray, mcp: int, pip: int, tip: int) -> float:
    """MCP→PIP, PIP→TIP 두 벡터 각도로 굽힘 비율 (0~1) 계산."""
    v1 = pts[pip] - pts[mcp]
    v2 = pts[tip] - pts[pip]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))
    return (1.0 - cos) / 2.0


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for p in points:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없음.")

start_t = time.time()

with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - start_t) * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        bends = {}
        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                draw_hand(frame, hand_lms)
                pts = to_xyz(hand_lms)
                for name, mcp, pip, tip in FINGERS:
                    bends[name] = bend_ratio(pts, mcp, pip, tip)
                break  # 1손만

        # 화면 좌상단에 굽힘 값 표시
        y = 60
        for name, _, _, _ in FINGERS:
            v = bends.get(name, 0.0)
            bar_w = int(v * 200)
            cv2.putText(frame, f"{name:6s} {v:.2f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(frame, (180, y - 18), (180 + 200, y - 4),
                          (60, 60, 60), 1)
            cv2.rectangle(frame, (180, y - 18), (180 + bar_w, y - 4),
                          (0, 200, 255), -1)
            y += 30

        cv2.imshow("Bend ratios - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
