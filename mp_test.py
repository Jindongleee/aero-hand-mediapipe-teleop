"""Step 1: MediaPipe Hands 카메라 테스트 (Tasks API).

목적: 맥북 카메라로 손 키포인트(21개) 잘 잡히는지 확인.
로봇 미연결 — 카메라만 사용.

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

# 21 키포인트 연결 정보 (MediaPipe Hands 표준)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                # palm base
]


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for p in points:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


# Tasks API 옵션
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없음. 권한 확인.")

start_t = time.time()
fps_t = time.time()
fps_count = 0
fps = 0.0

with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # 거울 모드
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - start_t) * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        if result.hand_landmarks:
            for hand_lms in result.hand_landmarks:
                draw_hand(frame, hand_lms)

        # FPS
        fps_count += 1
        if time.time() - fps_t >= 1.0:
            fps = fps_count / (time.time() - fps_t)
            fps_count = 0
            fps_t = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("MediaPipe Hands - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
