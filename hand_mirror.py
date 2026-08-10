"""Step 3: 카메라 손 → Aero Hand 실시간 미러링.

매핑은 `mapping.py` 로 분리했다. 월드 랜드마크를 정규 자세로 정렬한 뒤
7채널 원시 신호를 뽑고, 캘리브레이션으로 0~1 로 편다.
엄지 3채널을 서로 독립인 신호로 나눈 이유와 기존 매핑의 결함은
`mapping.py` 의 모듈 docstring 참고.

채널:
  0 엄지 벌림   1 엄지 굽힘   2 엄지 텐던(MCP+IP)
  3 검지   4 중지   5 약지   6 소지

캘리브레이션:
  `calibration.json` 이 없으면 시작할 때 안내에 따라 손을 움직여 범위를 잡는다.
  다시 잡으려면 `--calibrate`.

안전 장치:
  - EMA 필터   : 떨림 제거
  - Slew limit : 프레임 간 변화율 제한 (텐던 보호). 첫 구동은 --slew 0.05
  - Clamp 0..1 : 범위 초과 방지
  - SPACE 키   : 토크 해제 (긴급 정지처럼 사용)

종료: 'q'

실행:
  .venv/bin/python hand_mirror.py --slew 0.05      # 첫 구동
  .venv/bin/python hand_mirror.py                  # 확인 후
  .venv/bin/python hand_mirror.py --no-robot       # 로봇 없이 값만 확인
"""

import argparse
import os
import struct
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from aero_open_sdk.aero_hand import CTRL_TOR
from poses import send_normalized, PAPER
from config import (
    open_hand, CH_NAMES, N_CHANNELS, SLEW_LIMIT_SAFE, SLEW_LIMIT_NORMAL,
)
from mapping import compute_raw, Calibration, CALIB_PATH

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

EMA_ALPHA = 0.7   # 낮을수록 부드럽고 느림
SEND_HZ = 30
CALIB_SECONDS = 8.0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

CALIB_STEPS = [
    "1. Open hand wide, spread the thumb away",
    "2. Make a fist, thumb across the palm",
    "3. Curl ONLY the thumb tip, keep fingers open",
    "4. Move each finger through its full range",
]


def draw_hand(frame, lms):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def draw_channels(frame, values, y0=60):
    """7채널 막대 표시."""
    y = y0
    for i in range(N_CHANNELS):
        v = float(values[i])
        cv2.putText(frame, f"{CH_NAMES[i]:7s} {v:.2f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.rectangle(frame, (180, y - 14), (380, y - 2), (60, 60, 60), 1)
        cv2.rectangle(frame, (180, y - 14), (180 + int(v * 200), y - 2),
                      (0, 200, 255), -1)
        y += 22


def run_calibration(cap, landmarker, start_t) -> Calibration:
    """안내에 따라 손을 움직이게 하고 채널별 최소·최대를 관측한다.

    하드코딩된 범위 대신 이걸 쓴다. 사람마다, 카메라 거리마다 원시 신호의
    범위가 다르기 때문이다. 특히 엄지 텐던 채널은 원시 변동폭이 좁아서
    (합성 데이터 기준 0.01~0.33) 펴주지 않으면 로봇이 거의 안 움직인다.
    """
    calib = Calibration()
    t0 = time.time()
    print("\n캘리브레이션 시작 — 화면 안내대로 손을 움직이세요.")
    for s in CALIB_STEPS:
        print("  " + s)

    while True:
        elapsed = time.time() - t0
        if elapsed >= CALIB_SECONDS:
            break

        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(
            mp_img, int((time.time() - start_t) * 1000))

        if result.hand_landmarks:
            draw_hand(frame, result.hand_landmarks[0])
            raw = compute_raw(result.hand_world_landmarks[0])
            calib.observe(raw)
            draw_channels(frame, calib.apply(raw) if calib.is_ready()
                          else np.zeros(N_CHANNELS))
        else:
            cv2.putText(frame, "NO HAND", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        remain = CALIB_SECONDS - elapsed
        cv2.putText(frame, f"CALIBRATING  {remain:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        step = CALIB_STEPS[min(int(elapsed / CALIB_SECONDS * len(CALIB_STEPS)),
                               len(CALIB_STEPS) - 1)]
        cv2.putText(frame, step, (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "move through the FULL range of each motion",
                    (10, frame.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Aero Hand mirror - q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if not calib.is_ready():
        raise RuntimeError(
            "캘리브레이션 실패 — 손이 한 번도 검출되지 않았습니다. "
            "조명과 카메라 각도를 확인하세요."
        )

    calib.save()
    print(f"캘리브레이션 저장: {CALIB_PATH}")
    for i in range(N_CHANNELS):
        print(f"  {CH_NAMES[i]:7s} raw {calib.lo[i]:8.4f} ~ {calib.hi[i]:8.4f}"
              f"  (폭 {calib.hi[i] - calib.lo[i]:.4f})")
    return calib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-robot", action="store_true",
                        help="로봇 없이 채널값만 확인 (웹캠만 필요)")
    parser.add_argument("--calibrate", action="store_true",
                        help="저장된 캘리브레이션을 무시하고 다시 잡는다")
    parser.add_argument("--slew", type=float, default=SLEW_LIMIT_SAFE,
                        help=f"프레임 간 변화율 제한 (기본 {SLEW_LIMIT_SAFE}, "
                             f"동작 확인 후 {SLEW_LIMIT_NORMAL} 권장)")
    args = parser.parse_args()

    hand = None
    if not args.no_robot:
        hand = open_hand()
        print(f"Aero Hand connected: {hand.ser.port}")
        send_normalized(hand, PAPER)
        time.sleep(0.5)

    def release_torque():
        """모든 서보 토크 0 — 손 자유 상태."""
        if hand is None:
            return
        msg = struct.pack("<2B7H", CTRL_TOR & 0xFF, 0x00, *[0] * N_CHANNELS)
        hand.ser.write(msg)
        hand.ser.flush()

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        if hand is not None:
            hand.close()
        raise RuntimeError("카메라를 열 수 없음. USB 웹캠 연결 확인.")

    ema = np.array(PAPER, dtype=np.float32)
    sent = ema.copy()
    torque_on = True
    start_t = time.time()
    last_send = 0.0
    fps_t, fps_n, fps = time.time(), 0, 0.0

    try:
        with mp_vision.HandLandmarker.create_from_options(options) as lm:
            calib = None if args.calibrate else Calibration.load()
            if calib is None:
                calib = run_calibration(cap, lm, start_t)
            else:
                print(f"캘리브레이션 로드: {CALIB_PATH}  "
                      f"(다시 잡으려면 --calibrate)")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int((time.time() - start_t) * 1000)
                result = lm.detect_for_video(mp_img, ts_ms)

                target = ema.copy()  # 손 못 찾으면 이전 값 유지
                if result.hand_landmarks:
                    draw_hand(frame, result.hand_landmarks[0])
                    target = calib.apply(compute_raw(result.hand_world_landmarks[0]))

                ema = EMA_ALPHA * target + (1 - EMA_ALPHA) * ema

                now = time.time()
                if (hand is not None and torque_on
                        and (now - last_send) >= 1.0 / SEND_HZ):
                    delta = np.clip(ema - sent, -args.slew, args.slew)
                    sent = np.clip(sent + delta, 0.0, 1.0)
                    send_normalized(hand, sent.tolist())
                    last_send = now
                elif hand is None:
                    sent = ema

                fps_n += 1
                if now - fps_t >= 1.0:
                    fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now

                mode = "NO-ROBOT" if hand is None else (
                    "torque:ON" if torque_on else "torque:OFF")
                cv2.putText(frame, f"FPS:{fps:.1f}  {mode}  slew={args.slew}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)
                draw_channels(frame, sent)
                cv2.putText(frame, "[q]uit  [SPACE]toggle torque",
                            (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                cv2.imshow("Aero Hand mirror - q to quit", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" ") and hand is not None:
                    torque_on = not torque_on
                    if not torque_on:
                        release_torque()
                    else:
                        sent = ema.copy()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        release_torque()
        if hand is not None:
            hand.close()
        print("done")


if __name__ == "__main__":
    main()
