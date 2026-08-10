"""Step 9: 학습된 연속 회귀 정책으로 실물 미러링.

`hand_mirror.py` 와 하는 일은 같은데, 7채널을 만드는 주체가 다르다.

  hand_mirror.py     : 랜드마크 → mapping.py 의 해석적 수식 → 7채널
  teleop_bc_play.py  : 랜드마크 → 학습된 MLP → 7채널

같은 입력에 대해 규칙 기반과 학습 기반을 나란히 비교할 수 있다.
`--compare` 를 주면 두 값을 화면에 같이 띄우고 차이를 보여준다.
(정책이 expert 를 얼마나 따라가는지 실물에서 확인하는 용도)

안전 장치는 hand_mirror.py 와 동일하게 유지한다. 오히려 더 중요하다 —
학습된 출력은 수식보다 튈 수 있고, 그 값이 그대로 관절로 간다.
  - EMA 필터
  - Slew limit (첫 구동은 SLEW_LIMIT_SAFE)
  - poses.send_normalized 의 ch0 하한 클램프
  - SPACE 토크 해제

종료: 'q'

실행:
  .venv/bin/python teleop_bc_play.py --no-robot            # 로봇 없이 값만
  .venv/bin/python teleop_bc_play.py --slew 0.05           # 첫 실물 구동
  .venv/bin/python teleop_bc_play.py --compare             # 수식 vs 정책
"""

import argparse
import os
import struct
import time

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from landmarks import extract_features
from mapping import compute_raw, Calibration
from poses import send_normalized, PAPER
from train_teleop_bc import TeleopPolicy
from config import CH_NAMES, N_CHANNELS, SLEW_LIMIT_SAFE, SLEW_LIMIT_NORMAL

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
POLICY_PATH = os.path.join(os.path.dirname(__file__), "teleop_policy.pt")

EMA_ALPHA = 0.7
SEND_HZ = 30

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_hand(frame, lms):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def load_policy():
    if not os.path.exists(POLICY_PATH):
        raise SystemExit(
            f"정책 파일이 없습니다: {POLICY_PATH}\n"
            "먼저 train_teleop_bc.py 를 실행하세요."
        )
    ckpt = torch.load(POLICY_PATH, map_location="cpu", weights_only=False)
    model = TeleopPolicy(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"],
                         out_dim=ckpt["out_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"정책 로드: {POLICY_PATH}  (검증 MAE {ckpt.get('val_mae', float('nan')):.4f})")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-robot", action="store_true",
                        help="로봇 없이 채널값만 확인")
    parser.add_argument("--compare", action="store_true",
                        help="해석적 매핑과 나란히 표시하고 차이를 보여줌")
    parser.add_argument("--slew", type=float, default=SLEW_LIMIT_SAFE,
                        help=f"프레임 간 변화율 제한 (기본 {SLEW_LIMIT_SAFE}, "
                             f"확인 후 {SLEW_LIMIT_NORMAL} 권장)")
    args = parser.parse_args()

    model = load_policy()

    # --compare 나 로봇 구동에는 캘리브레이션이 필요 없다 (정책이 직접 냄).
    # 다만 비교용 expert 값을 계산하려면 있어야 한다.
    calib = Calibration.load() if args.compare else None
    if args.compare and calib is None:
        raise SystemExit("--compare 에는 calibration.json 이 필요합니다.")

    hand = None
    if not args.no_robot:
        from config import open_hand
        hand = open_hand()
        print(f"Aero Hand connected: {hand.ser.port}")
        send_normalized(hand, PAPER)
        time.sleep(0.5)

    def release_torque():
        if hand is None:
            return
        msg = struct.pack("<2B7H", 0x12, 0x00, *[0] * N_CHANNELS)
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
        raise RuntimeError("카메라를 열 수 없음.")

    ema = np.array(PAPER, dtype=np.float32)
    sent = ema.copy()
    torque_on = True
    start_t = time.time()
    last_send = 0.0

    try:
        with mp_vision.HandLandmarker.create_from_options(options) as lm:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = lm.detect_for_video(
                    mp_img, int((time.time() - start_t) * 1000))

                target = ema.copy()
                expert = None
                if result.hand_landmarks:
                    draw_hand(frame, result.hand_landmarks[0])
                    world = result.hand_world_landmarks[0]

                    feat = extract_features(world)
                    with torch.no_grad():
                        target = model(
                            torch.from_numpy(feat).unsqueeze(0)
                        )[0].numpy()

                    if calib is not None:
                        expert = calib.apply(compute_raw(world))

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

                # ---------- HUD ----------
                mode = "NO-ROBOT" if hand is None else (
                    "torque:ON" if torque_on else "torque:OFF")
                cv2.putText(frame, f"BC policy  {mode}  slew={args.slew}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)

                y = 60
                for i in range(N_CHANNELS):
                    v = float(sent[i])
                    if expert is not None:
                        e = float(expert[i])
                        d = abs(v - e)
                        col = (0, 0, 255) if d > 0.10 else (255, 255, 255)
                        cv2.putText(frame,
                                    f"{CH_NAMES[i]:7s} BC {v:.2f}  eq {e:.2f}  d{d:.2f}",
                                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                        cv2.rectangle(frame, (330, y - 12), (430, y - 2),
                                      (60, 60, 60), 1)
                        cv2.rectangle(frame, (330, y - 12),
                                      (330 + int(v * 100), y - 7), (0, 200, 255), -1)
                        cv2.rectangle(frame, (330, y - 7),
                                      (330 + int(e * 100), y - 2), (0, 255, 0), -1)
                    else:
                        cv2.putText(frame, f"{CH_NAMES[i]:7s} {v:.2f}", (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                        cv2.rectangle(frame, (180, y - 14), (380, y - 2),
                                      (60, 60, 60), 1)
                        cv2.rectangle(frame, (180, y - 14),
                                      (180 + int(v * 200), y - 2), (0, 200, 255), -1)
                    y += 22

                if expert is not None:
                    cv2.putText(frame, "orange=BC  green=equation",
                                (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (200, 200, 200), 1)

                cv2.putText(frame, "[q]uit  [SPACE]toggle torque",
                            (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                cv2.imshow("Teleop BC policy - q to quit", frame)
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
