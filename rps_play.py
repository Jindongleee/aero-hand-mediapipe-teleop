"""Step 6: 학습된 BC 정책으로 실물 가위바위보.

카메라로 사람 손을 보고 → MLP 가 가위/바위/보를 분류 → 해당 포즈를
Aero Hand 에 전송한다.

`--no-robot` 을 주면 로봇 없이 분류 결과만 화면에 띄운다.
웹캠만 있으면 정책 검증이 되므로, 손 세션 전에 이걸로 먼저 확인할 것.

안전 장치 (`hand_mirror.py` 와 동일 + 추가):
  - Slew limit  : 프레임 간 변화율 제한. 첫 실행은 SLEW_LIMIT_SAFE(0.05)
  - 확률 임계값  : 확신이 낮으면 아예 전송하지 않는다
  - 디바운스     : 연속 N 프레임 같은 예측일 때만 포즈를 바꾼다
  - SPACE       : 토크 해제 (긴급 정지)

디바운스와 임계값이 없으면 손이 계속 덜덜 떨린다. 사람 손이 포즈 사이를
지나가는 중간 프레임은 어느 클래스도 아닌데 분류기는 매 프레임 답을 내기
때문이다. BC 정책 출력은 휴리스틱 매핑보다 튈 수 있어서, 오히려 규칙 기반
때보다 안전장치가 더 중요하다.

종료: 'q'
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from landmarks import extract_features
from poses import POSES, PAPER, send_normalized
from train_rps import RPSPolicy
from config import N_CHANNELS, SLEW_LIMIT_SAFE, SLEW_LIMIT_NORMAL

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
POLICY_PATH = os.path.join(os.path.dirname(__file__), "rps_policy.pt")

CONF_THRESHOLD = 0.85   # 이 확률 미만이면 포즈를 바꾸지 않는다
DEBOUNCE_FRAMES = 5     # 연속 이 횟수만큼 같은 예측이어야 전환
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


def clamp01(x):
    return max(0.0, min(1.0, x))


def load_policy():
    if not os.path.exists(POLICY_PATH):
        raise SystemExit(
            f"정책 파일이 없습니다: {POLICY_PATH}\n"
            "먼저 train_rps.py 를 실행하세요."
        )
    ckpt = torch.load(POLICY_PATH, map_location="cpu", weights_only=False)
    model = RPSPolicy(in_dim=ckpt["in_dim"], hidden=ckpt["hidden"],
                      n_classes=len(ckpt["label_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"정책 로드: {POLICY_PATH}  (검증 정확도 {ckpt.get('val_acc', float('nan')):.3f})")
    return model, ckpt["label_names"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-robot", action="store_true",
                        help="로봇 없이 분류 결과만 확인 (웹캠만 필요)")
    parser.add_argument("--slew", type=float, default=SLEW_LIMIT_SAFE,
                        help=f"프레임 간 변화율 제한 (기본 {SLEW_LIMIT_SAFE}, "
                             f"동작 확인 후 {SLEW_LIMIT_NORMAL} 권장)")
    args = parser.parse_args()

    model, label_names = load_policy()

    hand = None
    release_torque = None
    if not args.no_robot:
        import struct
        from aero_open_sdk.aero_hand import CTRL_TOR
        from config import open_hand

        hand = open_hand()
        print(f"Aero Hand connected: {hand.ser.port}")

        def release_torque():
            msg = struct.pack("<2B7H", CTRL_TOR & 0xFF, 0x00, *[0] * N_CHANNELS)
            hand.ser.write(msg)
            hand.ser.flush()

        send_normalized(hand, PAPER)
        time.sleep(0.5)

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

    sent = list(PAPER)
    current_pose = "paper"
    torque_on = True

    candidate = None       # 디바운스 후보 클래스
    candidate_count = 0

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
                ts_ms = int((time.time() - start_t) * 1000)
                result = lm.detect_for_video(mp_img, ts_ms)

                pred_name = "-"
                conf = 0.0

                if result.hand_landmarks:
                    draw_hand(frame, result.hand_landmarks[0])
                    feat = extract_features(result.hand_world_landmarks[0])

                    with torch.no_grad():
                        logits = model(torch.from_numpy(feat).unsqueeze(0))
                        probs = torch.softmax(logits, dim=1)[0]
                    pred = int(probs.argmax())
                    conf = float(probs[pred])
                    pred_name = label_names[pred]

                    # 확신이 충분할 때만 디바운스 카운터를 올린다.
                    if conf >= CONF_THRESHOLD:
                        if pred_name == candidate:
                            candidate_count += 1
                        else:
                            candidate = pred_name
                            candidate_count = 1

                        if (candidate_count >= DEBOUNCE_FRAMES
                                and candidate != current_pose):
                            current_pose = candidate
                            print(f"→ {current_pose}  (conf {conf:.2f})")
                    else:
                        candidate_count = 0
                else:
                    candidate_count = 0

                # ---------- 전송 ----------
                target = POSES[current_pose]
                now = time.time()
                if (hand is not None and torque_on
                        and (now - last_send) >= 1.0 / SEND_HZ):
                    limited = []
                    for i in range(N_CHANNELS):
                        delta = target[i] - sent[i]
                        delta = max(-args.slew, min(args.slew, delta))
                        limited.append(clamp01(sent[i] + delta))
                    send_normalized(hand, limited)
                    sent = limited
                    last_send = now

                # ---------- HUD ----------
                mode = "NO-ROBOT" if hand is None else (
                    "torque:ON" if torque_on else "torque:OFF")
                cv2.putText(frame, f"{mode}  slew={args.slew}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                bar_color = (0, 200, 255) if conf >= CONF_THRESHOLD else (100, 100, 100)
                cv2.putText(frame, f"pred: {pred_name}  {conf:.2f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, bar_color, 2)
                cv2.rectangle(frame, (10, 85), (210, 100), (60, 60, 60), 1)
                cv2.rectangle(frame, (10, 85), (10 + int(conf * 200), 100),
                              bar_color, -1)
                cv2.line(frame, (10 + int(CONF_THRESHOLD * 200), 82),
                         (10 + int(CONF_THRESHOLD * 200), 103), (255, 255, 255), 1)

                cv2.putText(frame, f"ROBOT POSE: {current_pose}", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"debounce {candidate_count}/{DEBOUNCE_FRAMES}",
                            (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.putText(frame, "[q]uit  [SPACE]toggle torque",
                            (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                cv2.imshow("RPS BC policy - q to quit", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" ") and hand is not None:
                    torque_on = not torque_on
                    if not torque_on:
                        release_torque()
                    else:
                        sent = list(sent)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if hand is not None:
            release_torque()
            hand.close()
        print("done")


if __name__ == "__main__":
    main()
