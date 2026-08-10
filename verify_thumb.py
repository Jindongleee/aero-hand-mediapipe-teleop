"""엄지 벌림-굽힘 간섭이 실제 손에서 해소됐는지 측정.

기존 매핑은 벌림을 엄지 TIP 과 소지 MCP 사이 거리로 쟀다. 그래서 벌림 자세를
그대로 둔 채 **말단만 말아도** TIP 이 손바닥 쪽으로 들어오면서 벌림값이 같이
떨어졌다. 주먹을 쥐면 엄지 벌림이 제멋대로 움직이던 원인이다.

이 스크립트는 같은 프레임에 대해 기존 공식과 새 매핑을 나란히 계산해서,
'엄지 말단만 마는' 동작 동안 각각이 얼마나 흔들리는지 비교한다.

사용법:
  .venv/bin/python verify_thumb.py

  1) 손가락 4개는 편 채로 엄지를 옆으로 벌린 자세를 잡는다
  2) SPACE 로 측정 시작
  3) **벌림 자세는 그대로 두고** 엄지 말단 마디만 접었다 폈다 반복
  4) SPACE 로 종료 → 결과 출력

판정:
  새 벌림(ch0)의 변동이 기존 공식보다 뚜렷하게 작아야 하고,
  텐던(ch2)은 반대로 크게 움직여야 한다. 말림 정보가 사라진 게 아니라
  올바른 채널로 옮겨갔다는 뜻이다.
"""

import os
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from landmarks import to_xyz
from mapping import compute_raw

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MIN_FRAMES = 60
KEY_LOCKOUT_S = 1.0

# 기존 hand_mirror.py 의 상수
OLD_ABD_MIN, OLD_ABD_MAX = 0.55, 1.40


def old_abduction(screen_pts: np.ndarray) -> float:
    """기존 공식 — 화면 정규화 좌표에서 thumb TIP ↔ pinky MCP 거리."""
    hand_size = float(np.linalg.norm(screen_pts[9] - screen_pts[0]) + 1e-6)
    spread = float(np.linalg.norm(screen_pts[4] - screen_pts[17])) / hand_size
    return max(0.0, min(1.0, (spread - OLD_ABD_MIN) / (OLD_ABD_MAX - OLD_ABD_MIN)))


def main():
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

    print(__doc__)

    old_vals, new_abd, new_ten = [], [], []
    recording = False
    start_t = time.time()
    rec_t0 = 0.0

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

                o = a = t = None
                if result.hand_landmarks:
                    screen = to_xyz(result.hand_landmarks[0])
                    raw = compute_raw(result.hand_world_landmarks[0])
                    o, a, t = old_abduction(screen), float(raw[0]), float(raw[2])
                    if recording:
                        old_vals.append(o)
                        new_abd.append(a)
                        new_ten.append(t)

                locked = recording and (time.time() - rec_t0) < KEY_LOCKOUT_S
                enough = len(new_abd) >= MIN_FRAMES

                if not recording:
                    head, col = "SPACE to start measuring", (255, 255, 255)
                elif not enough or locked:
                    head = f"MEASURING  {len(new_abd)}/{MIN_FRAMES}"
                    col = (0, 165, 255)
                else:
                    head, col = "SPACE to finish", (0, 220, 0)
                cv2.putText(frame, head, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

                if o is not None:
                    for i, (nm, v) in enumerate(
                            [("OLD abd", o), ("NEW abd", a), ("NEW tendon", t)]):
                        y = 70 + i * 26
                        cv2.putText(frame, f"{nm:11s} {v:.3f}", (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.rectangle(frame, (200, y - 14), (400, y - 2),
                                      (60, 60, 60), 1)
                        cv2.rectangle(frame, (200, y - 14),
                                      (200 + int(min(v, 1.0) * 200), y - 2),
                                      (0, 200, 255), -1)
                else:
                    cv2.putText(frame, "NO HAND", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                cv2.putText(frame, "keep the spread FIXED, curl only the thumb tip",
                            (10, frame.shape[0] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

                cv2.imshow("Verify thumb decoupling", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    if not recording:
                        recording, rec_t0 = True, time.time()
                    elif enough and not locked:
                        break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(new_abd) < MIN_FRAMES:
        print(f"\n측정 프레임 부족 ({len(new_abd)}개). 다시 실행하세요.")
        return

    ov, na, nt = np.array(old_vals), np.array(new_abd), np.array(new_ten)
    print(f"\n{len(na)} 프레임 측정\n")
    print(f"{'지표':24s} {'변동폭':>10s} {'표준편차':>10s}")
    print("-" * 48)
    print(f"{'기존 벌림 (TIP 거리)':24s} {ov.max()-ov.min():10.4f} {ov.std():10.4f}")
    print(f"{'새 벌림 ch0 (CMC 각도)':24s} {na.max()-na.min():10.4f} {na.std():10.4f}")
    print(f"{'새 텐던 ch2 (말림)':24s} {nt.max()-nt.min():10.4f} {nt.std():10.4f}")

    print()
    if nt.max() - nt.min() < 0.02:
        print("텐던 채널이 거의 안 움직였습니다 — 엄지 말단을 충분히 말지 않은 것 같습니다.")
        print("다시 실행해서 말단을 확실히 접었다 펴 주세요.")
        return

    ratio = (ov.max() - ov.min()) / max(na.max() - na.min(), 1e-6)
    print(f"기존 벌림이 새 벌림보다 {ratio:.1f}배 더 흔들렸습니다.")
    if ratio > 2.0:
        print("→ 벌림-굽힘 간섭이 해소되었습니다.")
    else:
        print("→ 차이가 뚜렷하지 않습니다. 벌림 자세를 고정한 채 말단만 말았는지 확인하세요.")


if __name__ == "__main__":
    main()
