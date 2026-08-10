"""Step 7: 연속 회귀 BC 데이터 수집.

`collect_rps.py` 가 3-class 라벨을 모았다면, 이쪽은 **7채널 연속 명령**을 모은다.

  관측 obs    : 63차원 정규자세 랜드마크 (landmarks.extract_features)
  행동 action : 7채널 정규화 구동값 (mapping.compute_raw → 캘리브레이션 적용)

expert 는 `mapping.py` 의 해석적 매핑이다. 텔레옵이 매 프레임 계산하는 그 값을
그대로 정답으로 쓴다. 즉 학습된 정책은 손으로 짠 수식을 근사하게 된다.
BC 골격(관측 → expert 행동의 지도학습)은 지키지만, "사람 시연에서 새 기술을
배운다"는 의미는 아니라는 점은 분명히 해둔다.

**로봇이 필요 없다.** expert 라벨이 사람 손 랜드마크에서 계산되지 로봇 상태에서
오는 게 아니기 때문이다. 카메라만 있으면 수집도 학습도 된다.

수집 요령:
  3개 포즈만 찍으면 그 3점만 재현하는 정책이 된다. 상태공간을 골고루 훑어야 한다.
    - 손가락 하나씩 개별로 굽혔다 펴기
    - 여러 손가락 조합
    - 엄지 벌림/모음, 엄지 말림
    - 가위바위보 같은 실제 쓸 자세
    - 중간 자세들 (반쯤 굽힌 상태)
  손 위치·각도·카메라 거리도 바꿔가며.

조작:
  SPACE  녹화 시작/일시정지
  u      마지막 50 프레임 취소
  q      저장하고 종료

실행:
  .venv/bin/python collect_teleop.py

  먼저 캘리브레이션이 있어야 한다. 없으면:
  .venv/bin/python hand_mirror.py --no-robot --calibrate
"""

import os
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from landmarks import extract_features
from mapping import compute_raw, Calibration, CALIB_PATH
from config import CH_NAMES, N_CHANNELS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(DATA_DIR, "teleop_demos.npz")

TARGET_SAMPLES = 6000   # 30fps 기준 약 3분 30초
UNDO_FRAMES = 50

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


def load_existing():
    if not os.path.exists(OUT_PATH):
        return [], []
    d = np.load(OUT_PATH)
    obs, acts = list(d["obs"]), list(d["actions"])
    print(f"기존 데이터 {len(obs)}개 로드, 이어서 수집합니다.")
    return obs, acts


def save(obs, acts):
    os.makedirs(DATA_DIR, exist_ok=True)
    if not obs:
        print("수집된 데이터가 없어 저장하지 않음.")
        return
    A = np.array(acts, dtype=np.float32)
    np.savez(OUT_PATH, obs=np.array(obs, dtype=np.float32), actions=A)
    print(f"\n저장: {OUT_PATH}  ({len(obs)} 샘플)")

    # 채널별 커버리지. 어떤 채널의 범위가 좁으면 그 관절을 덜 움직인 것이고,
    # 학습된 정책도 그 채널을 제대로 못 낸다.
    print(f"\n{'채널':8s} {'min':>7s} {'max':>7s} {'폭':>7s} {'표준편차':>9s}")
    for i in range(N_CHANNELS):
        col = A[:, i]
        warn = "  <-- 범위 좁음" if col.max() - col.min() < 0.3 else ""
        print(f"{CH_NAMES[i]:8s} {col.min():7.3f} {col.max():7.3f} "
              f"{col.max() - col.min():7.3f} {col.std():9.3f}{warn}")


def main():
    calib = Calibration.load()
    if calib is None:
        raise SystemExit(
            f"캘리브레이션이 없습니다: {CALIB_PATH}\n"
            "먼저 실행하세요:\n"
            "  .venv/bin/python hand_mirror.py --no-robot --calibrate"
        )
    print(f"캘리브레이션 로드: {CALIB_PATH}")

    obs, acts = load_existing()

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없음. USB 웹캠 연결 확인.")

    recording = False
    start_t = time.time()

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

                action = None
                if result.hand_landmarks:
                    draw_hand(frame, result.hand_landmarks[0])
                    world = result.hand_world_landmarks[0]
                    action = calib.apply(compute_raw(world))
                    if recording:
                        obs.append(extract_features(world))
                        acts.append(action.astype(np.float32))

                # ---------- HUD ----------
                state = "REC" if recording else "paused"
                color = (0, 0, 255) if recording else (200, 200, 200)
                cv2.putText(frame, f"{state}   {len(obs)}/{TARGET_SAMPLES}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                bar = int(min(len(obs) / TARGET_SAMPLES, 1.0) * 300)
                cv2.rectangle(frame, (10, 40), (310, 52), (60, 60, 60), 1)
                cv2.rectangle(frame, (10, 40), (10 + bar, 52), color, -1)

                if action is None:
                    cv2.putText(frame, "NO HAND", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    y = 80
                    for i in range(N_CHANNELS):
                        v = float(action[i])
                        cv2.putText(frame, f"{CH_NAMES[i]:7s} {v:.2f}", (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.rectangle(frame, (170, y - 12), (370, y - 2),
                                      (60, 60, 60), 1)
                        cv2.rectangle(frame, (170, y - 12),
                                      (170 + int(v * 200), y - 2), (0, 200, 255), -1)
                        y += 20

                cv2.putText(frame, "SPACE=rec/pause  [u]ndo50  [q]uit save",
                            (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("Collect teleop BC - q to save and quit", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    recording = not recording
                elif key == ord("u"):
                    n = min(UNDO_FRAMES, len(obs))
                    del obs[len(obs) - n:]
                    del acts[len(acts) - n:]
                    print(f"마지막 {n} 프레임 취소")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save(obs, acts)


if __name__ == "__main__":
    main()
