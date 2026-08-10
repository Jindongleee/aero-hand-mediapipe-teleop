"""Step 4: 가위바위보 데모 수집.

웹캠으로 손을 보면서 (월드 랜드마크 특징 63차원, 라벨) 쌍을 모은다.
**로봇은 필요 없다.** 카메라만 있으면 된다.

조작:
  r  바위(rock) 라벨로 녹화 시작
  s  가위(scissors) 라벨로 녹화 시작
  p  보(paper) 라벨로 녹화 시작
  SPACE  녹화 일시정지
  u  현재 라벨의 마지막 50 프레임 취소 (잘못 들어간 구간 되돌리기)
  q  저장하고 종료

수집 요령:
  한 자세로만 모으면 실물 데모에서 실패한다. 녹화하는 동안 손을
  좌우로 움직이고, 기울이고, 카메라와의 거리도 바꿀 것.
  (특징이 위치·회전·크기 불변이라 완전히 같은 값이 나오진 않지만,
   MediaPipe 검출 자체의 오차가 자세마다 다르므로 다양성이 여전히 중요하다.)

목표: 포즈당 500 프레임 정도. 30fps 기준 약 17초씩.

실행:
  .venv/bin/python collect_rps.py
"""

import os
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from landmarks import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(DATA_DIR, "rps_demos.npz")

# 라벨 정의. poses.py 의 POSES 키와 이름을 맞춰둔다.
LABEL_NAMES = ["rock", "scissors", "paper"]
KEY_TO_LABEL = {ord("r"): 0, ord("s"): 1, ord("p"): 2}

TARGET_PER_CLASS = 500
UNDO_FRAMES = 50

# 라벨 키를 누른 직후에는 아직 포즈를 만드는 중이라 손모양이 라벨과 다르다.
# 이 구간을 그대로 기록하면 라벨이 틀린 샘플이 섞인다.
# (실측: 이걸 안 넣었을 때 오분류 10건이 전부 rock 블록의 앞 28프레임에 몰렸다)
SETTLE_FRAMES = 30

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
    """기존 데이터가 있으면 이어서 모은다."""
    if not os.path.exists(OUT_PATH):
        return [], []
    d = np.load(OUT_PATH)
    obs = list(d["obs"])
    labels = list(d["labels"])
    print(f"기존 데이터 {len(obs)}개 로드, 이어서 수집합니다.")
    return obs, labels


def save(obs, labels):
    os.makedirs(DATA_DIR, exist_ok=True)
    if not obs:
        print("수집된 데이터가 없어 저장하지 않음.")
        return
    np.savez(
        OUT_PATH,
        obs=np.array(obs, dtype=np.float32),
        labels=np.array(labels, dtype=np.int64),
    )
    print(f"\n저장: {OUT_PATH}  ({len(obs)} 샘플)")
    counts = np.bincount(np.array(labels), minlength=len(LABEL_NAMES))
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:9s} {counts[i]:5d}")


def main():
    obs, labels = load_existing()

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없음. USB 웹캠이 연결되어 있는지 확인.")

    active_label = None  # None 이면 녹화 안 함
    settle_left = 0      # 라벨 전환 직후 버릴 프레임 수
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
                ts_ms = int((time.time() - start_t) * 1000)
                result = lm.detect_for_video(mp_img, ts_ms)

                detected = bool(result.hand_landmarks)
                if detected:
                    draw_hand(frame, result.hand_landmarks[0])
                    # 손이 검출된 프레임만 기록한다. 미검출 프레임을 넣으면
                    # 라벨은 있는데 특징은 쓰레기인 샘플이 섞인다.
                    if active_label is not None:
                        if settle_left > 0:
                            settle_left -= 1
                        else:
                            feat = extract_features(result.hand_world_landmarks[0])
                            obs.append(feat)
                            labels.append(active_label)

                # ---------- HUD ----------
                counts = np.bincount(
                    np.array(labels, dtype=np.int64) if labels else np.array([], dtype=np.int64),
                    minlength=len(LABEL_NAMES),
                )
                if active_label is None:
                    state, color = "paused", (200, 200, 200)
                elif settle_left > 0:
                    state = f"SETTLING [{LABEL_NAMES[active_label]}] {settle_left}"
                    color = (0, 165, 255)
                else:
                    state = f"REC [{LABEL_NAMES[active_label]}]"
                    color = (0, 0, 255)
                cv2.putText(frame, state, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if not detected:
                    cv2.putText(frame, "NO HAND", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                y = 100
                for i, name in enumerate(LABEL_NAMES):
                    n = int(counts[i])
                    done = "OK" if n >= TARGET_PER_CLASS else f"/{TARGET_PER_CLASS}"
                    cv2.putText(frame, f"{name:9s} {n:5d} {done}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    bar = int(min(n / TARGET_PER_CLASS, 1.0) * 200)
                    cv2.rectangle(frame, (200, y - 14), (400, y - 2), (60, 60, 60), 1)
                    cv2.rectangle(frame, (200, y - 14), (200 + bar, y - 2),
                                  (0, 200, 255), -1)
                    y += 28

                cv2.putText(frame, "[r]ock [s]cissors [p]aper  SPACE=pause  [u]ndo50  [q]uit",
                            (10, frame.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow("Collect RPS - q to save and quit", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    active_label = None
                elif key in KEY_TO_LABEL:
                    if KEY_TO_LABEL[key] != active_label:
                        settle_left = SETTLE_FRAMES
                    active_label = KEY_TO_LABEL[key]
                elif key == ord("u"):
                    n = min(UNDO_FRAMES, len(obs))
                    del obs[len(obs) - n:]
                    del labels[len(labels) - n:]
                    print(f"마지막 {n} 프레임 취소")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save(obs, labels)


if __name__ == "__main__":
    main()
