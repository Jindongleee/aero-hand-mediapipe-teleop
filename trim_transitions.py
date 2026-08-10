"""라벨이 바뀐 직후의 전환 프레임을 잘라낸다.

`collect_rps.py` 에 settle 구간이 없던 시절 모은 데이터를 정리하는 용도.
녹화 키를 누른 직후에는 아직 포즈를 만드는 중이라, 붙은 라벨과 실제 손모양이
다르다. 잘못 붙은 라벨이므로 학습에서 빼는 게 맞다.

실측 근거:
  1561 샘플로 학습했을 때 검증 오분류 10건이 **전부** rock 블록의 앞 28프레임
  (인덱스 1~28) 에 몰렸다. 블록은 0~525 인데 28 이후로는 하나도 틀리지 않았다.

사용법:
  .venv/bin/python trim_transitions.py [--frames 30]

원본은 `*_untrimmed.npz` 로 남긴다.
"""

import argparse
import os
import shutil

import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "rps_demos.npz")
LABEL_NAMES = ["rock", "scissors", "paper"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30,
                        help="각 라벨 블록 앞에서 버릴 프레임 수 (기본 30)")
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"데이터가 없습니다: {DATA_PATH}")

    d = np.load(DATA_PATH)
    obs, labels = d["obs"], d["labels"]

    # 라벨이 바뀌는 지점을 찾아 연속 블록으로 나눈다.
    # 같은 라벨이라도 중간에 일시정지 후 다시 녹화했으면 별개 블록이므로,
    # 라벨값 변화만 보고 나누는 것으로 충분하다.
    boundaries = [0] + (np.where(np.diff(labels) != 0)[0] + 1).tolist() + [len(labels)]

    keep = np.zeros(len(labels), dtype=bool)
    print(f"블록 {len(boundaries) - 1}개, 각 앞 {args.frames} 프레임 제거\n")
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        cut = min(args.frames, e - s)
        keep[s + cut:e] = True
        print(f"  블록 {i}: {LABEL_NAMES[labels[s]]:9s} "
              f"{s:5d}~{e - 1:5d} ({e - s:4d}개) → {e - s - cut:4d}개 유지")

    backup = DATA_PATH.replace(".npz", "_untrimmed.npz")
    if not os.path.exists(backup):
        shutil.copy(DATA_PATH, backup)
        print(f"\n원본 백업: {backup}")

    np.savez(DATA_PATH, obs=obs[keep], labels=labels[keep])

    counts = np.bincount(labels[keep], minlength=len(LABEL_NAMES))
    print(f"\n{len(labels)} → {int(keep.sum())} 샘플")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:9s} {counts[i]:5d}")


if __name__ == "__main__":
    main()
