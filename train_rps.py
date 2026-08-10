"""Step 5: 가위바위보 BC 학습.

`data/rps_demos.npz` 만 읽어서 지도학습한다. 순수 오프라인 BC —
학습 중 카메라나 로봇에 접근하지 않는다.

  관측 obs    : 63차원 (21 랜드마크 x xyz, 정규 자세로 정렬됨)
  행동 action : 3-class (rock / scissors / paper)
  모델        : MLP 63 -> 64 -> 3
  손실        : CrossEntropy

모델이 작아서 CPU 로 몇 초면 끝난다. GPU 불필요.

실행:
  .venv/bin/python train_rps.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "rps_demos.npz")
OUT_PATH = os.path.join(os.path.dirname(__file__), "rps_policy.pt")

LABEL_NAMES = ["rock", "scissors", "paper"]

SEED = 42
EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3
HIDDEN = 64
VAL_RATIO = 0.2


class RPSPolicy(nn.Module):
    """63 -> 64 -> 3 MLP."""

    def __init__(self, in_dim: int = 63, hidden: int = HIDDEN, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"데이터가 없습니다: {DATA_PATH}")
        print("먼저 collect_rps.py 를 실행해서 데모를 수집하세요.")
        sys.exit(1)

    d = np.load(DATA_PATH)
    obs = d["obs"].astype(np.float32)
    labels = d["labels"].astype(np.int64)

    counts = np.bincount(labels, minlength=len(LABEL_NAMES))
    print(f"데이터 {len(obs)} 샘플")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:9s} {counts[i]:5d}")

    if (counts == 0).any():
        missing = [LABEL_NAMES[i] for i in np.where(counts == 0)[0]]
        print(f"\n샘플이 하나도 없는 클래스가 있습니다: {missing}")
        print("collect_rps.py 로 해당 포즈를 더 수집하세요.")
        sys.exit(1)

    return obs, labels


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    obs, labels = load_data()

    # 학습/검증 분할. 수집이 클래스별로 연속 구간이라 섞지 않으면
    # 검증셋이 한 클래스로만 채워진다.
    idx = rng.permutation(len(obs))
    n_val = max(1, int(len(obs) * VAL_RATIO))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_train = torch.from_numpy(obs[train_idx])
    y_train = torch.from_numpy(labels[train_idx])
    x_val = torch.from_numpy(obs[val_idx])
    y_val = torch.from_numpy(labels[val_idx])

    print(f"\n학습 {len(x_train)} / 검증 {len(x_val)}")

    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )

    model = RPSPolicy(in_dim=obs.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(x_val)
                val_loss = criterion(val_logits, y_val).item()
                val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
            print(f"  epoch {epoch:3d}  train_loss {total / len(x_train):.4f}  "
                  f"val_loss {val_loss:.4f}  val_acc {val_acc:.3f}")

    # 최종 평가 — 클래스별 정확도까지 본다.
    # 전체 정확도만 보면 한 클래스가 통째로 틀려도 눈치채기 어렵고,
    # 실물 데모에서는 그 포즈만 계속 실패하는 형태로 드러난다.
    model.eval()
    with torch.no_grad():
        pred = model(x_val).argmax(1)

    acc = (pred == y_val).float().mean().item()
    print(f"\n검증 정확도: {acc:.3f}")
    print("\n클래스별:")
    for i, name in enumerate(LABEL_NAMES):
        mask = y_val == i
        n = int(mask.sum())
        if n == 0:
            print(f"  {name:9s}   (검증셋에 없음)")
            continue
        cls_acc = (pred[mask] == i).float().mean().item()
        print(f"  {name:9s} {cls_acc:.3f}  ({n} 샘플)")

    print("\n혼동 행렬 (행=정답, 열=예측):")
    print(f"  {'':9s}" + "".join(f"{n:>10s}" for n in LABEL_NAMES))
    for i, name in enumerate(LABEL_NAMES):
        row = [int(((y_val == i) & (pred == j)).sum()) for j in range(len(LABEL_NAMES))]
        print(f"  {name:9s}" + "".join(f"{v:10d}" for v in row))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": obs.shape[1],
            "hidden": HIDDEN,
            "label_names": LABEL_NAMES,
            "val_acc": acc,
        },
        OUT_PATH,
    )
    print(f"\n저장: {OUT_PATH}")

    if acc < 0.95:
        print("\n정확도가 0.95 미만입니다. 실물 데모 전에 데이터를 더 모으는 걸 권합니다.")


if __name__ == "__main__":
    main()
