"""Step 8: 연속 회귀 BC 학습.

`data/teleop_demos.npz` 만 읽어서 지도학습한다. 순수 오프라인 —
학습 중 카메라나 로봇에 접근하지 않는다.

  관측 obs    : 63차원 (정규자세 랜드마크)
  행동 action : 7채널 연속값 (0~1)
  모델        : MLP 63 -> 128 -> 128 -> 7, 출력 sigmoid
  손실        : MSE

가위바위보 쪽(`train_rps.py`)과 다른 점:
  출력층 3 -> 7, CrossEntropy -> MSE, argmax 후처리 없음.
  정책이 관절 명령을 직접 낸다.

평가는 accuracy 가 아니라 **채널별 MAE** 로 본다. 전체 MSE 하나만 보면
어느 채널이 망가졌는지 안 보이고, 실물에서는 그 손가락만 안 움직이는
형태로 드러난다.

실행:
  .venv/bin/python train_teleop_bc.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import CH_NAMES, N_CHANNELS

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "teleop_demos.npz")
OUT_PATH = os.path.join(os.path.dirname(__file__), "teleop_policy.pt")

SEED = 42
EPOCHS = 200
BATCH_SIZE = 128
LR = 1e-3
HIDDEN = 128
VAL_RATIO = 0.2


class TeleopPolicy(nn.Module):
    """63 -> 128 -> 128 -> 7, 출력은 sigmoid 로 0~1 에 가둔다.

    구동값이 정의상 0~1 이라 출력에서 범위를 보장해두면 실물에 그대로 보낼 수
    있다. clamp 로 자르면 범위 밖 구간에서 기울기가 죽는데, sigmoid 는 그렇지
    않다.
    """

    def __init__(self, in_dim: int = 63, hidden: int = HIDDEN,
                 out_dim: int = N_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"데이터가 없습니다: {DATA_PATH}")
        print("먼저 collect_teleop.py 를 실행해서 데모를 수집하세요.")
        sys.exit(1)

    d = np.load(DATA_PATH)
    obs = d["obs"].astype(np.float32)
    actions = d["actions"].astype(np.float32)
    print(f"데이터 {len(obs)} 샘플, obs {obs.shape[1]}차원, action {actions.shape[1]}채널")

    print(f"\n{'채널':8s} {'min':>7s} {'max':>7s} {'폭':>7s}")
    for i in range(actions.shape[1]):
        col = actions[:, i]
        warn = "  <-- 범위 좁음, 학습해도 잘 안 나옴" if col.max() - col.min() < 0.3 else ""
        print(f"{CH_NAMES[i]:8s} {col.min():7.3f} {col.max():7.3f} "
              f"{col.max() - col.min():7.3f}{warn}")

    return obs, actions


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    obs, actions = load_data()

    idx = rng.permutation(len(obs))
    n_val = max(1, int(len(obs) * VAL_RATIO))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_train = torch.from_numpy(obs[train_idx])
    y_train = torch.from_numpy(actions[train_idx])
    x_val = torch.from_numpy(obs[val_idx])
    y_val = torch.from_numpy(actions[val_idx])
    print(f"\n학습 {len(x_train)} / 검증 {len(x_val)}")

    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )

    model = TeleopPolicy(in_dim=obs.shape[1], out_dim=actions.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # 마지막 epoch 가 최선이라는 보장이 없다. 연속 프레임이 서로 비슷해서
    # 학습 손실은 계속 내려가는데 검증 손실은 도로 올라가는 구간이 생긴다.
    # 검증 기준 최고 시점을 따로 들고 있다가 그걸 저장한다.
    best_val = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}

        if epoch % 25 == 0 or epoch == 1:
            mark = "  *best" if epoch == best_epoch else ""
            print(f"  epoch {epoch:4d}  train_mse {total / len(x_train):.5f}  "
                  f"val_mse {val_loss:.5f}{mark}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n최고 검증 시점: epoch {best_epoch} (val_mse {best_val:.5f}) 복원")
    if best_epoch < EPOCHS * 0.5:
        print(f"  검증 손실이 절반 지점 전에 최저였습니다 — 과적합 구간이 깁니다.")
        print(f"  데이터를 더 모으거나 EPOCHS 를 줄이는 걸 권합니다.")

    model.eval()
    with torch.no_grad():
        pred = model(x_val)

    err = (pred - y_val).abs()
    mae = err.mean().item()
    print(f"\n검증 MAE (전체): {mae:.4f}  (정규화 단위, 0~1 기준)")

    print(f"\n{'채널':8s} {'MAE':>8s} {'최대오차':>9s}")
    for i in range(y_val.shape[1]):
        col = err[:, i]
        flag = "  <-- 오차 큼" if col.mean().item() > 0.05 else ""
        print(f"{CH_NAMES[i]:8s} {col.mean().item():8.4f} "
              f"{col.max().item():9.4f}{flag}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": obs.shape[1],
            "hidden": HIDDEN,
            "out_dim": actions.shape[1],
            "val_mae": mae,
        },
        OUT_PATH,
    )
    print(f"\n저장: {OUT_PATH}")

    # 0.05 는 구동 전체 범위의 5%. 이 이상이면 눈에 띄게 어긋난다.
    if mae > 0.05:
        print("\nMAE 가 0.05 를 넘습니다. 데이터를 더 모으거나, 위 채널별 표에서")
        print("범위가 좁은 채널이 있는지 확인하세요.")


if __name__ == "__main__":
    main()
