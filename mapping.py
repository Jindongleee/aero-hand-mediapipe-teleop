"""손 랜드마크 → 7채널 구동값 매핑 (+ 캘리브레이션).

기존 `hand_mirror.py` 의 인라인 매핑이 가진 문제를 정리한 모듈.

## 기존 매핑의 결함

1. 엄지에 4손가락과 같은 코사인 공식을 그대로 썼다.
   엄지는 MCP-IP-TIP 이 굽혀도 거의 일직선이라 각도 변화폭이 작아서,
   bend 값이 좁은 대역에 갇히고 로봇 엄지가 거의 안 움직인다.

2. ch1 과 ch2 에 같은 값을 스케일만 바꿔 넣었다 (둘 다 `t_bend * 0.85`).
   상관계수 1.0 이라 독립 자유도 2개가 1개로 낭비된다.

3. 벌림을 엄지 TIP 과 소지 MCP 사이 거리로 쟀다.
   벌리지 않고 **굽히기만 해도** TIP 이 손바닥으로 들어와 거리가 줄어서,
   주먹을 쥐면 벌림값도 같이 0 으로 떨어진다.

4. 화면 정규화 좌표를 썼고 (축마다 스케일이 다름), 정규화 기준이
   손목 회전에 따라 변하며, 벌림 범위가 하드코딩(0.55~1.40)이었다.

## 이 모듈의 접근

- 모든 계산을 **월드 랜드마크의 정규 자세**에서 한다 (`landmarks.canonicalize`).
  손의 위치·방향·크기가 제거되므로 4번이 통째로 사라진다.

- 엄지를 세 개의 **서로 독립인** 신호로 분해한다.
  CMC→MCP 벡터를 손바닥 평면 기준으로 나누면:
      평면 안쪽 각도  → 벌림 (ch0)  : 엄지가 검지에서 멀어지는 정도
      평면 바깥 각도  → 굽힘 (ch1)  : 엄지가 손바닥을 가로지르는 정도
  여기에 말단 말림을 따로 잰다:
      MCP 와 IP 관절 각도 평균 → 텐던 (ch2)
  ch2 는 손목 쪽 두 신호와 다른 관절을 보므로 3번이 해결된다.

- 범위는 하드코딩하지 않고 **실행 시 캘리브레이션**으로 잡는다.

## 부호 규약에 관한 주의

ch0 과 ch1 중 어느 쪽이 로봇의 벌림이고 어느 쪽이 굽힘인지, 그리고 각
채널의 증가 방향이 사람 손과 같은지는 실물로 한 번 확인해야 한다.
캘리브레이션이 범위는 흡수하지만 **방향은 흡수하지 못한다.**
반대로 움직이면 `calibration.json` 의 `invert` 를 켜면 된다.
"""

import json
import os

import numpy as np

from landmarks import (
    to_xyz, canonicalize, WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP,
)

# 랜드마크 인덱스 — 엄지
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

# (MCP, PIP, TIP) — 4손가락
FINGER_JOINTS = [
    (5, 6, 8),     # index
    (9, 10, 12),   # middle
    (13, 14, 16),  # ring
    (17, 18, 20),  # pinky
]

N_CHANNELS = 7
CALIB_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")


def joint_bend(pts: np.ndarray, a: int, b: int, c: int) -> float:
    """세 점이 이루는 굽힘. 0 = 펴짐(일직선), 1 = 완전히 접힘.

    b 를 꼭짓점으로 하는 각도의 코사인을 쓴다.
    """
    v1 = pts[b] - pts[a]
    v2 = pts[c] - pts[b]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))
    return (1.0 - cos) / 2.0


def compute_raw(world_landmarks) -> np.ndarray:
    """월드 랜드마크 → 7채널 원시 신호 (아직 0~1 로 정규화되지 않음).

    Returns:
        (7,) float32. 순서는 config.CH_NAMES 와 같다.
    """
    pts = canonicalize(to_xyz(world_landmarks))
    raw = np.zeros(N_CHANNELS, dtype=np.float32)

    # ---- 엄지 ----
    # 정규 자세에서 y 는 손가락이 뻗는 방향, z 는 손바닥 법선이다.
    # CMC→MCP 벡터를 이 좌표계로 보면 벌림과 굽힘이 자연스럽게 갈린다.
    thumb_dir = pts[THUMB_MCP] - pts[THUMB_CMC]
    n = float(np.linalg.norm(thumb_dir))
    if n > 1e-6:
        thumb_dir = thumb_dir / n

        # ch0 벌림: 손바닥 평면 안에서 손가락 방향(y)으로부터 벌어진 각도.
        # 평면 성분만 쓰므로 엄지를 말아도 값이 흔들리지 않는다.
        in_plane = np.array([thumb_dir[0], thumb_dir[1]], dtype=np.float32)
        p = float(np.linalg.norm(in_plane))
        raw[0] = float(np.arctan2(abs(in_plane[0]), in_plane[1])) if p > 1e-6 else 0.0

        # ch1 굽힘: 손바닥 평면에서 벗어난 각도 (엄지가 손바닥을 가로지름).
        raw[1] = float(np.arcsin(max(-1.0, min(1.0, thumb_dir[2]))))

    # ch2 텐던: MCP 와 IP 를 하나의 텐던이 함께 당기므로 두 관절을 평균낸다.
    raw[2] = 0.5 * (
        joint_bend(pts, THUMB_CMC, THUMB_MCP, THUMB_IP)
        + joint_bend(pts, THUMB_MCP, THUMB_IP, THUMB_TIP)
    )

    # ---- 4손가락 ----
    for i, (mcp, pip, tip) in enumerate(FINGER_JOINTS):
        raw[3 + i] = joint_bend(pts, mcp, pip, tip)

    return raw


class Calibration:
    """채널별 원시 신호 범위를 관측해서 0~1 로 정규화한다.

    하드코딩된 범위를 없애기 위한 것. 사람마다, 카메라 거리마다 원시 신호의
    범위가 다르므로 실행 시점에 관측해서 잡는다.
    """

    def __init__(self, lo=None, hi=None, invert=None):
        self.lo = np.full(N_CHANNELS, np.inf, dtype=np.float32) if lo is None \
            else np.asarray(lo, dtype=np.float32)
        self.hi = np.full(N_CHANNELS, -np.inf, dtype=np.float32) if hi is None \
            else np.asarray(hi, dtype=np.float32)
        self.invert = np.zeros(N_CHANNELS, dtype=bool) if invert is None \
            else np.asarray(invert, dtype=bool)

    def observe(self, raw: np.ndarray) -> None:
        """캘리브레이션 중 호출. 관측된 최소·최대를 넓혀간다."""
        self.lo = np.minimum(self.lo, raw)
        self.hi = np.maximum(self.hi, raw)

    def is_ready(self) -> bool:
        return bool(np.all(np.isfinite(self.lo)) and np.all(np.isfinite(self.hi)))

    def apply(self, raw: np.ndarray) -> np.ndarray:
        """원시 신호 → 0~1 정규화값.

        범위가 거의 0 인 채널(그 관절을 캘리브 중에 안 움직인 경우)은
        0.5 로 둔다. 0 으로 나눠 폭주시키는 것보다 안전하다.
        """
        span = self.hi - self.lo
        out = np.full(N_CHANNELS, 0.5, dtype=np.float32)
        valid = span > 1e-6
        out[valid] = (raw[valid] - self.lo[valid]) / span[valid]
        out = np.clip(out, 0.0, 1.0)
        out[self.invert] = 1.0 - out[self.invert]
        return out

    def save(self, path: str = CALIB_PATH) -> None:
        with open(path, "w") as f:
            json.dump(
                {
                    "lo": self.lo.tolist(),
                    "hi": self.hi.tolist(),
                    "invert": self.invert.tolist(),
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str = CALIB_PATH):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            d = json.load(f)
        return cls(lo=d["lo"], hi=d["hi"], invert=d.get("invert"))
