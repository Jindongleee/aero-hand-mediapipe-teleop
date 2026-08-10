"""MediaPipe 랜드마크 → BC 입력 특징 변환.

BC 정책의 관측(observation)을 만드는 곳. 화면 정규화 좌표(`hand_landmarks`)
대신 **월드 랜드마크**(`hand_world_landmarks`)를 쓴다.

화면 정규화 좌표를 쓰면 안 되는 이유:
  - x 는 화면 가로폭, y 는 세로폭 기준이라 축마다 스케일이 다르다.
    16:9 화면에서는 x 가 1.78배 눌려서 각도·거리 계산이 왜곡된다.
  - z 는 또 다른 스케일이라 xyz 를 섞은 3D 연산이 원리적으로 맞지 않는다.

월드 랜드마크는 미터 단위 실제 3D 좌표라 이 왜곡이 없다.

그 위에 정규 자세(canonical pose)로 정렬한다. 손의 위치·방향·크기를 제거해서
"손이 어떤 모양인가"만 남기는 과정이다. 이게 있으면 손을 기울이거나 카메라에서
멀어져도 같은 특징이 나오므로, 적은 데이터로도 실물에서 잘 버틴다.
"""

import numpy as np

# 랜드마크 인덱스
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

N_LANDMARKS = 21
FEATURE_DIM = N_LANDMARKS * 3  # 63


def to_xyz(landmarks) -> np.ndarray:
    """랜드마크 리스트 → (21, 3) 배열."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def canonicalize(pts: np.ndarray) -> np.ndarray:
    """손의 위치·방향·크기를 제거하고 손모양만 남긴다.

    좌표계 정의:
      원점 = 손목
      y축  = 손목 → 중지 MCP (손가락이 뻗는 방향)
      z축  = 손바닥 법선 (검지 MCP 와 소지 MCP 로 만든 평면의 수직)
      x축  = y × z (오른손 좌표계 완성)
      스케일 = 손목 → 중지 MCP 거리로 나눔

    Args:
        pts: (21, 3) 월드 랜드마크

    Returns:
        (21, 3) 정규 자세로 정렬된 좌표
    """
    pts = pts - pts[WRIST]

    y_axis = pts[MIDDLE_MCP]
    hand_size = float(np.linalg.norm(y_axis))
    if hand_size < 1e-6:
        return np.zeros_like(pts)
    y_axis = y_axis / hand_size

    # 손바닥 평면의 법선. 검지 MCP 와 소지 MCP 가 손바닥 폭을 정의한다.
    z_axis = np.cross(pts[INDEX_MCP], pts[PINKY_MCP])
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < 1e-6:
        return np.zeros_like(pts)
    z_axis = z_axis / z_norm

    # y 와 z 가 완전히 직교하지 않을 수 있으므로 z 를 y 에 대해 직교화한다.
    z_axis = z_axis - np.dot(z_axis, y_axis) * y_axis
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < 1e-6:
        return np.zeros_like(pts)
    z_axis = z_axis / z_norm

    x_axis = np.cross(y_axis, z_axis)

    rotation = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3, 3)
    return (pts @ rotation.T) / hand_size


def extract_features(world_landmarks) -> np.ndarray:
    """월드 랜드마크 → (63,) float32 특징 벡터.

    BC 정책의 입력. collect / train / play 세 곳이 반드시 같은 함수를 써야 한다.
    여기가 어긋나면 학습은 잘 되는데 실물에서만 틀리는, 찾기 어려운 버그가 된다.
    """
    pts = to_xyz(world_landmarks)
    return canonicalize(pts).reshape(-1).astype(np.float32)
