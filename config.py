"""공용 설정 — 시리얼 포트 해석 및 채널 정의.

포트 하드코딩을 없애기 위한 모듈.
기존에는 파일마다 macOS 경로가 제각각 박혀 있었다
(`/dev/cu.usbmodem101`, `/dev/cu.usbmodem1101`).

해석 순서:
  1. 환경변수 AERO_HAND_PORT 가 있으면 그 값
  2. 없으면 None → SDK 의 AeroHand() 자동 탐지에 위임
     (SDK 는 /dev/serial/by-id/ 에서 Espressif USB JTAG 장치를 찾는다)

리눅스에서는 보통 아무것도 지정할 필요가 없다.
"""

import os

# ---------- 시리얼 포트 ----------

ENV_PORT_KEY = "AERO_HAND_PORT"
BAUD = 921600


SERIAL_BY_ID_DIR = "/dev/serial/by-id/"
ESP32_PREFIX = "usb-Espressif_USB_JTAG_serial_debug_unit_"


def resolve_port() -> str | None:
    """환경변수 지정이 있으면 반환, 없으면 None (SDK 자동 탐지 사용)."""
    port = os.environ.get(ENV_PORT_KEY)
    return port if port else None


def detect_port() -> str:
    """구체적인 포트 경로를 반드시 반환. Serial() 을 직접 여는 쪽에서 사용.

    SDK 의 자동 탐지와 같은 규칙 (/dev/serial/by-id/ 의 Espressif 장치).
    by-id 심볼릭 링크를 쓰므로 ttyACM 번호가 바뀌어도 안전하다.
    """
    port = resolve_port()
    if port:
        return port

    if not os.path.isdir(SERIAL_BY_ID_DIR):
        raise RuntimeError(
            f"{SERIAL_BY_ID_DIR} 가 없음. 손이 연결되어 있는지 확인.\n"
            f"  → 포트를 직접 지정하려면 {ENV_PORT_KEY} 환경변수 사용."
        )

    found = [d for d in os.listdir(SERIAL_BY_ID_DIR) if ESP32_PREFIX in d]
    if not found:
        raise RuntimeError(
            f"Aero Hand 시리얼 장치를 찾지 못함. USB 연결 확인.\n"
            f"  → 포트를 직접 지정하려면 {ENV_PORT_KEY} 환경변수 사용."
        )
    if len(found) > 1:
        raise RuntimeError(
            f"장치가 여러 개 감지됨: {found}\n"
            f"  → {ENV_PORT_KEY} 환경변수로 하나를 지정할 것."
        )
    return os.path.join(SERIAL_BY_ID_DIR, found[0])


# SDK 는 시리얼 읽기 타임아웃을 0.01초로 잡는다. GET_POS 같은 조회 명령은
# 손이 응답할 시간이 그보다 오래 걸릴 때가 있어서, 첫 응답을 놓치면 다음 읽기가
# 프레임 경계에서 어긋난 채 16바이트를 가져온다 (opcode 가 엉뚱한 값으로 나옴).
READ_TIMEOUT_S = 0.2


def open_hand():
    """AeroHand 인스턴스 생성. 포트는 위 규칙으로 해석."""
    from aero_open_sdk.aero_hand import AeroHand

    hand = AeroHand(port=resolve_port())
    hand.ser.timeout = READ_TIMEOUT_S
    return hand


# ---------- 액추에이터 채널 ----------
# SDK 의 actuation_names 와 동일한 순서.
# 엄지는 관절이 4개(cmc_abd, cmc_flex, mcp, ip)인데 액추에이터는 3개뿐이다.
# ch2(thumb_tendon)가 MCP 와 IP 를 함께 당기는 언더액추에이티드 구조라,
# 엄지 말단(IP)의 굽힘은 ch2 가 지배한다.

CH_THUMB_ABD = 0    # thumb_cmc_abd_act
CH_THUMB_FLEX = 1   # thumb_cmc_flex_act
CH_THUMB_TENDON = 2  # thumb_tendon_act  ← MCP + IP 동시 구동
CH_INDEX = 3
CH_MIDDLE = 4
CH_RING = 5
CH_PINKY = 6

N_CHANNELS = 7

CH_NAMES = [
    "TH-ABD", "TH-FLX", "TH-TEN", "INDEX", "MIDDLE", "RING", "PINKY",
]


# ---------- 안전 파라미터 ----------
# 손이 한 대뿐이라 텐던 손상 시 복구가 안 된다.
# 처음 구동하거나 새 정책을 붙일 때는 SLEW_LIMIT_SAFE 로 시작한다.

SLEW_LIMIT_SAFE = 0.05   # 첫 구동 / BC 정책 검증용
SLEW_LIMIT_NORMAL = 0.15  # 동작 확인 후


# ---------- 손별 엄지 안전 범위 (스톨 방지) ----------
# 엄지 벌림/굽힘(ch0/ch1)은 개체·펌웨어(좌/우)마다 편안한 방향이 반대다.
# 정규화 명령 [0,1] 을 아래 (lo,hi) 로 재매핑해서 스톨존을 아예 안 건드린다.
#
# 오른손(2026-08-13 실측, GUI 스윕):
#   ch0 는 0.68→0mA(편안), 0.60→-500, 0.48→-3126mA(하드 스톨). → 하한 ~0.6
#   ch1 는 0.17→-572, 0.145→-1703mA(스톨). → 하한 ~0.16
#   즉 ch0·ch1 이 높아야 편안. 낮추면 수 A 로 서보를 태운다.
# 왼손(LEFT 펌웨어): 방향이 반대(낮은쪽 편안)로 추정되나 정밀 실측 전 → 미측정.

# 보드 MAC → 좌/우
HAND_MAC_SIDE = {
    "90:70:69:12:D2:0C": "right",
    "98:A3:16:F7:51:14": "left",
}

# 측정된 손만 등록. (ch0_lo, ch0_hi, ch1_lo, ch1_hi)
# lo>hi 면 방향 반전(사람 손과 반대로 움직일 때). 안전범위는 min/max 로 유지됨.
#
# ch0 스톨은 사실 ch0·ch1 을 "둘 다 낮게" 몰 때만 나는 엄지 커플링 바인딩이었다
# (SDK 경고: 엄지 액추에이션은 독립이 아님). ch1 을 클램프 하한 0.30 에 둔 채
# ch0 을 0.20 까지 내려도 전 구간 0 mA 로 확인(2026-08-13). → ch1 하한이 바인딩을
# 막으므로 ch0 은 거의 풀레인지 사용 가능. (원래 스톨은 ch1 이 0.05까지 떨어질 때였음)
THUMB_SAFE_RANGE = {
    # ch0: 정방향 [0.20,1.00](거의 풀). ch1: 반전 [1.00,0.30](하한 0.30이 커플링 바인딩 차단).
    "right": (0.20, 1.00, 1.00, 0.30),
    # "left": 왼손 실측 후 채울 것 (오른손과 반대 방향으로 추정)
}


def detect_hand_side() -> str | None:
    """연결된 보드 MAC 으로 좌/우 판별. 못 찾으면 None (→ 레거시 클램프)."""
    if not os.path.isdir(SERIAL_BY_ID_DIR):
        return None
    try:
        entries = os.listdir(SERIAL_BY_ID_DIR)
    except OSError:
        return None
    for d in entries:
        if ESP32_PREFIX not in d:
            continue
        for mac, side in HAND_MAC_SIDE.items():
            if mac in d:
                return side
    return None
