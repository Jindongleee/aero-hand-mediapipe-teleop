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


def open_hand():
    """AeroHand 인스턴스 생성. 포트는 위 규칙으로 해석."""
    from aero_open_sdk.aero_hand import AeroHand

    return AeroHand(port=resolve_port())


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
