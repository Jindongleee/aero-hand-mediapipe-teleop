"""엄지 말단(IP) 강제 굴곡 진단 — 위험도 순 단계별 실행.

증상:
  엄지 말단이 기본 상태에서 접혀 있고 모터가 잡고 있어 펼 수 없다.

가설:
  엄지는 관절 4개(cmc_abd, cmc_flex, mcp, ip)에 액추에이터가 3개뿐이고,
  ch2(thumb_tendon)가 MCP 와 IP 를 함께 당긴다. PAPER 자세의 ch2 정규화값
  0.058 은 actuation 으로 환산하면 약 -0.06도, 즉 사실상 0도 명령이다.
  0도인데도 말려 있다면 텐던이 짧은 게 아니라 서보 영점이 어긋난 것이다.

단계 (뒤로 갈수록 위험):
  1. ch2 = 0.0 전송 (-15.28도). PAPER 보다 15도 더 신전. SDK 범위 내 → 위험 없음
  2. 토크 해제 후 자연 복원 관찰 → 위험 없음
  3. send_homing() 재호밍 → 낮음 (손이 자유로워야 함, 최대 175초)
  4. trim_servo(2, -N) 영점 보정 → 중간

한 번에 한 단계씩 실행한다. 3, 4 는 되돌리기 어려우므로
1, 2 에서 원인이 잡히면 거기서 멈출 것.

실행:
  .venv/bin/python thumb_diag.py                      # 기준선만 (아무것도 안 건드림)
  .venv/bin/python thumb_diag.py --step extend        # 단계 1
  .venv/bin/python thumb_diag.py --step release       # 단계 2
  .venv/bin/python thumb_diag.py --step homing        # 단계 3
  .venv/bin/python thumb_diag.py --step trim --degrees -10   # 단계 4
"""

import argparse
import sys
import time
from datetime import datetime

from aero_open_sdk.aero_hand_constants import AeroHandConstants
from config import open_hand, CH_NAMES, CH_THUMB_TENDON, N_CHANNELS
from poses import PAPER, send_normalized

LOG_PATH = "thumb_diag_log.txt"

CONST = AeroHandConstants()
LOWER = CONST.actuation_lower_limits
UPPER = CONST.actuation_upper_limits


def log(msg: str = "") -> None:
    """콘솔과 로그 파일에 동시 기록."""
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def to_normalized(actuations: list[float]) -> list[float]:
    """degree → 정규화 0~1 (send_normalized 의 역변환)."""
    return [
        (a - LOWER[i]) / (UPPER[i] - LOWER[i]) for i, a in enumerate(actuations)
    ]


def read_retry(fn, tries: int = 4):
    """조회 명령을 몇 번 재시도한다.

    응답 프레임이 한 번 어긋나면 그 뒤로도 계속 어긋난 채 읽히므로,
    실패할 때마다 입력 버퍼를 비우고 다시 요청한다.
    """
    for _ in range(tries):
        value = fn()
        if value is not None:
            return value
        time.sleep(0.1)
    return None


def snapshot(hand, label: str) -> None:
    """현재 위치·전류·온도를 한 줄씩 기록.

    전류가 진단의 핵심이다. 모터가 엄지를 굽힌 채 버티고 있다면
    ch2 전류가 다른 채널보다 뚜렷하게 높게 나온다.
    """
    log(f"\n===== {label} =====")

    acts = read_retry(hand.get_actuations)
    currs = read_retry(hand.get_actuator_currents)
    temps = read_retry(hand.get_actuator_temperatures)

    if acts is None:
        log("  위치 읽기 실패 (시리얼 응답 없음)")
        return

    norms = to_normalized(acts)
    log(f"  {'채널':8s} {'각도(deg)':>12s} {'정규화':>9s} {'전류(mA)':>10s} {'온도':>7s}")
    for i in range(N_CHANNELS):
        c = f"{currs[i]:.0f}" if currs else "-"
        t = f"{temps[i]:.0f}" if temps else "-"
        mark = "  <-- 엄지 텐던" if i == CH_THUMB_TENDON else ""
        log(f"  {CH_NAMES[i]:8s} {acts[i]:12.2f} {norms[i]:9.3f} {c:>10s} {t:>7s}{mark}")


def move_to(hand, target: list[float], seconds: float = 2.5, hz: int = 30) -> None:
    """현재 위치에서 목표까지 천천히 보간해서 이동한다.

    `send_normalized` 로 목표를 한 번에 보내면 서보가 최대 속도로 달린다.
    기준선에서 엄지 벌림이 0.899 인데 PAPER 는 0.018 이라, 그대로 보내면
    전 범위를 한 번에 튄다. 손이 한 대뿐이고 어댑터를 손으로 잡고 있는
    상황에서는 급격한 동작을 피한다.
    """
    acts = read_retry(hand.get_actuations)
    if acts is None:
        log("  현재 위치를 못 읽어 램프 없이 바로 전송한다.")
        send_normalized(hand, target)
        time.sleep(seconds)
        return

    start = to_normalized(acts)
    steps = max(1, int(seconds * hz))
    for i in range(1, steps + 1):
        t = i / steps
        send_normalized(hand, [s + (g - s) * t for s, g in zip(start, target)])
        time.sleep(1.0 / hz)


def step1_extend(hand) -> None:
    """ch2 에 정규화 0.0 을 보낸다 = -15.28도, PAPER 보다 15도 더 신전.

    지금까지 PAPER[2]=0.058(약 0도)만 써왔고 이 여유분은 안 써봤다.
    7개 채널 중 ch2 만 lower_limit 이 음수인 이유가 바로 이 신전 여유분이다.
    """
    log("\n### 단계 1: ch2 = 0.0 전송 (신전 여유분 사용)")

    target = list(PAPER)
    log(f"  현재 PAPER[2] = {PAPER[2]:.3f} → actuation "
        f"{LOWER[2] + PAPER[2] * (UPPER[2] - LOWER[2]):.2f} deg")

    target[CH_THUMB_TENDON] = 0.0
    log(f"  전송할 ch2   = 0.000 → actuation {LOWER[2]:.2f} deg")
    log("  (현재 위치에서 2.5초에 걸쳐 천천히 이동)")

    move_to(hand, target)
    time.sleep(1.5)
    snapshot(hand, "단계 1 이후")

    log("\n  >> 두 가지를 볼 것.")
    log("     1) 엄지 말단이 펴졌는가 (눈으로)")
    log("     2) TH-TEN 위치가 명령값 근처까지 갔는가, 전류가 다른 채널보다 큰가")
    log("        위치가 안 따라가고 전류만 크면 = 텐던이 기계적으로 막힌 것")
    log("        위치는 따라갔는데 여전히 접혀 있으면 = 영점이 어긋난 것 (호밍/트림)")


def step2_release(hand) -> None:
    """토크를 모두 해제하고 손이 어디로 돌아가는지 본다.

    토크 해제 후에도 굽힌 위치에 머무르면 텐던이 물리적으로 짧다는 뜻이고,
    스르륵 펴지면 모터가 잘못된 위치를 잡고 있었다는 뜻이다.
    """
    log("\n### 단계 2: 토크 전체 해제")

    hand.ctrl_torque([0] * N_CHANNELS)
    time.sleep(2.0)
    snapshot(hand, "토크 해제 후")

    log("\n  >> 엄지가 저절로 펴지는가?")
    log("     (a) 펴짐          → 영점 어긋남. 단계 3 재호밍으로 해결 가능")
    log("     (b) 손으로만 펴짐  → 텐던 과장력. 하드웨어 장력 조정 필요")
    log("     (c) 안 움직임      → 텐던 걸림/손상. 오늘 해결 불가")


def step3_homing(hand) -> None:
    """재호밍. 손이 자유롭게 움직일 수 있는 상태여야 한다."""
    log("\n### 단계 3: 재호밍")
    log("  주의: 호밍 중에는 다른 명령에 응답하지 않는다. 최대 175초.")
    log("        손에 아무것도 닿지 않게 하고, 손가락이 자유롭게 움직일 공간을 확보할 것.")

    log("  호밍 중... (최대 175초, 기다릴 것)")
    t0 = time.time()
    hand.send_homing()
    log(f"  호밍 완료 ({time.time() - t0:.1f}초)")

    time.sleep(1.0)
    send_normalized(hand, PAPER)
    time.sleep(1.5)
    snapshot(hand, "호밍 + PAPER 이후")


def step4_trim(hand, degrees: int) -> None:
    """ch2 영점을 보정한다.

    trim_servo 는 SDK GUI 가 쓰는 미세조정 API 다.
    한 번에 크게 주지 말고 -10도씩 나눠서, 매번 눈으로 확인하며 진행한다.
    과하면 반대로 늘어진다.
    """
    log(f"\n### 단계 4: ch2 영점 보정 (trim_servo, {degrees:+d}도)")

    ack = hand.trim_servo(id=CH_THUMB_TENDON, degrees=degrees)
    log(f"  trim {degrees:+d}도 적용 → ack={ack}")

    time.sleep(0.5)
    send_normalized(hand, PAPER)
    time.sleep(1.0)
    snapshot(hand, f"트림 {degrees:+d}도 이후")

    log("\n  >> 아직 접혀 있으면 같은 명령을 한 번 더 실행한다.")
    log("     누적 -90도를 넘어가면 하드웨어 문제일 가능성이 높으니 멈출 것.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step", default="baseline",
        choices=["baseline", "extend", "release", "homing", "trim"],
        help="실행할 단계. 기본값 baseline 은 아무것도 건드리지 않고 상태만 읽는다.")
    parser.add_argument("--degrees", type=int, default=-10,
                        help="trim 단계에서 적용할 각도 (기본 -10)")
    args = parser.parse_args()

    log("\n" + "=" * 60)
    log(f"엄지 진단 [{args.step}] {datetime.now():%Y-%m-%d %H:%M:%S}")
    log("=" * 60)

    hand = open_hand()
    log(f"연결됨: {hand.ser.port}")

    try:
        snapshot(hand, "기준선 (연결 직후)")

        if args.step == "baseline":
            log("\n  >> 위 전류값을 볼 것. TH-TEN 전류가 다른 채널보다 높다면")
            log("     모터가 엄지를 굽힌 채 버티고 있다는 객관적 증거다.")
            log("     전류가 전부 0 이면 서보 전원이 안 들어온 것이다 (USB 는 컨트롤러 전용).")
        elif args.step == "extend":
            step1_extend(hand)
        elif args.step == "release":
            step2_release(hand)
        elif args.step == "homing":
            step3_homing(hand)
        elif args.step == "trim":
            step4_trim(hand, args.degrees)
    finally:
        # release 단계는 토크가 풀린 상태를 유지해야 관찰이 가능하다.
        if args.step != "release":
            log("\n토크 해제하고 종료.")
            try:
                hand.ctrl_torque([0] * N_CHANNELS)
            except Exception as e:
                log(f"  토크 해제 실패: {e}")
        hand.close()
        log(f"로그 저장: {LOG_PATH}")


if __name__ == "__main__":
    sys.exit(main())
