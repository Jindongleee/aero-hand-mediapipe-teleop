# 따라하기 가이드 — Aero Hand 텔레옵 + Behavior Cloning

> Aero Hand Open 실물 하나만 있으면 **처음부터 끝까지** 따라 할 수 있는 실습 가이드입니다.
> 흐름: **설치 → SDK GUI로 손 준비(호밍) → MediaPipe만으로 확인 → 실물 텔레옵 → BC 수집·학습·재생.**
> 프로젝트 배경·설계 원리는 [README](README.md) 참고.

---

## ⚠️ 먼저 — 안전 수칙 (손이 한 대뿐일 때 필독)

힘줄(텐던) 구동 로봇손은 **한 번 손상되면 복구가 안 됩니다.** 아래는 협상 불가입니다.

1. **첫 구동은 항상 `--slew 0.05`** — 프레임 간 변화율 제한. 서보가 급격히 안 튐.
2. **전류가 유일한 지표.** 위치는 못 믿습니다. 화면/`Get All`로 전류를 보세요.
   - `~10 mA` 무부하(정상) · `~200 mA` 물림 · **`1000 mA+` 지속 = 스톨(하드스톱 처박음) → 즉시 멈춤**
3. **스톨 감지 시 즉시 `SPACE`(토크 해제)** 또는 전원 OFF.
4. **엄지 ch0·ch1을 "둘 다 낮게" 동시에 몰지 마세요** → 엄지 커플링 바인딩으로 수 A 스톨. (하나가 높으면 다른 하나는 자유롭게 내려도 됨)
5. **발열 70 °C 근처면 식히세요** (펌웨어가 열보호로 토크를 줄여 "힘 빠진 듯" 됩니다).
6. **막힌 기구를 재호밍하지 마세요** — 막힌 상태를 영점으로 굳혀버립니다.
7. GUI에서 **Zero All 금지** (전 채널을 하한으로 몰아 4번 바인딩 유발).

---

## 0. 준비물

- **Aero Hand Open** (조립 완료, 조립가이드는 [TetherIA 레포](https://github.com/TetherIA/aero-hand-open))
- **USB 웹캠** 1개
- **리눅스 PC** (이 가이드 기준), **Python 3.12+**
- 손의 **메인 전원**(서보용)과 **USB**(보드 통신용) 두 케이블

---

## 1. 설치

```bash
# 가상환경 + 의존성
uv venv --python 3.12 .venv
uv pip install -r requirements.txt

# Aero Hand SDK
uv pip install "aero-open-sdk @ git+https://github.com/TetherIA/aero-hand-open.git#subdirectory=sdk"

# MediaPipe 손 모델 (별도 다운로드)
curl -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

포트는 SDK가 `/dev/serial/by-id/` 에서 자동 탐지합니다. 리눅스에서 보통 아무 설정 필요 없습니다.

---

## 2. 손 연결 + SDK GUI + 호밍 (제일 중요)

### 2-1. 연결

메인 전원 ON → USB 연결. 포트 확인:

```bash
ls /dev/serial/by-id/    # usb-Espressif_USB_JTAG_serial_... 가 보이면 OK
```

### 2-2. GUI 실행

```bash
.venv/bin/aero-open-gui
```

**Connect** 클릭 → **Get All** 로 위치·전류 확인.

### 2-3. 호밍 — 시작 위치가 결과를 좌우한다 ⭐

호밍은 각 채널을 **하드스톱까지 당겨 접촉점을 영점으로 등록**합니다(전류 임계 도달까지 travel). 그래서 **시작 위치가 중요**합니다.

- ❌ 엄지 채널(ch0·ch1)이 **호밍이 접촉하러 가는 끝에 이미 붙어 있으면** → 즉시 임계 도달 → **엉뚱한 영점** → 이후 그 채널이 스톨.
- ✅ **호밍 전에 ch0·ch1 슬라이더를 1.0(반대편 끝)에 두고** Homing → 서보가 **전 범위를 제대로 travel**하며 진짜 접촉을 찾음 → **올바른 영점**.

**절차:**
1. GUI 슬라이더로 **ch0 = 1.0, ch1 = 1.0** 으로 이동
2. **Homing** 클릭
3. 손이 **수십 초** 움직이며 호밍하면 정상. **몇 초 만에 끝나면** 시작 위치가 잘못됐다는 신호 → 다시.

> **판정:** 호밍 후 손가락이 쭉 펴지고, 엄지가 전류 낮게(수십 mA) 편안히 쉬면 성공.

### 2-4. 완전한 전원-투입 호밍이 필요할 때

자동 호밍은 **전원 투입(POWERON)** 때만 돕니다. 이 보드의 **ESP는 USB로 전원**을 받는 경우가 있어 — **메인 전원만 꺼도 ESP가 안 죽어** 자동 호밍이 안 됩니다.

- **메인 전원 + USB 둘 다 OFF** → 포트가 사라졌는지 확인(`ls /dev/ttyACM*` 없음) → 다시 켜기.
- 이때 **호밍 춤(손가락이 범위를 움직임)** 이 나와야 진짜 호밍이 된 것.

---

## 3. MediaPipe만 — 로봇 없이 (안전한 첫 단계)

로봇 없이 인식·매핑부터 확인합니다.

```bash
.venv/bin/python mp_test.py                             # 손 키포인트 21개 검출 확인
.venv/bin/python hand_mirror.py --no-robot              # 랜드마크→7채널 값만 화면에 표시
```

---

## 4. 캘리브레이션 + 실물 텔레옵

### 4-1. 캘리브레이션 (로봇 불필요)

채널별 가동범위를 **직접 움직여서** 잡습니다. 타이머로 안 끊고, 막대가 다 찰 때까지 다양하게 움직이세요.

```bash
.venv/bin/python hand_mirror.py --no-robot --calibrate  # → calibration.json 생성
```

### 4-2. 실물 미러링

```bash
.venv/bin/python hand_mirror.py --slew 0.05             # 첫 구동은 반드시 0.05
```

- `q` 종료 · `SPACE` 토크 토글
- 잘 되면 `--slew 0.15` 로 반응 빠르게.

---

## 5. Behavior Cloning — 수집 → 학습 → 재생

BC는 **(관측 63 → 행동)** 시연을 모아 신경망이 따라하게 학습하는 것입니다. 수집·학습은 **로봇 없이** 합니다.

### (A) 연속 회귀 — 63차원 → 7채널 관절값

```bash
.venv/bin/python collect_teleop.py                      # 시연 수집 → data/teleop_demos.npz
.venv/bin/python train_teleop_bc.py                     # 학습 → teleop_policy.pt
.venv/bin/python teleop_bc_play.py --no-robot --compare # 수식 vs 정책 비교 (로봇 없이)
.venv/bin/python teleop_bc_play.py --slew 0.05          # 학습된 정책으로 실물 미러링
```

### (B) 가위바위보 — 63차원 → 3클래스 분류

```bash
.venv/bin/python collect_rps.py                         # 포즈별 수집 → data/rps_demos.npz
.venv/bin/python train_rps.py                           # 학습 → rps_policy.pt
.venv/bin/python rps_play.py --no-robot                 # 정책 확인
.venv/bin/python rps_play.py --slew 0.05                # 실물 시연
```

전체 실행 순서 한눈에:

```
설치 → (2)GUI 호밍 → (3)mp_test → (4)calibrate → hand_mirror  ← 여기까지 "미디어파이프 텔레옵"
                                                    │
                                                    ├─ collect_teleop → train_teleop_bc → teleop_bc_play   (연속 BC)
                                                    └─ collect_rps    → train_rps       → rps_play          (가위바위보 BC)
```

---

## 6. 트러블슈팅 (이틀간 실제로 겪은 것들)

| 증상 | 원인 | 해결 |
|---|---|---|
| **엄지 한 채널이 스톨(수 A)/안 움직임** | 호밍 영점이 스톨 극단에 잡힘 | **2-3 호밍 절차**(ch0·ch1을 1.0에 두고 Homing) 재실행 |
| ch0·ch1 같이 내리면 스톨 | 엄지 **커플링 바인딩**(액추에이터 독립 아님) | 한쪽을 높게 유지. 손별 안전범위는 `config.THUMB_SAFE_RANGE` |
| **엄지가 사람 손과 반대로** 움직임 | 채널 부호 규약이 개체·펌웨어(좌/우)마다 다름 | `config.THUMB_SAFE_RANGE` 에서 해당 채널 `(lo,hi)` 를 뒤집기(`lo>hi`) |
| **엄지 풀·릴리즈가 안 돌아옴** | 단방향 텐던(당기기만, 스프링 없음) | 정상. 미러링은 당기는 방향이 주. |
| 벌림 방향이 손마다 반대 | **좌/우 펌웨어**(`LEFT_HAND`/`RIGHT_HAND`) 불일치 | 손 바꿨으면 올바른 `.bin` 플래시([레포 `firmware/main/bin/`](https://github.com/TetherIA/aero-hand-open)) |
| 호밍이 몇 초 만에 끝남 | 시작 위치가 접촉단에 붙어있음 | 반대편(1.0)에서 시작 |
| SDK 통신 불안정(ACK 타임아웃) | 부팅 직후 USB-CDC 미동기 | **USB만 재삽입**(호밍 영점 유지) |
| 카메라 안 열림 | 다른 프로세스가 점유 | 남은 python 프로세스 종료 후 재시도 |
| 엄지가 "힘 빠진 듯" | 발열 70 °C 열보호 | **식히기**(전원 OFF 몇 분) |

### 손별 엄지 안전범위 (선택)

엄지 벌림/굽힘은 개체·펌웨어마다 편안한 방향과 범위가 다릅니다. 스톨 없이 쓰려면 `config.py`의 `THUMB_SAFE_RANGE`에 **본인 보드 MAC → 안전범위**를 등록하세요. (측정: GUI에서 ch1을 편안한 값에 두고 ch0을 천천히 내리며 **전류가 치솟는 지점**을 찾음 = 그 위가 안전.)

```python
# config.py — 보드 MAC 으로 좌/우 자동 판별 후 해당 범위로 재매핑
THUMB_SAFE_RANGE = {
    "right": (0.20, 1.00, 1.00, 0.30),   # (ch0_lo, ch0_hi, ch1_lo, ch1_hi). lo>hi = 방향 반전
    # "left": 본인 왼손 실측값으로 추가
}
```

> ⭐ **2-3 호밍 절차를 제대로 지키면** 엄지가 정상 영점을 잡아, 이 소프트 클램프에 덜 의존하게 됩니다. 호밍이 1순위, 안전범위는 보강.

---

## 7. 채널 참고

| ch | 이름 | 동작 |
|---|---|---|
| 0 | `thumb_cmc_abd` | 엄지가 검지에서 멀어짐/모임 (벌림) |
| 1 | `thumb_cmc_flex` | 엄지가 손바닥을 가로지름 (굽힘) |
| 2 | `thumb_tendon` | 엄지 자체 말림 (MCP+IP 동시) |
| 3~6 | 검지·중지·약지·소지 | 각 손가락 굽힘 |

엄지는 관절 4개를 액추에이터 3개가 당기는 **언더액추에이티드** 구조라, 세 채널이 서로 완전히 독립은 아닙니다(위 커플링 주의).
