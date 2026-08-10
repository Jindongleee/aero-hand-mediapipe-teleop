# Aero Hand Open — MediaPipe 텔레오퍼레이션 + Behavior Cloning

TetherIA의 오픈소스 힘줄 구동 로봇손 [Aero Hand Open](https://github.com/TetherIA/aero-hand-open)을, 웹캠 하나로 실시간 조작하는 텔레오퍼레이션 파이프라인입니다. 여기에 수집한 시연 데이터로 모방학습(Behavior Cloning) 정책을 학습시키는 경로를 더했습니다.

## 왜 만들었나

ETRI 겨울 연구연수생 시절 매니퓰레이터(Franka Panda)를 대상으로 텔레오퍼레이션 데이터 수집 파이프라인을 구축했습니다. 인턴이 끝난 뒤에도 "사람의 시연을 로봇이 이해할 수 있는 신호로 어떻게 바꿀 것인가"라는 문제의식을 놓지 않고, 완전히 다른 자유도의 하드웨어(다지형 로봇손)에 같은 원리를 스스로 옮겨본 개인 프로젝트입니다.

## 데모

<img src="docs/demo_1.jpg" width="420"> <img src="docs/demo_2.jpg" width="420">

카메라로 손 모양을 인식하는 동시에, 실제 Aero Hand가 그 모양을 그대로 따라 움직입니다.

## 구성

두 갈래가 있습니다. 위쪽은 규칙 기반 텔레옵, 아래쪽은 그 위에 얹은 학습 경로입니다.

```
카메라 ─ MediaPipe 21 랜드마크 ─┬─ mapping.py (해석적 수식) ─→ 7채널 ─→ 손
                                │
                                └─ landmarks.py (63차원 특징) ─┬─ MLP → 3-class → 포즈 룩업 ─→ 손
                                                               └─ MLP → 7채널 회귀 ────────→ 손
```

| 파일 | 단계 | 내용 | 로봇 |
|---|---|---|---|
| `mp_test.py` | 1 | 손 키포인트 21개 검출 확인 | 불필요 |
| `mp_bend.py` | 2 | 손가락 굽힘 비율(0~1) 계산 | 불필요 |
| `hand_mirror.py` | 3 | 실시간 미러링 (메인) | 필요 |
| `collect_rps.py` | 4 | 가위바위보 데모 수집 | 불필요 |
| `train_rps.py` | 5 | 3-class BC 학습 | 불필요 |
| `rps_play.py` | 6 | 학습된 정책으로 가위바위보 | 시연만 |
| `collect_teleop.py` | 7 | 7채널 연속 명령 수집 | 불필요 |
| `train_teleop_bc.py` | 8 | 연속 회귀 BC 학습 | 불필요 |
| `teleop_bc_play.py` | 9 | 정책이 직접 관절 구동 | 시연만 |
| `mapping.py` | — | 랜드마크 → 7채널 해석적 매핑 + 캘리브레이션 |
| `landmarks.py` | — | 랜드마크 → 63차원 정규자세 특징 |
| `poses.py` | — | 사전 정의 포즈, 정규화값 → actuation 변환 |
| `config.py` | — | 포트 해석, 채널 정의, 안전 파라미터 |
| `thumb_diag.py` | — | 엄지 이상 단계별 진단 (위험도 순) |
| `verify_thumb.py` | — | 엄지 벌림·굽힘 간섭 실측 |
| `test_poses.py` | — | 포즈 순환 테스트 |

## 좌표계 — 화면 좌표를 쓰지 않는 이유

MediaPipe는 두 가지 랜드마크를 줍니다. 초기 버전은 화면 정규화 좌표(`hand_landmarks`)를 썼는데, 여기엔 구조적 문제가 있습니다.

- **x는 프레임 가로폭, y는 세로폭 기준**입니다. 16:9 카메라에서 x가 1.78배 눌리므로, 두 점 사이 거리나 각도를 계산하는 순간 이미 왜곡되어 있습니다.
- **z는 또 다른 스케일**입니다. xyz를 섞은 3D 연산이 원리적으로 성립하지 않습니다.

그래서 전부 `hand_world_landmarks`(미터 단위 실제 3D)로 옮겼습니다.

그 위에 **정규 자세(canonical pose)로 정렬**합니다.

```
원점   = 손목
y축    = 손목 → 중지 MCP
z축    = 손바닥 법선 (검지 MCP × 소지 MCP)
x축    = y × z
스케일 = 손목→중지 MCP 거리로 나눔
```

손의 위치·방향·크기가 제거되고 "손모양"만 남습니다. 평행이동·회전·스케일 변환에 대해 **최대 오차 1e-7**(float32 노이즈 수준)로 불변인 것을 확인했습니다. BC가 포즈당 수백 프레임으로도 버티는 이유입니다.

## 엄지 매핑 — 무엇이 틀렸고 어떻게 고쳤나

엄지는 나머지 네 손가락과 관절 구조가 달라 별도 처리가 필요한데, 초기 구현에 네 가지 결함이 겹쳐 있었습니다.

**① 엄지에 4손가락 공식을 그대로 적용**
`bend_ratio(pts, 2, 3, 4)` — MCP-IP-TIP 코사인. 그런데 엄지는 이 세 점이 굽혀도 거의 일직선을 유지합니다. 각도 변화폭이 작아 값이 좁은 대역에 갇히고, 로봇 엄지가 거의 움직이지 않습니다.

**② 독립 자유도 두 개를 하나로 낭비**
`ch1`과 `ch2`에 같은 `t_bend`를 스케일만 바꿔 넣어 상관계수가 1.0이었습니다.

**③ 벌림이 굽힘과 간섭**
벌림을 엄지 **TIP**과 소지 MCP 사이 거리로 쟀습니다. 벌리지 않고 **굽히기만 해도** TIP이 손바닥으로 들어오면서 벌림값이 함께 떨어집니다. 주먹을 쥐면 엄지 벌림이 제멋대로 움직였습니다.

**④ 스케일 기준이 불안정**
정규화 기준(`손목→중지 MCP`)이 손목 회전에 따라 투영 길이가 변하고, 벌림 범위가 상수 `0.55/1.40`으로 하드코딩되어 있었습니다.

### 수정

`mapping.py`에서 CMC→MCP 방향을 손바닥 평면 기준으로 분해합니다.

```
평면 안쪽 각도   → ch0 벌림   (엄지가 검지에서 멀어지는 정도)
평면 바깥 각도   → ch1 굽힘   (엄지가 손바닥을 가로지르는 정도)
MCP·IP 관절각 평균 → ch2 텐던 (엄지 자체 말림)
```

ch2는 손목 쪽 두 신호와 다른 관절을 보므로 ③이 해소됩니다. 범위는 하드코딩 대신 **실행 시 캘리브레이션**으로 잡습니다.

### 검증

엄지 벌림 자세를 고정한 채 **말단만 말면서** 288프레임 측정:

| 지표 | 정지 시(노이즈) | 말 때 | 노이즈 대비 |
|---|---|---|---|
| 기존 벌림 (std) | 0.022 | 0.176 | **7.9배** |
| 새 벌림 ch0 (std) | 0.017 | 0.068 | **4.0배** |
| 새 텐던 ch2 (std) | 0.0015 | 0.031 | 20.9배 |

잔여 4배는 공식 결함이 아닙니다. `compute_raw`의 ch0는 랜드마크 1(CMC)·2(MCP)만 읽고 엄지 끝(3, 4)은 계산에 들어가지 않습니다 — 두 점을 고정한 합성 테스트에서는 변동이 정확히 `0.000000`입니다. 실제 손에서 남는 것은 **MediaPipe가 추정한 엄지 뿌리 관절이 실제로 움직인 것**이고(엄지 끝을 말면 CMC/MCP가 따라 움직임), 그건 로봇이 따라가는 게 맞습니다. 기존 공식은 여기에 끝마디 위치까지 더해 7.9배로 부풀렸습니다.

## 캘리브레이션

`hand_mirror.py --calibrate`로 채널별 가동범위를 관측합니다. 타이머로 끊지 않고 **사용자가 끝낼 때까지** 돌아가며, 채널별 확보 폭을 실시간 막대로 보여줍니다.

고정 시간(8초)으로 했을 때 손가락 굽힘 최대가 0.13에 그쳤습니다 — `(1-cos)/2` 기준 거의 펴진 상태로, 주먹이 범위에 들어오지 못했습니다. 이 좁은 범위를 0~1로 확대하면 **손이 조금만 움직여도 로봇이 전 범위를 휘두르는** 위험한 매핑이 됩니다.

## Behavior Cloning

### (A) 가위바위보 — 3-class

```
obs    : 63차원 정규자세 랜드마크
action : rock / scissors / paper
model  : MLP 63 → 64 → 3, CrossEntropy
```

예측 클래스가 `poses.py`의 고정 포즈를 인덱싱합니다. 즉 **관절 명령 자체는 학습이 아니라 룩업**에서 나오므로, 엄밀히는 "제스처 분류기 + 포즈 재생"에 가깝습니다.

수집 1471 샘플 → **검증 정확도 1.000**, 혼동행렬 완전 대각. 라이브에서 17회 포즈 전환 전부 정상(확률 0.90~1.00). 다만 연속 프레임은 서로 거의 같아서 무작위 분할 검증셋에 학습셋의 이웃 프레임이 들어갑니다 — 100%는 낙관적 수치이고 라이브가 진짜 시험입니다.

**전환 프레임 문제**: 첫 학습에서 96.8%가 나왔고 rock→paper 오분류 10건이 있었습니다. 가장 다른 두 포즈인데 이상해서 인덱스를 추적하니 **전부 rock 블록의 앞 28프레임**에 몰려 있었습니다(블록은 0~525). 녹화 키를 누른 직후, 아직 주먹을 다 쥐지 않은 구간이었습니다. 라벨이 틀린 것이지 예측이 틀린 게 아니었습니다. `collect_rps.py`에 settle 구간을 넣고 기존 데이터는 `trim_transitions.py`로 정리했습니다.

### (B) 연속 회귀 — 7채널

```
obs    : 63차원 (동일)
action : 7채널 연속값 0~1
model  : MLP 63 → 128 → 128 → 7, sigmoid, MSE
expert : mapping.py 의 해석적 매핑
```

이쪽은 정책이 **관절 명령을 직접** 냅니다. 룩업이 없습니다.

expert가 해석적 수식이므로 **신경망은 손으로 짠 함수를 증류(distill)합니다.** BC 골격(관측 → expert 행동의 지도학습)은 지키지만, "사람 시연에서 새 기술을 배웠다"는 아닙니다. 평가는 accuracy가 아니라 **채널별 MAE**로 봅니다 — 전체 MSE 하나만 보면 어느 채널이 망가졌는지 안 보이고, 실물에서는 그 손가락만 안 따라오는 형태로 드러납니다.

학습은 마지막 epoch가 아니라 **검증 최고 시점**을 저장합니다. 연속 프레임이 near-duplicate라 학습 손실은 계속 내려가는데 검증 손실은 도로 올라가는 구간이 생깁니다.

> 상태: 파이프라인 작성 및 합성 데이터 검증 완료. 실제 데이터 학습은 미실시.

## 안전 장치

힘줄 구동 하드웨어는 급격한 신호에 취약하고, 손이 한 대뿐이면 텐던 손상은 복구가 안 됩니다.

- **EMA 필터** — 프레임 간 노이즈 완화
- **Slew limit** — 프레임 간 변화율 제한. 첫 구동은 `SLEW_LIMIT_SAFE = 0.05`
- **SPACE 토크 해제** — 즉시 전 서보 토크 차단
- **ch0 하한 클램프** — 실측 기반. ch0을 0.02에서 유지하면 모터가 −133 mA를 계속 뽑으며 56°C까지 올랐고, 0.05 이상에서는 2~14 mA로 떨어졌습니다. 미러링은 캘리브레이션 전 범위를 쓰므로 ch0이 반복적으로 0까지 내려갑니다.

BC 정책 배포에는 두 가지가 더 붙습니다.

- **확률 임계값 + 디바운스**(3-class) — 분류기는 매 프레임 답을 내지만, 손이 포즈 사이를 지나는 프레임의 정답은 세 클래스 중 어느 것도 아닙니다. 없으면 손이 떨립니다.
- 연속 회귀 쪽은 출력 떨림이 그대로 관절로 가므로 EMA·slew가 더 중요합니다.

## 실행

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt

# MediaPipe 모델 (별도 다운로드)
curl -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

```bash
# 텔레옵
.venv/bin/python mp_test.py                              # 손 검출 확인
.venv/bin/python hand_mirror.py --no-robot --calibrate   # 캘리브레이션 (로봇 불필요)
.venv/bin/python hand_mirror.py --slew 0.05              # 실물 미러링, 첫 구동

# 가위바위보 BC
.venv/bin/python collect_rps.py                          # 수집 (로봇 불필요)
.venv/bin/python train_rps.py                            # 학습
.venv/bin/python rps_play.py --no-robot                  # 정책 확인
.venv/bin/python rps_play.py --slew 0.05                 # 실물 시연

# 연속 회귀 BC
.venv/bin/python collect_teleop.py
.venv/bin/python train_teleop_bc.py
.venv/bin/python teleop_bc_play.py --no-robot --compare  # 수식 vs 정책 비교
```

`q` 종료, `SPACE` 토크 토글.

포트는 SDK의 리눅스 자동 탐지(`/dev/serial/by-id/`)에 위임합니다. 필요하면 `AERO_HAND_PORT` 환경변수로 지정합니다.

## 하드웨어 / 의존성

- [TetherIA Aero Hand Open](https://github.com/TetherIA/aero-hand-open) — 오픈소스 힘줄 구동 로봇손
  - SDK: `uv pip install "aero-open-sdk @ git+https://github.com/TetherIA/aero-hand-open.git#subdirectory=sdk"`
  - 조립 가이드(`hardware/Gen1_OPEN_Assembly_Guide.pdf`)에 엄지 케이블 경로와 링키지 조립이 있습니다
- MediaPipe Hand Landmarker 모델 — [공식 배포처](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- Python 3.12+, PyTorch (CPU로 충분)

### 액추에이터 채널

엄지는 관절이 4개(`cmc_abd, cmc_flex, mcp, ip`)인데 액추에이터는 3개입니다. `ch2`(thumb_tendon)가 MCP와 IP를 함께 당기는 언더액추에이티드 구조라, 엄지 말단 굽힘은 ch2가 지배합니다.

| ch | 이름 | 구동 방식 | 동작 |
|---|---|---|---|
| 0 | `thumb_cmc_abd` | 링키지 바 | 엄지가 검지에서 멀어짐/모임 |
| 1 | `thumb_cmc_flex` | CMC Flex Cable | 엄지가 손바닥을 가로지름 |
| 2 | `thumb_tendon` | Thumb Pull Cable | 엄지 자체 말림 (MCP+IP) |
| 3~6 | 각 손가락 | Pull Cable | 굽힘 |

## 다음 계획

- 실물 가위바위보 시연 녹화
- 연속 회귀 BC를 실제 데이터로 학습하고 해석적 매핑과 비교
- 엄지 3채널만 사람이 직접 조작해 라벨을 만들면, "해석적으로 짜기 어려운 매핑을 데이터로 학습"이라는 더 강한 구성이 됩니다
