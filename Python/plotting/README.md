# 그림 생성 스크립트

학습 로그와 평가 결과 파일을 읽어서 논문 그림을 만드는 스크립트 모음이다.
데이터를 생성하거나 수정하지 않는다. 읽기 전용이다.

## 입력

기본 경로는 이 폴더의 `logs/` 이며, 환경변수 `VESSEL_LOG_DIR` 로 바꿀 수 있다.
필요한 파일은 세 종류다.

| 파일 | 내용 |
|---|---|
| `<run>.log` | 학습 중 주기적으로 찍힌 진행 기록. `dec=..M \| ep=N len~L \| goal=..% vColl=..% oColl=..% TO=..% \| R=..` |
| `<run>.csv` | PPO 업데이트마다 기록된 `step,raw_reward,ema_reward` |
| `metrics_v2.txt` | 학습 종료 후 정책을 고정하고 다시 평가한 결과 (`eval_ckpt.py` 출력) |
| `metrics_sit.txt` | 위와 같되 COLREGs 준수율을 조우 상황별로 분해한 것 |

로그 파일 자체는 이 저장소에 포함하지 않았다.

## 실행

```
python regenerate_all.py     # 모든 그림 생성
python build_final.py        # 논문용 최종 폴더 구성
```

## 그림에 들어가는 값이 어떻게 계산되는가

로그를 그대로 읽되, 그리는 과정에서 다음 처리를 한다. 전부 스크립트 상단에
상수로 노출되어 있으며 숨긴 처리는 없다.

**학습곡선의 세로축** (`make_ablation_rewards.py`)

```
값 = W_GOAL x 도착률 - W_COLL x 충돌률 - W_TO x 타임아웃률
W_GOAL, W_COLL, W_TO = 1.5, 6.0, 0.5
SCALE = 3.0   (표시 배율)
```

충돌 가중치를 도착보다 크게 둔 것은 해상에서 충돌 비용이 지연 비용을
압도하기 때문이다. 이 값을 3에서 15까지 바꿔도 조건 간 순서는 바뀌지 않는다.
모든 조건에 같은 식을 적용한다.

**여러 학습 실행을 합치는 방법**

같은 조건을 서로 다른 난수로 여러 번 학습한 결과를 합칠 때, 배열 순서가 아니라
학습 스텝을 기준으로 맞춘다 (`BIN = 0.33M` 구간으로 묶음). 실행마다 로그 줄
수가 다를 수 있어서 순서로 더하면 서로 다른 시점을 더하게 된다.

비율을 단순 평균하지 않고 종료 에피소드 수로 가중해서 합산한다. 기록 구간마다
끝난 에피소드 수가 수십에서 수백까지 차이나기 때문이다.

**평활** (`ROLL = 4`) 인접 4구간을 묶어 계산한다. 원 신호의 변동은 남는다.

**표시 구간** (`X0 = 4.0`, `X1`) 가로축 시작을 4M으로 둔다. 그 이전은 학습되지
않은 정책이 크게 요동하는 구간이라 이후 구간의 세로축 해상도를 잡아먹는다.
끝 지점은 `VESSEL_FIG_XMAX` 로 지정하며 기본값은 전체 구간이다.

**막대 그림** (`build_final.py`) 은 `metrics_v2.txt` / `metrics_sit.txt` 의 값을
그대로 쓴다. 여러 실행의 평균과 표본표준편차만 계산한다.

## 주의

- 학습 중 기록은 에피소드가 몰려서 끝나는 시점에 따라 흔들린다. 최종 성능
  판단에는 학습 종료 후 평가 결과(`metrics_v2.txt`)를 쓴다.
- 실행이 하나뿐인 조건은 표준편차가 0으로 표시된다. 오차가 없다는 뜻이 아니다.

## 학습곡선 — trunk/분기 배치 (2026-09-30, 스펙 `2026-09-30-reward-v3-decode-sweep-design.md` §7 '목표 2(그림)')

| 스크립트 | 입력 (`--dir`, 없으면 `--also_dir` 순서로 찾음) | 출력 |
|---|---|---|
| `plot_reward_v3.py` | `<prefix><arm>_s<seed>.csv` (`step,raw_reward,ema_reward`, update 65,536결정당 1행; 갈래 파일은 trunk 행 1–138 뒤에 자기 행) + `<prefix>trunk_d<dim>_s<seed>.csv` | 결정당 보상: raw(옅게) + **중심 이동평균 20 update**, 시드별 행(`--layout mean` 은 시드 평균 + 얇은 시드별 선), 왼쪽 0–16.06M(y 범위에 0·첫 값 포함, 분기 점선·첫 update 주석), 오른쪽 분기 뒤 확대(끝값 직접 라벨). PNG·PDF 둘 다 |
| `plot_ep_return.py` | `<prefix><arm>_s<seed>_ep.csv` (`step,n_ep,ep_return_mean,goal,vColl,oColl,TO`; 빈칸·nan·n_ep 0 행은 건너뜀, 비율은 학습기가 % 로 씀 = `--rate_unit pct` 기본) | 같은 배치 + goal(실선)/vColl(점선) MA % 패널. 폴더에 `*_ep.csv` 가 하나도 없으면 알리고 rc 0 으로 끝남(s_ 등 옛 배치) |

```
C:/Users/OSH/anaconda3/python.exe plot_reward_v3.py --dir <out dir> --prefix t_ --arms off,a6,a8,a2,a4,offb --seeds 43,44,45 --out <file>.png
C:/Users/OSH/anaconda3/python.exe plot_ep_return.py --dir <out dir> --prefix t_ --arms off,a6,a8,a2,a4,offb --seeds 43,44,45 --out <file>.png
```

- matplotlib 은 base anaconda python 에만 있음(mltest 에 없음). 한글 = Malgun Gothic(없으면 대체 폰트).
- 이동평균 = 중심 창 20 update, 양 끝은 창 축소(pandas `rolling(20, center=True, min_periods=1)` 과 같음). `ema_reward` 열(α 0.02, ~50 update 지연)은 안 씀.
- stdout 표(팔×시드 + 시드 평균): 분기 뒤(step > 9,043,968) MA 평균 · arm MA ≥ off MA 인 update 비율(두 run 이 겹치는 update 만) · 끝 차이 arm−off(마지막 공통 update) · `offb−off` 끝 차이 = 재분기 잡음 N. 스펙 §7 규칙(비율 ≥ 90 % & 끝 차이 > |N|, 3/3)은 집계만 찍고 판정은 저자가 함.
- 같은 시드 팔들의 분기 전 구간이 trunk 와 다르면 `★ … §8-1 분기 규약 확인` 경고를 찍음(그림은 그대로 그림).
- 팔 색은 실행 이름에 고정(off 진회색, offb 연회색 점선, a6 파랑, a8 주황, a2 청록, a4 보라; z6/z12 = 파랑/주황). 모르는 팔은 남는 색을 `--arms` 순서로 받음.
