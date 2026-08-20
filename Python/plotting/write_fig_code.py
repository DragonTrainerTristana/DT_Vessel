"""99_FINAL 각 그림 폴더에 재현 문서(HOWTO.md)와 그 그림을 만든 코드를 넣는다.

목적: 그림 한 장만 떼어 봐도 "어떤 학습에서 나왔고 무슨 명령으로 쟀고 어느 코드가 그렸는지"를
알 수 있게 하는 것. 전체 파이프라인은 99_FINAL/CODE/ 에 한 벌만 두고, 각 폴더에는
그 그림이 직접 쓴 스크립트만 복사한다(networks.py 같은 큰 공용 파일은 CODE/ 한 곳에만).

사용: python write_fig_code.py
"""
import os, shutil

FIG = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(FIG, '99_FINAL')
PY = os.path.abspath(os.path.join(FIG, '..', 'Assets', 'Scripts', 'Python'))

# 전 그림 공통 학습 설정. 팔마다 바뀌는 값은 REPL 로 치환해 팔별 실제 명령을 찍는다
# (예: Fig3의 최근접-1 팔에 --max_partners 4 가 찍히면 틀린 문서가 된다).
ENVLINE = """VESSEL_MSG_DIM=6 VESSEL_USE_COMM=1 VESSEL_RADAR_RANGE=56 \\
VESSEL_THREAT_COEF=0.5 VESSEL_SIM_COLREGS_COEF=0.45 \\
VESSEL_POS_GROUND=1 VESSEL_USE_ATTENTION=0"""
COMMON = ENVLINE + """ \\
python vessel_gym_train.py --arm ON --comm_on_at 9000000 --max_partners 4 \\
    --envs 128 --vessels 16 --rollout 32 --steps 16000000 --ring 0.7 --seed <SEED>"""


def apply(text, repl):
    for a, b in repl:
        text = text.replace(a, b)
    return text

# 구조별 env (2절 표와 동일)
ARCH = {
    'single': 'VESSEL_USE_MOE=0',
    'thin':   'VESSEL_USE_MOE=1 VESSEL_MOE_WIDTH=0.44',
    'full':   'VESSEL_USE_MOE=1 VESSEL_MOE_WIDTH=1.0',
    'shared': 'VESSEL_USE_MOE=1 VESSEL_MOE_WIDTH=1.0 VESSEL_MOE_SHARED=1',
}
ARCH_KO = {'single': '단일망 369K', 'thin': '분리·얇게 363K',
           'full': '분리·두껍게 1.83M', 'shared': '공유(제안) 512K'}

EVAL = (ENVLINE + " \\\n"
        "python eval_ckpt.py --ckpt <ckpt> --arm ON --max_partners 4 \\\n"
        "    --envs 96 --burnin 1500 --eval_decisions 3500")

# (폴더, 제목, 한줄설명, [ (팔이름, 구조, [실행들], 이 팔만 다른 것, 치환) ], 곡선코드항목)
# 치환은 공통 명령에 적용돼 그 팔의 *실제* 학습/평가 명령이 된다. 평가도 학습과 같은 env여야
# 하므로(체크포인트가 그 차원·구조로 저장됨) 같은 치환을 평가 명령에도 적용한다.
FIGS = [
    ('Fig1_Communication', '통신 유무', '통신 하나만 빼고 전부 같음.', [
        ('통신 없음', 'shared', ['qf_SE_OFF_s42', 'qf_SE_OFF_s43', 'qf_SE_OFF_s44'],
         '`--arm OFF` (메시지 입력이 항상 0)', [('--arm ON', '--arm OFF')]),
        ('통신 있음', 'shared', ['qd_MOE_SE_s42', 'qd_MOE_SE_s43', 'qd_MOE_SE_s44'],
         '`--arm ON`', []),
    ], 'ablation_reward_1_communication'),

    ('Fig2_MoE_Architecture', '신경망 구조', 'COLREGs 5갈래를 신경망 어디까지 적용하냐만 다름.', [
        ('단일망', 'single', ['q_MOE_SINGLE', 'qd_MOE_SINGLE_s43', 'qd_MOE_SINGLE_s44'],
         '전문가 없음', []),
        ('분리·얇게', 'thin', ['q_MOE_ISO'],
         '폭 0.44로 줄여 단일망과 파라미터 맞춤. **시드 1개**', []),
        ('분리·두껍게', 'full', ['base_comm', 'base_comm_s43', 'base_comm_s44'],
         '인식부까지 5벌 복제', []),
        ('공유(제안)', 'shared', ['qd_MOE_SE_s42', 'qd_MOE_SE_s43', 'qd_MOE_SE_s44'],
         '인식부 1벌 + 판단부만 5벌', []),
    ], 'ablation_reward_2_moe'),

    ('Fig3_Message_Aggregation', '메시지 집계', '받는 이웃 수만 다름.', [
        ('최근접 1척', 'shared', ['qf_SE_NEAR1_s42', 'qf_SE_NEAR1_s43', 'qf_SE_NEAR1_s44'],
         '`--max_partners 1`', [('--max_partners 4', '--max_partners 1')]),
        ('4척 집계', 'shared', ['qd_MOE_SE_s42', 'qd_MOE_SE_s43', 'qd_MOE_SE_s44'],
         '`--max_partners 4`', []),
    ], 'ablation_reward_3_aggregation'),

    ('Fig4_Message_Dimension', '메시지 차원', '`VESSEL_MSG_DIM`만 다름. **이 축만 제안 구조가 아님.**', [
        ('차원 2', 'full', ['v2_DIM2', 'q_DIM2_s43', 'q_DIM2_s44'],
         '`VESSEL_MSG_DIM=2`', [('VESSEL_MSG_DIM=6', 'VESSEL_MSG_DIM=2')]),
        ('차원 4', 'full', ['v2_DIM4', 'qe_DIM4_s43', 'qe_DIM4_s44'],
         '`VESSEL_MSG_DIM=4`', [('VESSEL_MSG_DIM=6', 'VESSEL_MSG_DIM=4')]),
        ('차원 6', 'full', ['base_comm', 'base_comm_s43', 'base_comm_s44'],
         '`VESSEL_MSG_DIM=6`', []),
        ('차원 8', 'full', ['v2_DIM8', 'qe_DIM8_s43', 'qe_DIM8_s44'],
         '`VESSEL_MSG_DIM=8`', [('VESSEL_MSG_DIM=6', 'VESSEL_MSG_DIM=8')]),
        ('차원 10', 'full', ['v2_DIM10', 'qh_DIM10_s43', 'qh_DIM10_s44'],
         '`VESSEL_MSG_DIM=10`', [('VESSEL_MSG_DIM=6', 'VESSEL_MSG_DIM=10')]),
        ('차원 12', 'full', ['v2_DIM12', 'q_DIM12_s43', 'q_DIM12_s44'],
         '`VESSEL_MSG_DIM=12`', [('VESSEL_MSG_DIM=6', 'VESSEL_MSG_DIM=12')]),
    ], 'ablation_reward_5_dimension'),

    ('Fig5_COLREGs_Term', '규정 준수 보상 항', '보상에서 규정 항 계수만 다름.', [
        ('항 없음', 'full', ['q_COLREGSOFF'],
         '`VESSEL_SIM_COLREGS_COEF=0`. **시드 1개**',
         [('VESSEL_SIM_COLREGS_COEF=0.45', 'VESSEL_SIM_COLREGS_COEF=0')]),
        ('항 있음', 'full', ['base_comm', 'base_comm_s43', 'base_comm_s44'],
         '`VESSEL_SIM_COLREGS_COEF=0.45`', []),
    ], 'ablation_reward_6_colregs'),

    ('Fig6_Comm_Timing', '통신 시작 시점', '메시지를 언제부터 흘렸는지만 다름. 둘 다 16M 내내 학습함.', [
        ('처음부터', 'shared', ['ql_SE_START_s42', 'ql_SE_START_s43', 'ql_SE_START_s44'],
         '`--comm_on_at 0`', [('--comm_on_at 9000000', '--comm_on_at 0')]),
        ('9M부터', 'shared', ['qd_MOE_SE_s42', 'qd_MOE_SE_s43', 'qd_MOE_SE_s44'],
         '`--comm_on_at 9000000`', []),
    ], 'ablation_reward_7_comm_timing'),
]

MIXED = [
    ('Fig7_Various_Agent', 'radar', '통신 장비가 아예 없는 배 (보내지도 받지도 못함)'),
    ('Fig8_Small_Boat', 'rx', '듣기만 하는 배 (받지만 못 보냄) — 통통배'),
]


def seeds_of(runs):
    return ', '.join(r.rsplit('_s', 1)[-1] if '_s' in r else '42' for r in runs)


def arms_table(arms):
    out = ['| 조건 | 구조 | 시드 | 실행 | 이 팔만 다른 것 |', '|---|---|---|---|---|']
    for name, arch, runs, diff, _repl in arms:
        out.append(f'| {name} | {ARCH_KO[arch]} | {len(runs)} | '
                   f'`{"`, `".join(runs)}` | {diff} |')
    return '\n'.join(out)


def cmd_block(arms, base):
    """팔별 실제 명령. 공통 명령에 그 팔의 치환과 구조 env를 얹는다."""
    out = []
    for name, arch, runs, diff, repl in arms:
        out.append(f'**{name}** (시드 {seeds_of(runs)})\n\n```\n'
                   f'{ARCH[arch]} \\\n{apply(base, repl)}\n```')
    return '\n\n'.join(out)


def write_howto(folder, title, note, arms, curve):
    p = os.path.join(DST, folder, 'HOWTO.md')
    body = f"""# {folder} — {title}

{note}

## 1. 무엇을 비교했나

{arms_table(arms)}

시드 42/43/44 씀. 실행 이름 끝 `_s43` = 시드 43, 접미사 없으면 42임.
**이름으로 구조를 판단하지 말 것** — 이름과 설정이 안 맞는 실행이 있어서 구조는
체크포인트를 직접 열어 확인한 값임.

## 2. 학습 명령 (팔별 실제 명령)

`<SEED>` 를 42/43/44 로 바꿔 시드 수만큼 돌림. 아래 적힌 것 말고는 전 그림 동일.

{cmd_block(arms, COMMON)}

## 3. 측정 (막대 그림 값)

학습 로그가 아님. 학습 끝나고 **정책을 고정한 뒤 다시 항해시켜** 잰 값임.
학습 중 로그는 에피소드가 한꺼번에 끝나는 타이밍 때문에 창마다 흔들려서 안 씀.

평가도 학습과 **같은 env** 로 돌려야 함 — 체크포인트가 그 차원·구조로 저장돼 있어서임.

{cmd_block(arms, EVAL)}

`--burnin 1500` = 집계 없이 먼저 돌려 위상 흩뜨리는 구간. 전부 같은 시점에 출발하면
종료도 몰려서 통계가 깨짐. 결과는 `_data/metrics_v2.txt` · `metrics_sit.txt` 에 한 줄씩 쌓임.

## 4. 그림 코드

| 그림 | 만든 코드 | 항목 |
|---|---|---|
| `0_reward_curve` | `make_ablation_rewards.py` | `figure('{curve}', ...)` |
| `1`~`7` 막대 | `build_final.py` | `TOPICS['{folder}']` |

곡선 y축은 로그의 `R`이 아님. 같은 로그의 결과 분포로 다시 계산한 값임:

```
보상 = 1.5 x 도착 - 6.0 x 충돌 - 0.5 x 시간초과      (창별 종료 에피소드 수로 가중)
```

로그 `R`은 매 결정마다 주는 유도 보상 평균이라 최종 결과와 상관이 거의 없음(실측 r ~ 0.02).
위 식은 전 조건에 똑같이 적용하고, 충돌 가중치를 3~15배로 바꿔도 조건 간 순서 안 바뀜.

## 5. 이 그림만 다시 만들기

```
cd Figures
python regenerate_all.py     # 곡선. VESSEL_FIG_XMAX=15000000 이 여기서 주입됨
python build_final.py        # 막대
```

`make_ablation_rewards.py` 를 맨손으로 돌리면 안 됨 — x축 상한이 안 걸려 16M까지 그려짐.

## 6. 이 폴더의 코드

| 파일 | 뭐 하는 것 |
|---|---|
| `code/eval_ckpt.py` | 3절 측정 |
| `code/make_ablation_rewards.py` | 곡선 |
| `code/build_final.py` | 막대 |

학습 코드와 시뮬레이터는 `../CODE/` 에 있음(파일이 서로 물려 있어 한 벌만 둠).
"""
    open(p, 'w', encoding='utf-8').write(body)


def write_mixed_howto(folder, mode, desc):
    p = os.path.join(DST, folder, 'HOWTO.md')
    body = f"""# {folder} — 혼합 함대 ({mode})

**학습을 다시 하지 않음. 평가만임.**

통신하며 학습을 마친 정책을 고정해 두고, 평가할 때만 16척 중 일부의 통신을 끊어
함대 성적을 잰다. 끊는 배 수를 2·4·6·8·10·12·14 로 바꿔가며 반복.

이 그림에서 통신 못 하는 배: **{desc}**

## 1. 쓴 정책

`qd_MOE_SE_s42`, `qd_MOE_SE_s43`, `qd_MOE_SE_s44` — 공유(제안) 구조, 통신 9M부터,
이웃 4척 집계로 16M 학습을 마친 체크포인트. Fig1의 "통신 있음" 팔과 같은 실행임.

## 2. 측정 명령

```
VESSEL_MSG_DIM=6 VESSEL_USE_MOE=1 VESSEL_MOE_SHARED=1 VESSEL_MOE_WIDTH=1.0 \\
VESSEL_USE_COMM=1 VESSEL_RADAR_RANGE=56 VESSEL_THREAT_COEF=0.5 \\
VESSEL_POS_GROUND=1 VESSEL_USE_ATTENTION=0 VESSEL_CKPT_DIR=<체크포인트 폴더> \\
python eval_mixed.py --ckpt qd_MOE_SE_s42.pt --mode {mode} --tag s42 \\
    --envs_per 14 --burnin 1200 --eval_decisions 4500 --csv mixed_fleet.csv
```

시드 43·44도 같은 명령으로 반복. 결과는 `_data/mixed_fleet.csv` 에 이어 쌓임.

구현 요점 두 가지:

- **비율 7가지를 한 번에 잼.** 환경마다 다른 비율을 심어 놓고 한 번만 돌리므로
  7번 따로 돌리는 것보다 빠르고 burn-in도 한 번만 냄.
- **통신 불가 선박은 스폰 링 위에 고르게 흩어 배치함.** 한쪽에 몰리면 그 구역만
  통신 공백이 되어 비교가 왜곡됨.

## 3. 그림 코드

`make_mixed_fleet.py` 의 `MODES['{mode}']` 항목.

막대 하나 = **함대 전체 값**임. 통신 되는 무리와 안 되는 무리를 따로 그리지 않고 합쳤음.
합칠 때 단순평균이 아니라 표본 수 가중임 — 무리마다 배 수가 다르고(예: 2척 vs 14척)
지표마다 분모가 다름:

| 지표 | 가중치 |
|---|---|
| 도착·충돌·시간초과·최근접거리 | 에피소드 수 |
| COLREGs 준수 | 판정된 조우 수 |
| 연료·조타·소요시간 | **도착한** 에피소드 수 |

오차막대는 시드 3개 간 표준편차임.

## 4. 다시 만들기

```
cd Figures
python build_final.py        # 먼저. 99_FINAL 아래 png/pdf를 전부 지우고 다시 만듦
python make_mixed_fleet.py   # 반드시 그 뒤
```

## 5. 이 폴더의 코드

| 파일 | 뭐 하는 것 |
|---|---|
| `code/eval_mixed.py` | 2절 측정 |
| `code/make_mixed_fleet.py` | 그림 |

시뮬레이터와 신경망은 `../CODE/` 에 있음.
"""
    open(p, 'w', encoding='utf-8').write(body)


def main():
    # 공용 코드 한 벌
    code = os.path.join(DST, 'CODE')
    os.makedirs(code, exist_ok=True)
    for f in ('vessel_gym_train.py', 'vessel_gym.py', 'networks.py', 'config.py',
              'eval_ckpt.py', 'eval_mixed.py'):
        src = os.path.join(PY, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(code, f))
    for f in ('make_ablation_rewards.py', 'build_final.py', 'make_mixed_fleet.py',
              'regenerate_all.py', 'RUNS.md'):
        src = os.path.join(FIG, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(code, f))
    print(f'CODE/: {len(os.listdir(code))}개')

    n = 0
    for folder, title, note, arms, curve in FIGS:
        out = os.path.join(DST, folder)
        os.makedirs(out, exist_ok=True)
        write_howto(folder, title, note, arms, curve)
        sub = os.path.join(out, 'code'); os.makedirs(sub, exist_ok=True)
        shutil.copy2(os.path.join(PY, 'eval_ckpt.py'), os.path.join(sub, 'eval_ckpt.py'))
        for f in ('make_ablation_rewards.py', 'build_final.py'):
            shutil.copy2(os.path.join(FIG, f), os.path.join(sub, f))
        n += 1
        print(f'  {folder}: HOWTO.md + code/ 3개')

    for folder, mode, desc in MIXED:
        out = os.path.join(DST, folder)
        os.makedirs(out, exist_ok=True)
        write_mixed_howto(folder, mode, desc)
        sub = os.path.join(out, 'code'); os.makedirs(sub, exist_ok=True)
        shutil.copy2(os.path.join(PY, 'eval_mixed.py'), os.path.join(sub, 'eval_mixed.py'))
        shutil.copy2(os.path.join(FIG, 'make_mixed_fleet.py'),
                     os.path.join(sub, 'make_mixed_fleet.py'))
        n += 1
        print(f'  {folder}: HOWTO.md + code/ 2개')
    print(f'그림 폴더 {n}개 처리 완료')


if __name__ == '__main__':
    main()
