# Windows 실행 가이드 — commgate Stage 1/2

작성 2026-07-03. 이 폴더는 Dropbox로 동기화되므로 Windows에서 git pull 불필요 —
아래 0번으로 동기화만 확인하고 진행한다.

---

## 0. 동기화 확인 (1분)

```powershell
cd <Dropbox>\Private_Paper_Project\0702_NewVessel\Assets\Scripts
git log --oneline -1
```

`5e074c9` (convergence_gate fix) 이상이면 최신. 아니면 Dropbox 동기화 대기.

## 1. Unity 빌드 (필수 — C# 변경 포함)

1. Unity로 프로젝트 열기 → 콘솔에 컴파일 에러 없는지 확인
2. 씬 확인 (1분): 스폰 포인트가 16개 이상이거나 SpawnZone 컴포넌트가 있는지.
   부족하면 배 수가 클램프되어 밀집 regime이 약해짐 (경고 로그로 표시됨)
3. Build → 프로젝트 루트 `Build\Vessel_MLAgent.exe` (다른 위치면 아래에서 `-Exe`로 지정)
4. 빌드는 **한 번만**. Thick/Light MoE 브랜치도 C#은 동일하므로 같은 exe 사용

## 2. Preflight (~20분, 밤샘 배치 전 안전판)

이 코드의 스모크 검증은 Mac에서 수행됨 — Windows 환경에서의 첫 실행 확인 절차.

```powershell
# (1) PPO 미러 검증 (전부 PASS여야 함)
python Python\verify\_verify_ppo_mirror.py

# (2) 20k 스텝 미니런 (OFF+ORACLE × seed 42 = 2런, ~15분)
powershell -ExecutionPolicy Bypass -File Python\run_sweep_commgate.ps1 -Stage 1 -RunStep 20000 -Seeds 42
```

미니런 확인 포인트:
- 런처가 출력하는 exe 빌드 날짜가 2026-07-03 이후인지
- 각 창이 에러 없이 진행되는지 (obs 크기 에러 = 옛 빌드)
- `results\<timestamp>_gate1_*\outcome.csv`의 agentId 종류가 16개인지 (스폰 확인)
- ORACLE 런 폴더에 `comm_stats.csv` 생성되는지

## 3. Stage 1 본 런 (OFF vs ORACLE — 정보가치 천장)

```powershell
powershell -ExecutionPolicy Bypass -File Python\run_sweep_commgate.ps1 -Stage 1
```

기본 1M 스텝 × 6런(2 arm × 시드 42/43/44) — 동시 6개 = 병렬 한계에 정확히 맞음. 밤새 실행.

모니터링 (선택):
```powershell
tensorboard --logdir <models 경로>
```
- `Comm/GateMean_Ctr` : 0.4~0.7 사이 유지가 정상, 0으로 하강하면 게이트 붕괴 신호
- `Comm/Grad_MsgEncoder` : pos_ground 활성 경로의 채널 생존 신호
- ORACLE arm은 채널 우회라 위 둘은 참고만

## 4. 판정 (다음 날 아침)

```powershell
python Python\analysis\analyze_run.py "results\<폴더>"
python Python\analysis\convergence_gate.py results\...gate1_OFF_s42\metric.csv results\...gate1_OFF_s43\metric.csv ...
```

- 판정은 **수렴 꼬리(마지막 30%)만**, seed-paired 비교. 중간 구간 성적으로 결정 금지
- `oracle > OFF` (충돌률·near-miss에서 뚜렷) → Stage 2 진행
- `oracle ≈ OFF` → 이 regime엔 통신 가치 없음 → regime 강화 후 재측정:
  ```powershell
  powershell -ExecutionPolicy Bypass -File Python\run_sweep_commgate.ps1 -Stage 1 -VesselCount 20
  ```
  ⚠️ RingScale 0.4/0.5는 스폰이 장애물 그리드와 겹쳐 금지(트러블슈팅 표 참조). 밀도 강화는 VesselCount↑로.
  단 씬 스폰 포인트가 20개라 VesselCount 상한 = 20 (초과분은 클램프됨).

## 5. Stage 2 (천장 확인 후에만 — ONsevered vs ONc5c)

```powershell
powershell -ExecutionPolicy Bypass -File Python\run_sweep_commgate.ps1 -Stage 2
```

판정: ONc5c가 ONsevered를 이기고 Stage 1의 oracle에 근접하는가.
- ONc5c ≈ oracle : 통신 완승
- 중간 : 부분 성공 — 공급·소비 계수(GoalComm/Consumer) 또는 attention 상향 검토
- ONc5c ≈ OFF : 채널 학습 실패 — GateMean·Grad_MsgEncoder·comm_stats로 원인 분류
- ONc5c < OFF : C5c 간섭 — Consumer 계수 하향

## 트러블슈팅

| 증상 | 원인 / 조치 |
|---|---|
| collision_obstacle ~77%, stepCount 중앙값 1 (즉사) | **RingScale 0.5 사용** — 스폰링(원본 250m)×0.5=125m 모서리 4점이 장애물 3×3 그리드(간격 120m, 캡슐반경 20m) 모서리 장애물 캡슐 안(중심 7m 옆)에 스폰됨. 0.4도 표면 근접이라 위험. **0.7 사용(기본값 수정됨)**, 밀도 강화는 VesselCount↑로 (2026-07-03/04 충돌위치 원피팅 실측) |
| RuntimeError: obs 크기 ≠ 369 | 옛 빌드가 연결됨 → 1번 재빌드 |
| "Not enough spawn points — clamping" 경고 | 씬 스폰 포인트 부족 → 씬에 추가 후 재빌드 |
| Unity 연결 실패 / 포트 에러 | 잔여 프로세스 확인: `Get-Process Vessel_MLAgent,python` 종료 후 재시도 (commgate는 5600~5615 사용) |
| `verify\_verify_ppo_mirror.py` FAIL | 돌리지 말고 FAIL 항목을 기록해 둘 것 (Mac 세션에서 원인 추적) |
| 학습이 비정상적으로 느림 (<10 steps/s) | attention/pos_ground 경로 문제 가능성 — comm_stats와 함께 기록 |

## 참고

- 브랜치: 기본 `main`(2026-09-10 승격, 0905 라인 = 리팩토링·공유 인코더 포함). `msgComparision` 은 Fig4 메시지 차원 축 전용 브랜치(RUNS.md). `Thick_MoE`/`Light_MoE`는 MoE 3-arm 비교용 —
  이번 계단(Stage 1/2)에서는 사용하지 않음
- ★2026-09-10 **YUGIOH** = config 기본값(최종판). 학습은 `bash run_repro.sh train`(env 불필요, preflight 가 드리프트 검사).
  옛 체크포인트 평가는 `VESSEL_COMM_RANGE=200` 등 학습값을 줘야 함 — 모르면 `python ckpt_io.py <ckpt> --env` 가 찍어줌(없으면 중단).
- 전 run의 설정 스냅샷은 각 결과 폴더의 `run_meta.txt` + `config_snapshot.txt`
  (`AGG_MODE=mean` 표기는 pos_ground 기본 ON에서는 미사용 폴백 — 무시)
- 판정 이후의 계획(집계 ablation, H2 msgdim)은 Stage 2 결과를 보고 결정

## imo open-sea 파일럿 (2026-09-21, feat/dyn-profile-imo) — Git Bash

전제: GitHub clone(Dropbox `.git` 아님). 체크포인트·출력은 agile 배치와 **별도 폴더**.

```bash
git fetch origin && git checkout feat/dyn-profile-imo && git pull
cd Python
# 0) preflight + 기본(agile) 스모크 — 비트동일 확인
bash run_repro.sh smoke
#    프로필 게이트를 두 env 에서 따로 확인 (preflight 는 그때의 env 로만 돈다)
python verify/test_dyn_profile.py                                                   # agile/grid3x3 → ALL PASS
VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none python verify/test_dyn_profile.py      # imo/none → ALL PASS (SKIP 2 = legacy 기본 전용 케이스)
# 1) imo open-sea 스모크
export VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none
export VESSEL_CKPT_DIR=$HOME/VESSEL_checkpoints/imo_opensea VESSEL_OUT_DIR=$PWD/_repro_out_imo
bash run_repro.sh smoke
# 2) 파일럿 학습: trunk(OFF 9,043,968) → off / on6 갈래, 3시드, 통신 텔레메트리 ON
#    ★2026-09-23: 목표는 대척(crossing 2)이 아니라 무작위(≥ MIN_GOAL_DIST 400 m) — 대척은 open-sea 에서 16척이 원점에 몰려 OFF 가 학습 실패(1차 파일럿 goal 50 %/vColl 47 %).
#    새 CKPT/OUT 폴더를 쓸 것(1차 trunk 는 crossing 2 라 재사용 불가 — check_branch 가 crossing 불일치로 막음).
export VESSEL_CROSSING=0
VESSEL_TRAIN_ARMS="off on6" VESSEL_COMM_TELEMETRY=1 bash run_repro.sh train
# 3) 난수 대조군: on6 갈래의 msg_sd 를 diag 로 읽어 sd 로 준다
VESSEL_DIAG_CKPTS="on6_s43.pt on6_s44.pt on6_s45.pt" bash run_repro.sh diag      # _repro_out_imo/diag_on6_s4x.json → telemetry.msg_sd
VESSEL_MSG_RANDOM_SD=<msg_sd 평균> bash run_repro.sh random
# 4) 평가 (분기 검사 자동)
bash run_repro.sh eval
# 5) 차원 스윕 (파일럿 통과 시): dim 2·12 짝
VESSEL_TRAIN_ARMS="off2 on2 off12 on12" bash run_repro.sh train && bash run_repro.sh eval
# 6) coastal 보조: VESSEL_OBSTACLES=grid3x3 로 1)~4) 를 다른 CKPT/OUT 폴더에서
```

평가는 체크포인트 스냅샷의 `dyn_profile`·`obstacles`·`radar_range` 와 현재 env 가 다르면 **중단**한다(조용히 다른 조건으로 재는 사고 방지).
2026-09-23 부터 보상 계수·게이트 상수 24개(`config.SIM_SNAPSHOT_KEYS`, 스냅샷 `sim`)까지 같이 대조한다 — 평가·재개 모두. 교차평가는 `--allow_sim_mismatch` / `VESSEL_ALLOW_SIM_MISMATCH=1`, 재개는 우회 없음(`ckpt_io.py <ckpt> --env` 가 뽑아 주는 export 줄을 쓸 것).
일부러 교차평가할 때만 우회: `VESSEL_ALLOW_SIM_MISMATCH=1`(`eval_mixed.py`·`measure_regimes.py`·`corridor_run.py`·`astar_fig9/eval_astar_global.py`)
/ `--allow_sim_mismatch`(`eval_ckpt.py`·`diag_ckpt.py`). 그렇게 낸 숫자는 학습 조건과 다르다고 반드시 같이 적을 것.
재개·분기는 우회가 없다 — 프로필 이름이 같아도 `dyn` 숫자가 다르면 학습기가 거부한다.
`astar_fig9/eval_astar_global.py` 는 imo 프로필에서 무효(스펙 §2: R 28 m 로 웨이포인트 추종 불가).

판정 기준·지표 = 스펙 §4(사전등록). 결과 표는 `eval_*.txt` 의 goal/vColl/fuel/headTravel/minSep/colregs/colregsOK + 시드별 승패. 결과 보고 기준 바꾸지 말 것.

## 의도·역할 통신 배치 (2026-09-25, feat/comm-intent) — Git Bash

스펙·사전등록 = `docs/superpowers/specs/2026-09-25-comm-intent-design.md` (결과 전 고정). 레이더 56 고정(저자 결정).
팔 5개가 **같은 EXT trunk** 에서 분기: off / arpa6(ARPA@56) / onl6(ON-latent) / ons6(ON-state) / oni6(ON-intent).
통신 팔은 전부 보조손실 0(`VESSEL_AUX_LOSS_SCALE=0`, run_repro 가 팔마다 export). 체크포인트 이름 접두어 `x_` 자동.

```bash
git fetch origin && git checkout feat/comm-intent && git pull
cd Python
export VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=0 VESSEL_COMM_EXT=1
export VESSEL_CKPT_DIR=$HOME/VESSEL_checkpoints/comm_intent VESSEL_OUT_DIR=$PWD/_repro_out_intent
export VESSEL_SEEDS="43 44 45"            # 2026-09-26 저자 확정: 3시드
# ★2026-09-26 한 명령: smoke → train(5팔) → eval → traj(F5) → ablate. 실패 단계에서 멈춤, 시각은 $VESSEL_OUT_DIR/_all_timeline.txt
#   preflight 는 첫 단계에서 검사 7종을 동시에 돌리고(PPO 미러 포함), 이후 단계는 코드·env 지문 캐시로 건너뜀
VESSEL_COMM_TELEMETRY=1 bash run_repro.sh all
# 중간에 끊기면 남은 단계만: bash run_repro.sh train|eval|traj|ablate (trunk·갈래 체크포인트 재사용, preflight 캐시 적중)
```
- 팔 기본값: `VESSEL_COMM_EXT=1` 이면 train 이 `off arpa6 onl6 ons6 oni6` 를 돈다(VESSEL_TRAIN_ARMS 로 바꿈)
- 예전의 `VESSEL_SKIP_GOLDEN=1` 수동 skip 은 필요 없음 — 같은 코드·env 면 자동 skip, 코드가 바뀌면 자동 재검사. 강제 재검사 `VESSEL_FORCE_PREFLIGHT=1`

병행 — 계획서 G6 2단계(보조손실 비대칭 절제). EXT 코드 불필요. 1차 파일럿 trunk 를 **새 폴더에 복사**해서 쓴다
(원 폴더에서 돌리면 1차 off 갈래를 덮어씀. 복사본 trunk 의 SHA 는 같으므로 분기 검사 통과).
**새 셸에서** 돌릴 것(`env | grep VESSEL_` 로 남은 값 확인) — 1차 trunk 는 sim 스냅샷 이전이라 학습기의 sim 대조가 건너뛰어지고,
trunk↔갈래 crossing 대조도 없어서 '1차 설정 그대로' 인지는 셸 env 만 보장한다:

```bash
unset VESSEL_COMM_EXT VESSEL_COMM_FIELDS VESSEL_COMM_LATENT VESSEL_PARTNER_RANGE VESSEL_AUX_LOSS_SCALE
P1=<1차 파일럿 CKPT 폴더>; P1OUT=<1차 파일럿 OUT 폴더>
export VESSEL_DYN_PROFILE=imo VESSEL_OBSTACLES=none VESSEL_CROSSING=2    # 1차 파일럿 설정 그대로(crossing 2)
export VESSEL_CKPT_DIR=$HOME/VESSEL_checkpoints/g6_aux0 VESSEL_OUT_DIR=$PWD/_repro_out_g6
export VESSEL_SEEDS="43 44 45"            # 1차 파일럿 시드 (복사하는 trunk 가 이 셋뿐)
export VESSEL_REQUIRE_TRUNK=1             # trunk 가 없으면 새로 학습하지 않고 중단
mkdir -p $VESSEL_CKPT_DIR $VESSEL_OUT_DIR
cp $P1/trunk_d6_s4{3,4,5}.pt $VESSEL_CKPT_DIR/ && cp $P1OUT/trunk_d6_s4{3,4,5}.csv $VESSEL_OUT_DIR/
VESSEL_TRAIN_ARMS="off on6a0" bash run_repro.sh train && bash run_repro.sh eval
# 판정(결과 전 해석표): on6a0 ≥ off → aux 비대칭이 H1a 해로움 원인 / on6a0 ≈ 1차 RANDOM → aux 는 일부만 / 변화 없음 → 원인 아님
```

GPU: ON형 팔은 프로세스당 VRAM 이 OFF 보다 크다(1차 파일럿 ON ≈ 5.6 GB, OFF ≈ 1.6 GB). EXT 는 attention MLP 로 조금 더 큼 — 첫 배치에서 `nvidia-smi` 로 확인.
판정·지표 = 스펙 §6. 조율 진단(`[조율/…]`)·텔레메트리는 기전 설명용이고 H1 판정 근거가 아님.
