# 인계 — 저자 성공 지표 5 시드 실행 (2026-11-02, Windows 새 세션·다른 기계용)

새 세션은 이 순서로 읽음: 루트 `CLAUDE.md` §0 마지막 항목(11-01·02) → 이 파일 → 스펙 `docs/superpowers/specs/2026-11-02-author-goals-criteria.md`(`feat/fig1-latent` 03c4179, 판정 기준 = 결과 전 고정, 바꾸지 않음)

## 1. 지금 상태

- 2단계(qs_·qs2_) 결과 = 통신 이득 ~1 pt, 스펙 10-30 §3-2 대로 시드 45–47 은 대안 A(1단계 갈래)
- 11-02 저자 성공 지표 사전등록(03c4179): 곡선 M · Woerner · DCPA · 평균 |타각|, 지표마다 승 ≥ 4/5 + 95 % t-CI, 가드 도착·배 충돌 3 pt, 주 추정량 = 끝 모델 평균 행동 lrm, 시드 43–47 전부
- 남은 일 2개(아래 J1·J2). 11-02 오전 기준 어느 기계에서도 시작 기록 없음(analysis 5d5c9de 뒤 커밋 없음)
- 원인 분석 참고 = `analysis/pdf/2026-11-01_통신부진_원인_해결방안.pdf`

## 2. 할 일 — 일마다 기계 하나만

| 일 | 무엇 | 이 기계에 있어야 하는 것 | 시간·GPU |
|---|---|---|---|
| J1 | qd_·qd2_ 끝 모델 평균 행동 주 평가 lrm 4회(학습 없음) | `$CK/qd_off_s43.pt` · `qd_comm_s43.pt` · `qd2_off_s44.pt` · `qd2_comm_s44.pt`, `Python/_repro_out_qd/`·`_repro_out_qd2/` 의 `_codec_choice.txt` + `codec/` 폴더, `_curve.md`·`lr_*` | 각 약 70분, 카드당 1개(약 1.1 GB) |
| J2 | qd3_ → qd4_ → qd5_ phase1(코덱 → 갈래 off·comm → 분기 검사 → lr·lrm 평가 → 곡선 → 표) | `$CK/qd3_trunk_d6_s45.pt` · `qd4_trunk_d6_s46.pt` · `qd5_trunk_d6_s47.pt`, `Python/_repro_out_qd3/`·`qd4/`·`qd5/` 폴더 전체(`_timeline.txt`·`_seeds.txt`·`_batch_env.txt`·`_commit.txt`·trunk CSV) | 접두어당 4–5시간, 갈래 1개 약 8 GB |

- `$CK` = `VESSEL_CKPT_DIR`(기본 `$HOME/VESSEL_checkpoints/comm_intent`)
- **같은 일을 두 기계에서 돌리지 않음.** 같은 시드 갈래를 두 번 학습하면 GPU 비결정성 때문에 값이 달라지고, 둘 중 하나를 고르면 사후 선택이 됨. 실수로 두 번 돌았으면 먼저 끝난 것을 쓰고 둘 다 보고
- 결과 사본이 Dropbox `runs/2026-10-02_fig1/out/<접두어>/` 에 쓰임 → 두 기계가 같은 접두어를 돌리면 Dropbox 충돌 사본도 생김
- 어느 기계가 어느 일을 할지는 저자가 정함. 시작 전 다른 기계가 이미 시작했는지 저자에게 확인

## 3. 이 기계 점검 (Git Bash, 클론 루트 `C:\work\DT_Vessel_v3`)

```bash
cd /c/work/DT_Vessel_v3 && git fetch origin && git checkout feat/fig1-latent && git pull && git log -1 --oneline   # 03c4179 이상
git status --short                                     # 추적 파일 변경 없어야 함
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
tasklist //FI "IMAGENAME eq python.exe"                # 이미 도는 학습·평가가 있으면 손대지 말고 보고
CK=${VESSEL_CKPT_DIR:-$HOME/VESSEL_checkpoints/comm_intent}
ls -la $CK/qd_off_s43.pt $CK/qd_comm_s43.pt $CK/qd2_off_s44.pt $CK/qd2_comm_s44.pt          # J1
ls -la $CK/qd3_trunk_d6_s45.pt $CK/qd4_trunk_d6_s46.pt $CK/qd5_trunk_d6_s47.pt            # J2
grep CODEC_ Python/_repro_out_qd/_codec_choice.txt Python/_repro_out_qd2/_codec_choice.txt  # 코덱 파일 경로가 이 기계에 실제로 있는지 ls 로 확인
tail -2 Python/_repro_out_qd3/_timeline.txt Python/_repro_out_qd4/_timeline.txt Python/_repro_out_qd5/_timeline.txt
cat Python/_repro_out_qd3/_batch_env.txt
ls Python/_repro_out_qd/lrm_* Python/_repro_out_qd2/lrm_* 2>/dev/null                      # 이미 된 lrm 이 있나
```

- 파이썬 = `$HOME/anaconda3/envs/mltest/python.exe`(다르면 `VESSEL_PY=` 로 줌)
- 파일이 없으면 그 일은 이 기계에서 못 함 → 멈추고 저자에게 보고
  - **trunk 다시 학습 금지.** 사전등록한 시드 45–47 trunk 는 이미 돈 그 파일임(다시 돌리면 다른 trunk)
  - 다른 기계에서 복사하려면: 위 `.pt` + `Python/_repro_out_<접두어>/` 폴더 통째. 코덱 경로가 `_codec_choice.txt` 에 `C:/work/DT_Vessel_v3/...` 절대 경로로 박혀 있음 → 클론 경로가 같아야 함
- `_batch_env.txt` 는 `ARPA=56 RANK=v5 REWARD_RANGE=56 EXTRA=[VESSEL_ENTROPY_BONUS=0 VESSEL_LOGSTD_MIN=-3.0 VESSEL_LOGSTD_INIT0=-2.0 VESSEL_GRAD_TELEMETRY=1]` 이어야 함(J2 명령 env 와 같아야 실행기가 받음)

## 4. 명령

### J1 — lrm 4회 (여유 2.5 GB 이상인 카드만, 카드당 1개)

```bash
S=Python/runs_fig1/2026-10-02_fig1/lrm_eval.sh
CUDA_VISIBLE_DEVICES=0 VESSEL_F_REWARD_RANGE=56 VESSEL_F_SEED=43 nohup bash $S qd_  off  > lrm_qd_off.log  2>&1 &
CUDA_VISIBLE_DEVICES=1 VESSEL_F_REWARD_RANGE=56 VESSEL_F_SEED=43 nohup bash $S qd_  comm > lrm_qd_comm.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 VESSEL_F_REWARD_RANGE=56 VESSEL_F_SEED=44 nohup bash $S qd2_ off  > lrm_qd2_off.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 VESSEL_F_REWARD_RANGE=56 VESSEL_F_SEED=44 nohup bash $S qd2_ comm > lrm_qd2_comm.log 2>&1 &
# 넷 다 끝나면 표: VESSEL_F_SEED=43 bash $S qd_ --table ; VESSEL_F_SEED=44 bash $S qd2_ --table
```

### J2 — qd3_ → qd4_ → qd5_ phase1 (차례대로, 메모리가 빌 때까지 실행기가 기다림 = 멈춘 것 아님)

```bash
nohup bash -c 'for p in "qd3_ 45" "qd4_ 46" "qd5_ 47"; do set -- $p
  VESSEL_F_ALLOW_BUSY=1 VESSEL_F_IMIT=0 VESSEL_F_PREFIX=$1 VESSEL_F_SEEDS=$2 VESSEL_F_NGATE=1 VESSEL_F_REWARD_RANGE=56 \
  VESSEL_F_EXTRA_ENV="VESSEL_ENTROPY_BONUS=0 VESSEL_LOGSTD_MIN=-3.0 VESSEL_LOGSTD_INIT0=-2.0 VESSEL_GRAD_TELEMETRY=1" \
  VESSEL_F_RANK_OVERRIDE="저자 10-29: 순위 관문 v5 불통과(시드1 (iii) 여유 부족·(v) v4 와 같은 값) 넘기고 성공 경로 진행" \
  VESSEL_F_GPU_CAP=1 VESSEL_F_JOBS=4 PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3 \
  bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh phase1 || exit 1; done' > run_qd345_phase1.log 2>&1 &
```

- smoke·관문·trunk 단계는 `_timeline.txt` 에 '끝' 이 있어 건너뜀. 코덱부터 돎
- 진행 확인: `tail -3 Python/_repro_out_qd3/_timeline.txt`, `tail run_qd345_phase1.log`

### 끊긴 뒤 다시 이어 돌리기

- 같은 명령 그대로. `lrm_eval.sh` 는 끝난 파일(`R8=` 줄)을 건너뛰고, `_run_f.sh` 는 `_timeline.txt` 의 '끝' 단계를 건너뜀
- 먼저 `tasklist`·`nvidia-smi` 로 이전 프로세스가 아직 도는지 확인. 돌고 있으면 다시 띄우지 않음
- 프로세스 kill·락/표식 파일(`.gpu*_*`) 삭제 전 그 프로세스가 실제로 죽었는지 확인(CLAUDE.md §5, Windows 는 트리가 안 죽는 경우 있음)
- 감시 스크립트는 띄우지 않아도 됨(10-31 감시가 RAM 부족으로 멈춘 일 있음). nohup 이라 세션이 끊겨도 이어짐

## 5. 끝나면

1. 5 시드 표(J1·J2 둘 다 끝난 기계에서, 아니면 출력 폴더를 모은 뒤):
   ```bash
   cd Python && $HOME/anaconda3/envs/mltest/python.exe runs_fig1/2026-10-02_fig1/goals_fig1.py \
     _repro_out_qd:qd_ _repro_out_qd2:qd2_ _repro_out_qd3:qd3_ _repro_out_qd4:qd4_ _repro_out_qd5:qd5_ | tee _goals_fig1.md
   ```
2. 결과를 `analysis` 브랜치 `analysis/runs/2026-10-02_fig1/out/<접두어>/` 에 올림(.txt .md .log .csv .json, `.pt` 제외) — 저자 상시 지시라 묻지 않고 커밋·push. 실행 기록(시각·기계 이름·GPU·명령과 다르게 한 것)은 `out/q1102_notes/runnote.txt`
3. 보고 규칙: 숫자는 파일 값 그대로 · 가드 먼저 · 시드별 승패 같이 · 시드 1–2개 단독 주장 금지 · 판정 기준(스펙 2026-11-02) 안 바꿈 · 43·44 는 탐색, 45–47 은 확증 성격으로 구분
4. ★멈춤·★FAIL 이면 재시도하지 말고 로그와 함께 저자에게 보고

## 6. 주의

- 11-01 서버에서 다른 계정이 카드당 13–15 GB 를 씀 → J2 는 그동안 시작 못 함(실행기가 기다림). 한 GPU 에 통신 학습 3개 이상 금지
- 2단계 곡선 버그(`ckpt_io.make_env_from_snapshot` 이 `snap['sim']` 을 다시 덮어씀)는 J1·J2 와 무관(1단계는 `sim_from_env` 안 씀). 고치는 코드는 J2 끝난 뒤 별도 커밋으로 — 학습·평가 코드가 바뀐 커밋이면 실행기가 같은 접두어를 거부함
- 계획에 없는 실험·코드 변경 안 함
