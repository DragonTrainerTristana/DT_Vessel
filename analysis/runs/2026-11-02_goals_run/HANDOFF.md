# 인계 — 새 기계에서 5 시드 처음부터 (2026-11-02 개정, DESKTOP-JA8O9SS)

새 세션은 이 순서로 읽음: 루트 `CLAUDE.md` §0 마지막 항목 → 이 파일 → 스펙 `docs/superpowers/specs/2026-11-02-author-goals-criteria.md` **§6**(`feat/fig1-latent` 974f3d2, 판정 기준 = 결과 전 고정, 바꾸지 않음)

## 1. 상황

- 서버(LICS-SERVER) 고장. qd_·qd2_ 갈래·qd3–5_ trunk 체크포인트와 코덱은 서버 로컬에만 있었음 → 이 기계에 없음(11-02 점검 확인)
- 저자 지시 '이제부터 여기서 진행' → 같은 처방(qd_ env)으로 시드 43–47 을 새 접두어 `qr_`(43)·`qr2_`(44)·`qr3_`(45)·`qr4_`(46)·`qr5_`(47) 로 처음부터
- 판정 = 저자 성공 지표(곡선 M · Woerner · DCPA · 평균 |타각|, 승 ≥ 4/5 + 95 % t-CI, 가드 도착·충돌 3 pt, 끝 모델 평균 행동 lrm). 5 시드 모두 기준 고정 뒤 학습 → 모두 확증 성격
- 순서: **시드 43(`qr_`) 먼저** → 끝나면 저자가 보고 44–47 진행 결정(저자 상시 규칙 '시드 1개 먼저')
- 옛 서버 결과(analysis `out/qd*`)는 탐색 기록. 판정에 섞지 않음
- 이전 판 인계(J1·J2, 서버 파일 전제)는 폐기 — git 기록에만 남음

## 2. 기계 준비 (Git Bash)

```bash
cd /c/work && git clone https://github.com/DragonTrainerTristana/DT_Vessel.git DT_Vessel_v3
cd /c/work/DT_Vessel_v3 && git checkout feat/fig1-latent && git log -1 --oneline     # 974f3d2 이상
ls $HOME/anaconda3/envs/mltest/python.exe                                             # 다르면 VESSEL_PY= 로
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv             # 다른 앱 GPU 메모리
ls -d $HOME/Dropbox/Private_Paper_Project/0702_NewVessel/runs/2026-10-02_fig1 2>/dev/null   # 있으면 결과 사본이 Dropbox out/qr/ 에 쓰임
```

- 옛 클론 `C:\work\DT_Vessel`(9183a0f)은 건드리지 않음
- GPU 다른 앱 사용량이 크면(11-02 점검 3.1 GB) 저자에게 알림 — trunk 가 서버에서 약 9.3 GB 였음. 프로세스는 임의로 끄지 않음

## 3. 시드 43 실행 (`qr_` phase1 = smoke → 순위 관문 override 기록 → trunk → 관문 → 코덱 → 갈래 off·comm → 분기 검사 → lr·lrm 평가 → 곡선 → 표)

```bash
cd /c/work/DT_Vessel_v3
PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3 VESSEL_F_GPU=0 VESSEL_F_GPU_CAP=1 VESSEL_F_JOBS=1 \
VESSEL_F_IMIT=0 VESSEL_F_PREFIX=qr_ VESSEL_F_SEEDS=43 VESSEL_F_NGATE=1 VESSEL_F_REWARD_RANGE=56 \
VESSEL_F_EXTRA_ENV="VESSEL_ENTROPY_BONUS=0 VESSEL_LOGSTD_MIN=-3.0 VESSEL_LOGSTD_INIT0=-2.0 VESSEL_GRAD_TELEMETRY=1" \
VESSEL_F_RANK_OVERRIDE="저자 10-29: 순위 관문 v5 불통과(시드1 (iii) 여유 부족·(v) v4 와 같은 값) 넘기고 성공 경로 진행" \
nohup bash Python/runs_fig1/2026-10-02_fig1/_run_f.sh phase1 > run_qr_phase1.log 2>&1 &
```

- 진행: `tail -3 Python/_repro_out_qr/_timeline.txt`, `tail run_qr_phase1.log`
- smoke 안에 골든·미러 검사(preflight)가 있음. ★11-02 시도 1: 골든만 불통과(torch 2.6.0+cu124·CPU 다름), 다른 14종 PASS → 기록 `out/qr_try1_golden/`
  - 저자가 면제를 정하면 위 명령에 `VESSEL_PY=<Python310 python.exe> VESSEL_F_GOLDEN_WAIVE="<저자 문구>"` 를 더해 같은 명령으로 다시(`feat/fig1-latent` 42f59b5 이상). 실행기가 '불통과 = 골든 하나뿐 · 학습·평가 코드 = c034422' 일 때만 `_golden_waiver.txt` 기록 후 진행, 아니면 멈춤
  - mltest 는 CPU 전용 torch 1.13.1 → `VESSEL_PY` 로 torch 2.6.0+cu124 파이썬을 줌(runnote 에 적음)
- **trunk 시작 뒤 10분**: `nvidia-smi` 로 VRAM·`_repro_out_qr/qr_trunk_d6_s43.log` 의 dec/s 확인. 메모리가 넘치거나 속도가 크게 떨어지면(Windows 공유 메모리로 넘침) 멈추고 보고. **학습 인자(envs·vessels·rollout) 줄이기 금지**(다른 실험이 됨)
- 예상 시간(추정): trunk 2–3 h · 갈래 2개 차례로 4–5 h · 평가 4개 · 곡선 15점 → 시드 하나 약 12–15 h
- trunk 관문 불통과 = ★멈춤 → 시드 바꾸지 말고 보고

### 끊긴 뒤 다시

- 같은 명령 그대로(끝난 단계는 `_timeline.txt` 의 '끝' 으로 건너뜀)
- 먼저 `tasklist //FI "IMAGENAME eq python.exe"`·`nvidia-smi` 로 이전 런이 아직 도는지 확인. 돌면 다시 띄우지 않음
- kill·표식 파일(`.gpu*_*`) 삭제 전 프로세스가 실제로 죽었는지 확인(CLAUDE.md §5)

## 4. 시드 43 끝나면

1. trunk 건강 확인(기록, 판정 아님): `CUDA_VISIBLE_DEVICES=0 VESSEL_F_REWARD_RANGE=56 VESSEL_F_SEED=43 bash Python/runs_fig1/2026-10-02_fig1/meaneval_ckpts.sh qr_ trunk_d6` → `_repro_out_qr/meaneval/_meaneval_qr_.md`(4M·9.04M 표본/평균). 서버 qd_ 9.04M 평균 69.4 % 와 나란히 적음
2. 결과 올리기: `analysis` 브랜치 `analysis/runs/2026-10-02_fig1/out/qr/`(.txt .md .log .csv .json, `.pt` 제외) + 실행 기록 `out/q1102_notes/runnote.txt`(시각·기계·GPU·명령과 다르게 한 것). 저자 상시 지시라 묻지 않고 커밋·push
3. 저자에게 보고 후 44–47 결정 대기. 44–47 명령 = 위 명령에서 `VESSEL_F_PREFIX=qr2_ VESSEL_F_SEEDS=44` … `qr5_ 47` 로 바꿔 차례로(한 줄 for 문, `|| exit 1`)

## 5. 5 시드 다 끝나면

```bash
cd Python && $HOME/anaconda3/envs/mltest/python.exe runs_fig1/2026-10-02_fig1/goals_fig1.py \
  _repro_out_qr:qr_ _repro_out_qr2:qr2_ _repro_out_qr3:qr3_ _repro_out_qr4:qr4_ _repro_out_qr5:qr5_ \
  --subset '건강=<qd_ §2 3/3 시드>' | tee _goals_fig1.md
```

- 보고 규칙: 숫자는 파일 값 그대로 · 가드 먼저 · 시드별 승패 · 시드 1–2개 단독 주장 금지 · 판정 기준 안 바꿈

## 6. 주의

- 계획에 없는 실험·코드 변경 안 함. ★멈춤·★FAIL 이면 재시도 말고 로그와 함께 보고
- 2단계 곡선 버그(`ckpt_io.make_env_from_snapshot`)는 phase1 과 무관. 고치는 코드는 5 시드 끝난 뒤 별도 커밋(학습·평가 코드가 바뀐 커밋이면 실행기가 같은 접두어를 거부함)
- `.pt`·출력 폴더는 이 기계 로컬에만 생김 → 시드 하나 끝날 때마다 결과(.pt 제외)를 analysis 에 올려 둠. `.pt` 백업 위치는 저자와 정함(Dropbox 금지 규칙)
