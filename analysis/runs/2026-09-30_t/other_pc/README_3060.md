# 3060 PC 에서 돌릴 것 — Stage A (2026-09-30)

## 무엇
- 통신 갈래 6개 학습: `s_a6`(latent 복원형) · `s_c6`(latent 직접 읽기) × 시드 43·44·45
- 출발점 = 서버에서 이미 학습한 s_ 뿌리(통신 없이 9,043,968 결정) — 이 폴더 `transfer/ckpt/` 에 들어 있음(개당 3 MB, 합 18 MB)
- 평가는 서버에서 함. 여기선 학습만 하고 결과 체크포인트 6개(합 18 MB)가 `back/` 에 자동으로 복사됨

## 준비 (한 번만)
1. Git for Windows(Git Bash) 있어야 함
2. python 에 torch(CUDA)·numpy 필요. 없으면:
   - `pip install torch --index-url https://download.pytorch.org/whl/cu124`
   - `pip install "numpy<2"`
3. 확인: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → `True NVIDIA GeForce RTX 3060`

## 실행 (Git Bash, 한 줄)
```
bash "<Dropbox 경로>/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t/other_pc/_run_stageA_3060.sh"
```
- 저장소가 없으면 `C:\work\DT_Vessel` 에 자동으로 받음(브랜치 `feat/latent-sweep`, 커밋 9cc8f6d)
- 다른 위치·다른 python 이면 앞에 붙임: `VESSEL_REPO=/c/내/경로 VESSEL_PY=/c/경로/python.exe bash ...`
- GPU 메모리가 빠듯해서 멈추면(로그 15분 무진행) `VESSEL_GPU_CAP=1` 을 앞에 붙여 한 번에 1개씩

## 걸리는 시간 (추정)
- 갈래 1개 = 700만 결정. 서버에서 70분(초당 1,650 결정, CPU 1코어가 병목). 3060 PC 는 CPU 에 따라 70–110분
- 2개씩 동시에 → 3번 = 약 3.5–5.5시간. 1개씩이면 7–11시간
- 진행 확인: `C:\work\DT_Vessel\Python\_repro_out_s_stageA\s_a6_s43.log` 의 `[ON] dec=…M` 줄(16.06M 에서 끝)

## 끝나면
- `back/_DONE.txt` 가 생김 → 서버 쪽에서 체크포인트를 받아 평가·절제를 돌림
- 중간에 끊겨도 같은 명령을 다시 치면 끝난 갈래는 건너뜀

## 주의
- 이 PC 의 torch 버전이 서버와 달라 '골든 비트동일 검사'는 건너뜀(`VESSEL_SKIP_GOLDEN=1`). 미러 검사는 그대로 통과해야 시작함
- Stage A 는 탐색 배치(사후 추가 팔). 결과는 그대로 보고하되 확정 근거로 쓰지 않음

## 진행 상황을 서버 그림에서 보려면 (선택)
- 학습 스크립트와 **다른** Git Bash 창에서: `bash "<Dropbox>/Private_Paper_Project/0702_NewVessel/runs/2026-09-30_t/other_pc/_sync_3060.sh"`
- 5분마다 곡선 CSV·로그(작은 텍스트)를 `back/live/` 로 복사 → 서버가 만드는 `runs/2026-09-30_t/live/reward_live.pdf` 오른쪽 아래 칸에 Stage A 곡선이 뜸
