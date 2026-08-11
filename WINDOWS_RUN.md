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
python Python\_verify_ppo_mirror.py

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
python Python\analyze_run.py "results\<폴더>"
python Python\convergence_gate.py results\...gate1_OFF_s42\metric.csv results\...gate1_OFF_s43\metric.csv ...
```

- 판정은 **수렴 꼬리(마지막 30%)만**, seed-paired 비교. 중간 구간 성적으로 결정 금지
- `oracle > OFF` (충돌률·near-miss에서 뚜렷) → Stage 2 진행
- `oracle ≈ OFF` → 이 regime엔 통신 가치 없음 → regime 강화 후 재측정:
  ```powershell
  powershell -ExecutionPolicy Bypass -File Python\run_sweep_commgate.ps1 -Stage 1 -VesselCount 24 -RingScale 0.4
  ```

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
| RuntimeError: obs 크기 ≠ 369 | 옛 빌드가 연결됨 → 1번 재빌드 |
| "Not enough spawn points — clamping" 경고 | 씬 스폰 포인트 부족 → 씬에 추가 후 재빌드 |
| Unity 연결 실패 / 포트 에러 | 잔여 프로세스 확인: `Get-Process Vessel_MLAgent,python` 종료 후 재시도 (commgate는 5600~5615 사용) |
| `_verify_ppo_mirror.py` FAIL | 돌리지 말고 FAIL 항목을 기록해 둘 것 (Mac 세션에서 원인 추적) |
| 학습이 비정상적으로 느림 (<10 steps/s) | attention/pos_ground 경로 문제 가능성 — comm_stats와 함께 기록 |

## 참고

- 브랜치: 기본 `msgComparision`(single 기본). `Thick_MoE`/`Light_MoE`는 MoE 3-arm 비교용 —
  이번 계단(Stage 1/2)에서는 사용하지 않음
- 전 run의 설정 스냅샷은 각 결과 폴더의 `run_meta.txt` + `config_snapshot.txt`
  (`AGG_MODE=mean` 표기는 pos_ground 기본 ON에서는 미사용 폴백 — 무시)
- 판정 이후의 계획(집계 ablation, H2 msgdim)은 Stage 2 결과를 보고 결정
