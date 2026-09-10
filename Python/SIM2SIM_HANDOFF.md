# SIM2SIM 핸드오프 — Windows 이어작업

작성 2026-07-04. GPU 배치 시뮬레이터(vessel_gym)로 탐색 100배 가속 → Unity는 ground-truth 판정 전용.
파일은 Dropbox 동기화로 Windows에 이미 있음(★미커밋 — 커밋 금지, 검증 후 결정). 새 세션은 이 문서만 읽고 이어가면 됨.

---

## 0. 왜 이제 Windows인가

Mac에서 막힌 유일한 것 = Unity 실행 부재 → 진짜 충실도 대조 불가. Windows엔 Unity 빌드(`Build\0703`) + GPU + 동기화된 파일이 다 있음.
Windows에서 할 것: ① Unity 충실도 대조(핵심) ② GPU 100배 실측 ③ Stage 1(OFF vs ORACLE) 본실행.

## 1. 현재 상태 (Mac에서 검증 완료)

- `vessel_gym.py` — GPU 배치 시뮬레이터. 동역학(C# 수식 그대로, 10 서브스텝)·레이더 360ray(선박 OBB + 장애물 원 + 벽)·COLREGs 상황판정(사각지대 fix)·보상(navigation/collision/per-pair/far-field 등)·비동기 스폰/에피소드·369D obs.
- `vessel_gym_train.py` — networks.py의 CNNPolicy를 그대로 재사용하는 배치 PPO. OFF/ORACLE arm.
- `test_vessel_gym_fidelity.py` — 충실도 테스트 6종.

검증된 것 (Mac CPU):
- 배칭 정확성: 배치 텐서 == C# 수식 스칼라 전사, 오차 6e-5
- 물리 상식: full-rudder 선회율 45°/s, 직진 가속, drag 평형 정확
- 레이더 기하: 정면 장애물 거리·측면 max range 정확
- 상황판정: 정면→HeadOn, 우현횡단→GiveWay
- **학습됨**: OFF arm vColl 1%→0%, 보상 −12.7→−3.0 단조 상승, NaN 없음. ORACLE도 정상.
- **Unity 이식 호환**: 체크포인트가 networks.CNNPolicy에 strict 로드(파라미터 369,131 일치).

## 2. Windows 즉시 확인 (환경 sanity, ~5분)

```powershell
cd <Dropbox>\Private_Paper_Project\0702_NewVessel\Assets\Scripts\Python
python test_vessel_gym_fidelity.py       # 6종 전부 PASS + GPU면 처리량에 cuda 표시
```
Windows는 torch가 최신이라 `meshgrid` 경고가 뜰 수 있으나 무해(기본 'ij'). NaN/에러 없으면 통과.
GPU 처리량이 [6]에 찍힘 — 1024환경에서 초당 decision × Unity(~400) 대비 배율 확인.

## 3. 다음 작업 (우선순위)

### 3.1 ★Unity 충실도 대조 (핵심 — C# 수정 불필요)
obs 369D에 pos[366,367]·heading[364]·speed[362]·rudder[365]·radar[0:360]이 이미 있음.
고정 action을 Unity에 먹이고 받은 obs를 로깅 = ground-truth 궤적. 같은 초기조건·action을 vessel_gym에 넣어 비교.
- 스크립트 골격: `unity_fidelity_compare.py` (vessel_gym 쪽 완성, Unity 연결부는 main.py 패턴 재사용 — 아래 4장 레시피).
- 결정성 확보: `VESSEL_SPEED_MULT_MIN=1.0 VESSEL_SPEED_MULT_MAX=1.0`(maxSpeed 고정), `VESSEL_USE_COMM=0`.
- 통과 기준: 동역학 궤적(pos/heading/speed/rudder) 스텝별 오차 작음(<1% 수준). 어긋나면 그 항이 전사 오류 → 수정.

### 3.2 GPU 처리량 실측
`test_vessel_gym_fidelity.py`가 자동으로 cuda 감지. 1024환경 초당 decision을 기록 → 100배 목표 확인.
필요시 `vessel_gym_train.py --envs 1024 --arm OFF --steps 1000000` 벽시계 시간 측정.

### 3.3 스폰 좌표 정밀화 (현재 링 근사)
vessel_gym의 스폰은 링 근사(`spawn_ring_r`, `spawn_pts`). 실제 씬 spawnPoint 20개·goalPoint 16개 좌표 미추출.
`Assets/Scenes/Simulation.unity`에서 활성 아레나('대양', center 로컬 원점화)의 spawn/goal Transform 좌표를 뽑아
`VesselBatchEnv.__init__`의 `self.spawn_pts`를 실제 값으로 교체. 데이터 추출이라 Unity 없이도 가능.

### 3.4 보상 잔여항 + Stage 2
- 미구현: COLREGs compliance / earlyAvoid 보상항(spec_reward.json에 수식 있음). navigation·collision·per-pair·far-field는 구현됨.
  Stage 1(OFF vs ORACLE, 판정=collision_vessel/goal)엔 불필요하나 정밀 재현엔 추가.
- Stage 2(학습된 통신 ON): `vessel_gym_train.py`는 OFF/ORACLE만. ON은 통신 집계를 배치화해야 함
  (networks.py 통신경로가 per-env dict → 배치 텐서 재작성 필요). others_msg를 배치로 만들면 CNNPolicy 서브모듈 재사용 가능.

### 3.5 Stage 1 본실행 (충실도 통과 후) — ★완료 (2026-07-06, oracle>OFF 확정)

결과 요약 (상세는 results\vg_stage1\ 로그 + gteval 폴더들):
- vessel_gym 4M-eq(64M dec)×3seed, 꼬리 30%: ORACLE이 vColl 3/3 우세(평균 −0.51%p), goal 2/3 우세(+5.2%p), Pareto 2/3.
- **Unity GT 재확인**(run_eval_gt.ps1, VESSEL_TRAIN=0 eval모드+LOAD_MODEL=1, ~1,100ep/런):
  ORACLE vColl 3/3 우세(2.58→1.12 / 0.97→0.75 / 1.87→0.88, 평균 −0.89%p), goal 2/3(+10.9/+13.7%p, s42만 −1.4).
  gym↔GT 방향 완전 일치 → sim2sim 전이 실증. **Stage 1 게이트 통과: 통신할 가치 있는 정보 존재 → Stage 2 근거.**
- 유의: eval 표본 ~1,100ep면 vColl 차이 수 건~수십 건 규모 — 논문 수치는 RUN_STEP 300k+ 권장.
- 이 과정에서 잡은 sim 버그(전부 수정·주석화): 리스폰 점유 무시(가짜 vColl 연쇄), oracle 원좌표 주입(스케일 폭파),
  main.py rms 경로(.pt에서 자기 자신 np.load 크래시), 학습 로그 분모(전-step→종료 기준).
```powershell
python vessel_gym_train.py --arm OFF    --steps 1000000 --envs 1024 --seed 42 --save vg_off_s42.pt
python vessel_gym_train.py --arm ORACLE --steps 1000000 --envs 1024 --seed 42 --save vg_oracle_s42.pt
# seed 43,44 반복 → oracle > OFF (collision_vessel 기준) 확인 = 통신할 가치 있는 과제 판정
```
oracle이 OFF를 못 이기면 regime 강화(--ring 0.5, vessels↑). 이기면 Stage 2 통신 학습으로.
★검증용 최종은 이 정책을 Unity에 `VESSEL_LOAD_MODEL=1`로 로드해 ground-truth 재확인.

## 4. Unity 충실도 대조 레시피 (상세)

main.py의 mlagents 연결부(`UnityEnvironment`, `BehaviorSpec`, `get_steps`/`set_actions`)를 재사용.

1. env 설정: `VESSEL_ENV_PATH=Build\0703\Vessel_MLAgent.exe`, `VESSEL_SPEED_MULT_MIN=1.0`, `VESSEL_SPEED_MULT_MAX=1.0`, `VESSEL_USE_COMM=0`, `VESSEL_GRAPHICS=0`.
2. Unity 연결 → 첫 decision step의 obs 수신. vessel 0 하나만 추적.
3. 첫 obs에서 초기상태 복원: posX=obs[366], posZ=obs[367], heading=obs[364]×180, speed=obs[362]×1.0(maxSpeed 고정), rudder=obs[365]×30.
4. 고정 action 시퀀스 정의 (예: 처음 20결정 [0.5,0.8], 다음 20결정 [-0.7,1.0], 다음 20 [0.0,-0.5]).
5. 매 decision: action을 Unity에 set → step → obs 수신·로깅(pos/heading/speed/rudder/radar). vessel 0만.
6. vessel_gym: E=1,N=1로 3번 초기상태 세팅, maxSpeed=1.0, 같은 action 시퀀스로 step, 같은 값 로깅.
7. 스텝별 비교. **주의점**:
   - 동역학 궤적 비교는 장애물 없는 열린 공간이 깔끔(radar는 별도 테스트). Unity 첫 스폰이 장애물 근처면 결과가 지저분 → 초기 pos를 중앙에서 떨어진 곳으로 강제하거나 여러 시드 중 깨끗한 것 선택.
   - radar 대조: 정적 배(action 0, speed 0) 한 시점의 360 array를 Unity obs[0:360]와 vessel_gym radar 비교. 장애물/벽 좌표가 정확해야 일치(3.3 스폰·장애물 좌표 정밀화 후).
   - Unity heading 규약: eulerAngles.y (0=+Z, 시계+). vessel_gym도 동일 규약 사용(확인됨).
   - Unity는 물리스텝 10회/결정, obs는 결정시점. vessel_gym.step()도 10 서브스텝/호출 → 1:1 정렬.

## 5. 알려진 근사/갭 (정직)

★2026-07-04 Windows 검증 결과로 갱신:

- ~~**Unity 대조 미완**~~ → **동역학 대조 PASS** (80결정, pos 최대 4cm·heading 0.0006°·speed/rudder 0.000 — `unity_fidelity_compare.py` 실측). **레이더 3종 대조 PASS** (`radar_fidelity_compare.py`: 장애물 2,697ray 최대 0.019m / 벽 7,224ray 0.000 / 타선 OBB 563ray 평균 0.033m).
  - 이 과정에서 **벽 버그 발견·수정**: 씬의 벽은 두께 1.0 box(중심 ±300) → 레이더가 만나는 건 내면 ±299.5. 평면 ±300 가정은 평균 0.64m 계통오차였음 → vessel_gym `ARENA_INNER=299.5`로 수정 완료.
  - 잔여 양성 이상: bow-tip grazing(53~56m) ray ~0.1m (obs 기준 0.2% 미만, 전이 무해 — radar_fidelity_compare.py 헤더 참조).
- **GPU 처리량 실측**: cuda 1024환경 60,357 decision/s ≈ Unity 대비 **151×** (목표 100× 초과).
- ~~**스폰 링 근사**~~ → **3.3 완료**: 씬 파싱으로 전 좌표 확보·교체됨. spawn 20점 = ±250 사각 둘레(100m 간격, 원 아님!), goal 16점 = ±200 사각 둘레(스폰 안쪽 중첩), 장애물 9개 ±120 그리드 r20(캡슐 r0.5×scale40) ✓, 벽 ±300 두께 1.0 ✓. vessel_gym에 `SPAWN_PTS_SCENE`/`GOAL_PTS_SCENE` 상수 + C# antipodal 배정(리스트순서 타이브레이크 동일) 반영, 전 스폰 antipode 거리 70.7m 균일 검산. ⚠️CROSSING=1은 미구현(대척=2만, 주석 참조).
- **보상 일부**: per-step 항 결정단위×10 근사, COLREGs compliance/earlyAvoid 미구현(3.4).
- **선박-선박 레이더/충돌**: OBB로 정밀화됨(초기 원 근사의 vColl 50% 왜곡 → OBB로 0%대 해결). 벽 충돌은 대각반경 보수 근사(내면 기준으로 수정됨).
- **maxSpeed 재추첨**: Unity는 Initialize 1회 고정(배별), vessel_gym은 리셋마다 재추첨 → 분포 다름. 정밀 재현 시 배별 고정 옵션 추가.
- **Stage 2 통신 미지원**: OFF/ORACLE만.

## 6. 파일·상수 참조

- vessel_gym 상수: DT=0.04, SUBSTEPS=10, MAX_TURN_RATE=30, RUDDER_RATE=12, TURN_FACTOR=1.5, MAX_YAW=45, RADAR_RANGE=56, 장애물 반지름 20(3×3 step120), 배 box 반길이 7.09·반폭 0.96.
- 수식 추출 명세(근거): 세션 스크래치패드 `spec_{dynamics,radar,reward,colregs,episode,scene}.json` (휘발 — 재추출은 C# 재정독).
- 관련 메모리: sim2sim-plan, comm-curriculum-hypotheses, full-analysis-2026-07-02.
- Unity 학습(느린 ground-truth 경로)은 `WINDOWS_RUN.md` + `Build\0703\HANDOFF.md` 참조. 이 문서는 빠른 sim2sim 트랙.
