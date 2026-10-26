# 레퍼런스 포팅 보고서 (2026-10-26)

- 작성: 읽기 전용 workflow 13개 에이전트(주제 5 × 조사·검증 + 포팅 명세·리뷰 + 종합). 새 학습·평가 없음
- 입력·출력 원자료: 이 폴더 `references_verified.json`(레퍼런스·추출 설정·검증 판정) · `port_spec.json`(포팅 명세·리뷰)
- Zotero 컬렉션 `Vessel MARL 포팅 레퍼런스 (2026-10-26)`(MNMHENXE), 하위 R1 HID5VCBB · R2 KQ9WPEPN · R3 SIRMSCNA · R4 XT95ACKF · R5 85JUAMQN

## 0. 쉬운 요약

- 레퍼런스들은 네 가지를 우리와 다르게 함
  - 행동을 작게, 천천히 바꿈. 침로나 타각을 조금씩 더해 가는 방식이고 결정 간격은 3–30 s임
  - 위험 벌점은 에이전트가 실제로 보는 범위 안에서만 계산함
  - 평가는 평균 행동(잡음을 끈 상태)으로 함
  - 결과는 시드 여러 개와 마지막 여러 점의 평균으로 보고함
- 우리는 이 넷이 모두 반대임
  - ±120° 절대 침로를 0.4 s마다 새로 뽑음. 이 잡음이 10° 포화를 넘어서 대부분 전타가 됨
  - 엔트로피 0.01이 σ를 계속 위로 밂. 9.04M 시점 σ가 약 50°임
  - OFF는 56 m만 보는데 위험 비용은 300 m까지 계산함
  - 표본 행동 평가, 끝 체크포인트 1점, 시드 1개로 판정함
- 먼저 가져올 것
  - ① 결과를 안 바꾸는 계측(KL·clipfrac·logstd, 통신 KL)
  - ② 평균 행동 평가 병기와 last-K 평균. 기존 체크포인트로 학습 없이 할 수 있음
  - ③ σ 쪽 토글 3개(엔트로피 계수, logstd 하한, 잡음 유지)를 시드 43 trunk로 비교
- 보상 반경 56 m(P6)는 10-13 사전등록 조건(B 확정)을 못 채웠으므로 저자 override가 있어야 진행함
- 행동공간 교체(증분형 침로)와 관측 확장은 마지막 후보임. 체크포인트 호환이 깨짐
- Claude in Chrome은 이번 세션에서 쓸 수 없었음. 원문은 curl·gh·pdftotext로 확인했고, Zotero에는 MCP로 등록함

## 1. 찾은 레퍼런스 표

접근 = 전문 / 초록 / 코드만. 중복 항목은 한 줄로 합쳐 키를 모두 적음.

| 주제 | 인용 | DOI / arXiv | 코드 | 접근 | Zotero |
|---|---|---|---|---|---|
| 선박 | Meyer 2020, COLREG-compliant CA (IEEE Access) | 10.1109/ACCESS.2020.3022600 · 2006.09540 | gym-auv 7956acd | 전문 | N3QZJVJB |
| 선박 | Meyer 2020, Taming an ASV (IEEE Access) | 10.1109/ACCESS.2020.2976586 · 1912.08578 | gym-auv | 전문 | B2EHQQ2E |
| 선박 | Heiberg/Larsen 2022, Risk-based COLREGs (Neural Netw.) | 10.1016/j.neunet.2022.04.008 · 2112.00115 | gym-auv 기반 | 전문 | NUUNU46P |
| 선박 | Waltz & Okhrin 2023, ST-RRL (Neural Netw.) | 10.1016/j.neunet.2023.06.015 · 2211.01004 | TUD_RL 92a34ba | 전문 | IW4JRAM5 |
| 선박·통신 | Wang & Zhao 2025, ship MARL + comm (Ocean Eng.) | 10.1016/j.oceaneng.2024.120244 · 2410.21290 | 없음 | 전문 | JCQP8XZM |
| 선박 | Niu 2023, shared DDQN (JMSE) | 10.3390/jmse11112101 | 없음 | 전문 | HCSK7BCJ |
| 선박 | Wei & Kuo 2022, QMIX (JMSE) | 10.3390/jmse10101431 | 없음 | 전문 | AFEDQUQK |
| 선박·평가 | Krasowski & Althoff 2024, action masking (IEEE T-IV) | 10.1109/TIV.2024.3400597 · 2402.08502 | 없음 | 전문 | 2EBPDXCM |
| 선박 | Cui 2026, Sim2Sea (AAMAS) | 10.65109/CMRP6518 · 2603.04057 | 미확인 | 전문 | GQB7K4R8 |
| 선박 | Hart 2024, Two-step DOA (KBS) | 10.1016/j.knosys.2024.112402 · 2311.16841 | MarWaltz/Two-Step-DOA | 전문 | QKEJGHFZ |
| 선박 | Hart & Okhrin 2024, CR 기반 학습 (Neurocomp.) | 10.1016/j.neucom.2023.127097 · 2212.04123 | 없음 | 전문 | MCSHCWT6 |
| 선박 | Sawada 2021, LSTM-PPO (JMST 26) | 10.1007/s00773-020-00755-0 | 없음 | 전문 | DB3F9VJ4 |
| 로봇 | Long 2018 (ICRA) | 10.1109/ICRA.2018.8461113 · 1709.10082 | Acmece 비공식 c7cb933 | 전문 | CGB6R2UA |
| 로봇 | Fan 2020 (IJRR) | 10.1177/0278364920916531 · 1808.03841 | 없음 | 전문 | CVWSCHGH |
| 로봇 | Everett 2018, GA3C-CADRL (IROS) | 10.1109/IROS.2018.8593871 · 1805.01956 | gym-collision-avoidance 9035640 | 전문 | EJERX8IA |
| 로봇 | Everett 2021 (IEEE Access) | 10.1109/ACCESS.2021.3050338 · 1910.11689 | 위와 같음 | 전문 | ZN5BAII7 |
| 로봇 | Chen 2017, CADRL (ICRA) | 10.1109/ICRA.2017.7989037 · 1609.07845 | 없음 | 전문 | UZRR5QJK |
| 로봇 | Chen 2017, SA-CADRL (IROS) | 10.1109/IROS.2017.8202312 · 1703.08862 | 없음 | 전문 | MF698R8G |
| 로봇 | Semnani 2020 (RA-L) | 10.1109/LRA.2020.2974695 · 2001.06627 | 없음 | 전문 | C3Z2SZU3 |
| 로봇 | Han 2022, RL-RVO (RA-L) | 10.1109/LRA.2022.3161699 · 2203.10229 | rl_rvo_nav 8f3c34c | 전문 | RG5878RP |
| 로봇 | Arul 2022, DMCA (v3) | 2209.06415 | 없음 | 전문 | JV837RZF |
| PPO·로봇 | Yu 2022, MAPPO (NeurIPS D&B) | 2103.01955 (v4) | on-policy de66d7a | 전문 | IW384JC7 |
| PPO | Andrychowicz 2021 (ICLR) | 2006.05990 (v1) | 없음 | 전문 | 8A4A38HT |
| PPO | Engstrom 2020 (ICLR) | 2005.12729 | 없음 | 전문 | WBCXDMTD |
| PPO | Huang 2022, 37 Implementation Details (ICLR blog) | URL만 | 블로그 | 전문 | 없음 |
| PPO | CleanRL ppo_continuous_action.py | — | fe8d8a0 | 코드만 | 없음 |
| PPO·평가 | Stable-Baselines3 | — | b2878f4 | 코드만 | 없음 |
| PPO | Haarnoja 2018, SAC (ICML) | 1801.01290 | 없음 | 전문 | Z42PZXEV |
| PPO | Haarnoja 2018b, SAC Apps | 1812.05905 | 없음 | 전문 | CTMUUBPQ |
| PPO | Raffin 2021, gSDE (CoRL) | 2005.05719 | SB3 | 전문 | TDDW4VJZ |
| PPO | Hsu 2020, Revisiting PPO | 2009.10897 | 없음 | 전문 | KI367GE2 |
| PPO | Seyde 2021, Bang-Bang (NeurIPS) | 2111.02552 | 없음 | 전문 | UUXHM9ZK |
| 통신 | Lowe 2019, Pitfalls (AAMAS) | 1903.05168 | 없음 | 전문 | NDFJ25WK |
| 통신 | Lin 2021, ae-comm (NeurIPS) | 2110.15349 | 없음 | 전문 | U9K96XTK |
| 통신 | Eccles 2019, Biases (NeurIPS) | 1912.05676 | 없음 | 전문 | KHA3R5AW |
| 통신 | Jaques 2019, Social Influence (ICML) | 1810.08647 | 없음 | 전문 | SAI6K87U |
| 통신 | Das 2019, TarMAC (ICML) | 1810.11187 | 없음 | 전문 | DEGADCS5 |
| 통신 | Singh 2019, IC3Net (ICLR) | 1812.09755 | IC3Net | 전문 | WM87QVZN |
| 통신 | Sukhbaatar 2016, CommNet (NIPS) | 1605.07736 | 없음 | 전문 | SCHGW4WE |
| 통신 | Foerster 2016, DIAL (NIPS) | 1605.06676 | 없음 | 전문 | PMPCURBW |
| 통신 | Jiang & Lu 2018, ATOC (NeurIPS) | 1805.07733 | 없음 | 전문 | ZJIDD4QF |
| 통신 | Ding 2020, I2C (NeurIPS) | 2006.06455 | PKU-AI-Edge/I2C | 전문 | AMUVM67V |
| 통신 | Wang 2020, NDQ (ICLR) | 1910.05366 | 없음 | 전문 | G2PCT5R4 |
| 통신 | Wang 2020, IMAC (ICML) | 1911.06992 | 없음 | 전문 | SN48PTRT |
| 통신 | Guo 2024, MAPPO-PIS (ECCV-W) | 2408.06656 | AMAPPO-PIS | 전문 | 9I36WWMD |
| 통신 | Kim 2021, Intention Sharing (ICLR) | OpenReview qpsl2dR9twy | 없음 | **초록만** | 없음 |
| 평가 | Agarwal 2021, Statistical Precipice (NeurIPS) | 2108.13264 | rliable 3ccd9f4 | 전문 | KGQAXXRR |
| 평가 | Colas 2018, How Many Seeds | 1806.08295 | 미열람 | 전문 | C7X6E9C9 |
| 평가 | Colas 2019, Hitchhiker's Guide | 1904.06979 | 없음 | 전문 | ASZWTWDT |
| 평가 | Henderson 2018 (AAAI) | 1709.06560 | 없음 | 전문 | XJFTQKU9 |
| 평가 | Patterson 2024 (JMLR) | 2304.01315 | 없음 | 전문 | WHXFGKJ8 |
| 평가 | Jordan 2020 (ICML) | 2006.16958 | 미열람 | 전문 | G34VZKU7 |
| 평가 | Gorsane 2022, MARL protocol (NeurIPS) | 2209.10485 | 없음 | 전문 | EC332TG7 |
| 평가 | Woerner 2019, COLREGs metrics (Auton. Robots 43(4)) | 10.1007/s10514-018-9765-y | 없음 | 전문(MIT MS) | S6KEJPK6 |
| 평가 | Woerner 2016, MIT 박사논문 | hdl:1721.1/104118 | 없음 | 전문 | 없음 |

- 원문을 못 읽은 비공개 선행 연구: Zhao & Roh 2019 (10.1016/j.oceaneng.2019.106436), Chun 2021, Woo & Kim 2020, Xu 2020·2022. 2차 서술로만 앎. 학교 기관 접근으로 확인할 것을 권함

## 2. 우리 vs 레퍼런스 비교표

| 측면 | 우리 (코드 위치) | 레퍼런스 | 차이 |
|---|---|---|---|
| 행동 | a0×120° 절대 침로 오프셋을 0.4 s마다 새로 표본함. 침로 오차 10°에서 전타로 포화 (config.py:565-568, vessel_gym.py:1103-1111) | 증분형·저주파: Waltz ±5°/3 s, Niu ±10°/30 s + PD, Hart 2.4°/20 s, Krasowski 회전율 7단/10 s, Meyer 타 모멘트(적분형). 절대 명령형은 Sim2Sea뿐(이산 18개 + VO 마스크) | 큰 범위를 짧은 간격으로 독립 재표본하는 조합은 문헌에 없음 |
| std 초기값·경계 | logstd 초기 −1.0(σ≈44°), clamp [−2.3, 0](하한 σ≈12° > 포화 10°), clamp 밖은 기울기 0 (networks.py:705-709, :875, :905) | Andrychowicz: 초기 std에 민감함, 하한은 작게(1e-3), clamp는 straight-through. gSDE PPO 초기 log σ −2(Ant·Hopper는 −1). SB3 PPO 기본 0.0 | 하한이 포화폭보다 커서 '조금만 돌기'를 표본할 수 없음 (Fan 2020 §4.4 '직진 표본 확률≈0'과 같은 구조) |
| squash·엔트로피 | tanh, 엔트로피 = 닫힌형 + 저장 raw로 계산한 squash 항(기울기 0). logstd에 상수 +1 압력, 계수 0.01 (networks.py:915-918, config.py:298) | Andrychowicz B.8: tanh 항은 엔트로피에서만 기울기를 가져야 함. Sec 3.8: 보너스 이득 없음. CleanRL·SB3·baselines·gSDE PPO 모두 ent 0.0. SB3 squash 추정치 = mean(−log π(a_stored)) | 공개 기본 경로에 없는 조합이고, σ가 안 줄어드는 원인과 직결됨 |
| 탐색 잡음 | 매 결정 i.i.d. Gaussian (networks.py:878-879) | gSDE: 저역통과 시스템에서 i.i.d. 잡음은 상쇄됨. n스텝마다 재표집, worker별 잡음 | 타각률 3°/s인 느린 배에서는 회피 방향 탐색이 쌓이지 않음 |
| 결정 간격 | 0.4 s | 3–30 s. 고주파 성공 사례(Meyer 0.14 s)는 행동이 적분형 | 고주파 + 절대 침로 조합은 문헌에 없음 |
| PPO 데이터 사용 | 65,536 결정 / 미니배치 512 × 2 epoch = 256 step. clip 0.2, KL 기록 없음, Adam eps 1e-8 (vessel_gym_train.py:1272-1346) | MAPPO: 미니배치 1–2개, clip < 0.2(이산 행동만 검증). Engstrom: ratio clip이 trust region을 못 지킴 → KL을 잴 것. Huang: approx_kl은 보통 < 0.02 | 미니배치 수가 MAPPO 권고의 64배. drift 진단 로그가 없음 |
| LR | 3e-4 고정. 감쇠 토글은 있으나 단독 시험 없음 | 선형 감쇠가 기본(Huang·CleanRL). Andrychowicz는 5개 과제 중 4개에서 소폭 이득 | 단독 시험만 빠짐 |
| value·adv 정규화 | ValueNorm, adv는 update 전체, 망별 grad clip 0.5, central critic | MAPPO Sug 1·2와 일치 | 차이 거의 없음 |
| 정책 출력층 init | 기본 init × 0.1 (networks.py:701-703) | gain 0.01 (orthogonal 기준이라 대략 10배 차이) | 침로 모드에서는 효과가 작을 것으로 봄 |
| 위험 비용 범위 | #6·#7 300 m, DCPA 게이트 48 m. OFF 관측 56 m (vessel_gym.py:1426-1431, :1569-1576) | Meyer: 센서값으로 벌점. Heiberg: 센서 범위 안만, dU = 탐지 범위. Krasowski: d_obs. RL-RVO: 감지 4 m 안 | 문헌은 보상 위험 = 관측 위험. 우리 OFF만 어긋남 |
| 관측의 위험 특징 | DCPA·TCPA를 명시적으로 주지 않음 | Hart two-step: dCPA·tCPA를 넣으면 충돌 절반. Hart CR: 장애물별 6특징(d, u, θ, θ_obst, DCPA, TCPA) | 정책이 위험량을 학습으로 역산해야 함 |
| 장면 분포 | 16척 respawn, 커리큘럼 없음, 위험 장면 비율 미측정 | Hart CR: 무작위 초기화의 36 %가 위험 0, 24 %가 회피 불가. 로봇 계열은 다들 척수 커리큘럼 사용 | 회피 신호가 얼마나 희석되는지 모름 |
| 평가 모드 | 주 평가 = 표본 행동. 평균 행동은 진단용 eval_mean.py로만 잼(평균 54.6 vs 표본 28.8) | 다수가 평균/결정 행동: Meyer, Long(Acmece), Everett, MAPPO, SB3, SAC, gSDE, Colas 2019. 반대 예 = CleanRL, Andrychowicz | 계보와 비교가 안 됨. Woerner R8 점수가 지터로 오염될 수 있음 |
| 시드·체크포인트 | 시드 1개, 끝 1점, 재분기 잡음 약 20 pt | Colas 2018: 마지막 10점 평균, Welch t. Patterson: 짝 차이 D. Henderson: 같은 설정 5 vs 5에서 p = 0.0016. Krasowski: 10시드 짝 t | 같은 trunk 짝 구조는 이미 있음 → last-K 평균 + 짝 차이로 보강 가능 |
| 통신 도입·측정 | 9.04M OFF trunk에서 분기. **Fig1 배치 워밍업 2400**(run_repro 기본은 1200). 동결 코덱 k=8. act_shuf 등 전 결정 평균 (vessel_gym_train.py:628-638) | Lowe: 인과 KL 권장. Jaques: 영향은 timestep의 10 % 미만이라 구간 한정 필요. Lin ae-comm·I2C: 동결/2단계 설계를 지지함 | KL 지표와 조우 구간 한정이 없음. trunk가 0 입력을 9M 동안 봤으므로 Lowe의 0-입력 분포 경고는 약하게만 적용됨 |

## 3. 포팅 항목 (우선순위 순)

진단 원인: (1) 침로 표본 잡음·σ가 안 줄어듦, (2) OFF가 못 보는 300 m 위험 비용과 평균 정책의 직진 drift, (3) 평가 잡음(표본 평가·끝 1점·1시드).
승인 등급: **구현 수정** = 결과 불변, 알리고 진행 / **학습 설정** = 저자 결정 / **논문 내용** = 저자 승인 + 사전등록.

### P1. 평균 행동 평가 병기
- 근거: Meyer run.py:567, Long(Acmece) circle_test.py:58, Everett 2018 §IV-A·2021 §III-D, MAPPO mpe_runner.py:154, SB3 evaluation.py:16, SAC Sec 5, gSDE Sec 4.1, Colas 2019 Sec 5
- 겨냥: 원인 (3), 그리고 원인 (1)의 잡음이 지표에 섞이는 문제
- 변경(리뷰 정정): **작은 래퍼가 필요함.** 주 평가 `eval_all`(_run_f.sh:332-344)은 `eval_scripted.py --policy learned`로 돌고 Woerner 줄을 덧붙임. eval_mean.py는 eval_ckpt.main()만 감싸서 Woerner·R8 열이 빠짐. 그래서 eval_scripted learned 경로에 tanh(mean) 패치를 얹어야 함. 출력 파일 이름은 summarize_fig1의 glob `lr_{pre}{arm}_s*.txt`와 최선 OFF 풀에 섞이지 않게 정함
- 토글: `VESSEL_F_EVAL_MEAN=1`(기본 0, 현행 그대로)
- 등급: 논문 내용(평가 프로토콜)
- 체크포인트: 영향 없음
- 검사: 미설정이면 산출물이 그대로임. 같은 seed로 두 번 돌려 값이 같음. Windows에서 qb_ 체크포인트로 두 열 비교(학습 없음)
- 위험: 결과를 본 뒤 주 지표를 고르면 rigging이 됨 → 주 지표를 먼저 사전등록

### P2. update별 approx_kl·clipfrac·전문가별 logstd 기록
- 근거: Engstrom 2020 Sec 4, Huang 37 'Check KL', CleanRL :272, SB3 ppo.py:262-271, Hsu 2020 failure mode 1
- 겨냥: 원인 (2) drift인지 과대 step인지 구분, 원인 (1) σ 추이
- 변경: 미니배치 루프에서 `no_grad`로 kl·clipfrac를 누적함. **출력은 PPO epoch 루프가 끝난 뒤(:1346 이후)에 새 `[ppo]` 줄로 함.** EVb 블록(:1238)은 루프 앞이라 거기서는 못 찍음(리뷰 정정)
- 토글: 기존 `VESSEL_GRAD_TELEMETRY=1`일 때만
- 등급: 구현 수정
- 체크포인트: 영향 없음
- 검사: test_golden --check 3케이스 PASS. 텔레메트리를 켠 비교는 env를 따로 주입해야 함(test_golden이 VESSEL_*를 지움). epoch0 첫 미니배치에서 kl≈0, clipfrac=0. diag_trunk.sh의 EVb grep에 영향 없음
- 위험: 거의 없음

### P3. 엔트로피 계수 토글 `VESSEL_ENTROPY_BONUS`
- 근거: Andrychowicz Sec 3.8·B.8, CleanRL·SB3·baselines·gSDE ent 0.0, Long(Acmece) 5e-4(우리와 같은 상수 기울기 구조)
- 겨냥: 원인 (1). logstd에 늘 +1 압력이 걸려 9.04M logstd가 −0.869임
- 변경: config.py:298을 env로 바꿈. _run_f.sh:93·:105 허용 목록, _cur_opt(:801), check_branch(:57)에 추가. **ckpt_io.snapshot_config에 `entropy_bonus` 추가(리뷰 정정: gym 스냅샷에 없음. 그대로 두면 대조 검사가 늘 None이라 아무것도 못 막음)**
- 토글: 기본 0.01(비트동일). 시험값 0 또는 5e-4
- 등급: 학습 설정
- 체크포인트: 키·shape 불변. 두 갈래 같은 값이어야 함
- 검사: golden PASS. 0일 때 logstd.grad == 정책 손실 단독 기울기
- 위험: 하한 −2.3 때문에 σ 12° 아래로는 못 내려감 → P4와 함께 봐야 효과가 드러남

### P4. logstd 하한 토글 + straight-through clamp
- 근거: Andrychowicz C62·B.8 fn25, gSDE Table 10(초기 −2), Fan 2020 §4.4
- 겨냥: 원인 (1). 하한 12°가 포화 10°보다 큼(포화 밖으로 덜 나가려면 logstd < ln(10/120) = −2.48)
- 변경: `LOGSTD_MIN`을 networks.py:875·:905 두 곳에 같은 값으로 적용. ST 옵션도 두 곳 같게. config.py:496 assert 완화, _verify_ppo_mirror.py:342 하드코딩 제거. **ckpt_io.restore_policy가 스냅샷 `logstd_min`을 되돌려야 함(리뷰 정정).** 표본 평가는 forward clamp가 σ를 정하므로, 이게 없으면 −3.0으로 학습한 모델을 −2.3으로 조용히 평가하게 됨
- 토글: `VESSEL_LOGSTD_MIN` 기본 −2.3, `VESSEL_LOGSTD_CLAMP_ST` 기본 0. 시험값 −3.0(약 6°)
- 등급: 학습 설정
- 체크포인트: 키 불변. 스냅샷에 키 2개 추가, _cur_opt에도 넣음
- 검사: golden PASS, 미러 검증기 2종 ALL PASS(Windows). forward와 get_logprob_entropy의 logprob 비트 일치. ST=1이고 logstd = 0.5일 때 grad ≠ 0
- 위험: σ가 너무 작으면 회피 탐색이 사라짐 → P2 kl 감시 필수. **ST=1을 엔트로피 0.01('closed')과 같이 켜지 말 것**(원시 파라미터가 0 위로 계속 커짐)

### P5. 잡음 n결정 유지 `VESSEL_NOISE_HOLD`
- 근거: gSDE Sec 1·4.2·A.6·Table 4, Sawada §6. Seyde Sec 5.3 위치는 **미확인**
- 겨냥: 원인 (1). 느린 배에서 i.i.d. 잡음이 상쇄됨
- 변경: ControlActor.forward에 `eps` 인자를 추가함(None이면 현행). rollout(:1160)이 배별 eps를 들고 있다가 n결정마다, 또는 done인 배만 재표집함
- 토글: 기본 0(비트동일). 시험 4(1.6 s), 8(3.2 s)
- 등급: 학습 설정
- 체크포인트: 영향 없음. 스냅샷 `noise_hold` 추가
- 검사: golden PASS. 같은 배에서 4결정 동안 eps가 같고 done 직후 바뀜. epoch0 ratio = 1
- 위험(리뷰 정정): 유지된 ε는 상태 조건부로 N(0,1)이 아님. 그래서 중요도 비율이 **편향될 수 있는 근사**임(gSDE·SB3도 감수하는 근사). epoch0 테스트로는 못 잡음. n은 하나씩만 시험

### P6. 위험 비용 반경 56 m (두 팔 동일) 실행
- 근거: Meyer rewarder.py:152-156·:206, Heiberg Eq.18·Fig.7, Krasowski §VI-B, RL-RVO, CADRL 계열. Waltz는 verify에서 CR이 시야 검사 없이 계산된다고 확인돼 근거에서 뺌
- 겨냥: 원인 (2)
- 변경: 없음. b6a4341의 `VESSEL_F_REWARD_RANGE=56` 경로(보상 #6·#7만 56 m, 판정기 300 m 유지, 순위 관문 v5)를 실행함
- 등급: 논문 내용. **10-13 사전등록에서 단계 2는 'B 확정' 조건부였는데 결과가 미확정(47.1 < 50)이었음 → 실행하려면 저자 override를 명시적으로 받아야 함**(리뷰 지적)
- 체크포인트: 새 trunk 필요
- 검사: 순위 관문 v5 → qc_ trunk 시드 43 → trunk 관문 → off·comm, P1 열과 함께 보고
- 위험: comm의 300 m 정보가 보상과 연결되지 않아 H1b가 약해질 수 있음. OFF를 불구화하는 것은 아니므로 anti-rigging 원칙에는 맞음

### P7. 미니배치 확대 + 스냅샷 기록
- 근거: MAPPO Sug 3·5(이산 행동만 검증), Andrychowicz C4
- 겨냥: 원인 (2) drift·비정상성
- 변경: env는 이미 있음(config.py:296). `minibatch_size`·`n_epoch`를 스냅샷·_cur_opt·check_branch·_run_f.sh 허용 목록에 추가. **골든 regen은 필요 없음**(test_golden이 키 추가를 허용함, 리뷰 정정)
- 토글: 기본 512. 시험 8192(N_EPOCH 2에서 update당 16 step)
- 등급: 학습 설정
- 검사: golden PASS. Windows에서 VRAM 피크 측정 후 trunk 75분
- 위험: GPU 메모리(qa_가 15.7 GB까지 불어난 이력). N_EPOCH를 같이 바꿀지 사전에 정함. P2에서 drift 증거가 나올 때만 시험

### P8. last-K 체크포인트 평균 + 시드별 짝 차이
- 근거: Colas 2018(마지막 10점 평균), MAPPO(마지막 10 평가 중앙값), Patterson Sec 4.4(D = A − B), Henderson Fig 5, Colas 2019(ε별 N, Bonferroni). Agarwal의 'Sec 3 max 금지' 절 번호는 **미확인**(인용 전 확인 필요)
- 겨냥: 원인 (3)
- 변경: step10/12/14/16M 갈래 체크포인트를 전부 평가해 평균을 냄(best·max 선택 금지). 시드별 D와 승패 수를 적고, 시드 ≥ 3이면 짝 t-CI를 붙임. step16M은 끝 모델과 같은 파일이라 한 번만 셈. 출력 파일 이름 주의는 P1과 같음
- 토글: `VESSEL_F_LASTK=1`
- 등급: 논문 내용
- 검사: qb_ 기존 체크포인트로 무학습 재평가 → 순위 안정성 확인
- 위험: 평가 시간 약 4배. 1시드는 여전히 단독 주장 금지

### P9. 통신 KL(CIC 연속판), 조우 구간 한정
- 근거: Lowe §3.1·§6.2, Jaques §5 Eq.2·§5.1, I2C §3.1
- 겨냥: 통신 사용 판정. 지금 act_shuf는 전 결정 평균이라 희석됨
- 변경(리뷰 정정 반영):
  - a_base·a_s를 forward와 같게 clamp(mean ±3, logstd [하한, 0])한 뒤 KL = Σ(Δμ)²/(2σ²)로 계산함
  - 마스크 = sit≠0 또는 56–300 m 파트너가 있는 배
  - **새 열은 COMM_TELE_COLS 끝이 아니라 별도 튜플로 EXT 열 뒤에 붙임**(안 그러면 EXT CSV 열 위치가 밀림)
  - 위치는 comm_telemetry :527, act_* :628-638
- 토글: 학습기에서는 `VESSEL_COMM_TELEMETRY=1`일 때만. **diag_ckpt.py:143은 늘 호출하므로 diag 출력에는 열이 추가됨**
- 등급: 구현 수정(논문 지표로 쓰려면 별도 승인)
- 검사: 메시지를 그대로 넣으면 KL = 0. **fc2 메시지 슬라이스 가중치를 0으로 둔 정책에서 KL = 0**('게이트 0'이 아님. 게이트 0은 sigmoid 0.5 중립임)
- 위험: Gaussian으로 옮긴 것은 우리 가정(원 실험은 이산 행렬게임)

### P10. squash 엔트로피 추정 방식 토글
- 근거: Andrychowicz B.8, SB3 distributions.py:244-247·ppo.py:247-252(추정치 = mean(−log π(a_stored))), SAC 2018b Sec 4
- 겨냥: 원인 (1). 현재 squash 항은 기울기 0이라 제동 역할을 못 함
- 변경: networks.py:915-918에 'closed'(기본) / 'logp' / 'reparam' 분기 추가
- 등급: **학습 설정**(리뷰 정정. 켜면 목적함수가 바뀜)
- 체크포인트: 키 불변, 스냅샷 `entropy_est` 추가
- 위험·판단: 'logp'의 logσ 기울기는 기댓값 0이라 기댓값으로는 P3(계수 0)과 같음 → P3 결과를 본 뒤 'reparam'만 따로 검토

### P11. LR 선형 감쇠 단독 시험
- 근거: Andrychowicz C31, Huang core #4, CleanRL anneal_lr, Engstrom Fig 1
- 겨냥: 원인 (2) 후반 V자. qb_에서는 logstd −2.0과 섞여 단독 효과를 모름
- 변경: 없음. `VESSEL_LR_DECAY_DEC`·`VESSEL_LR_FINAL`만 줌(config.py:497-499 assert)
- 등급: 학습 설정
- 위험: 원인을 고치지 않고 가리기만 할 수 있음

### P12. 정책 출력층 init 0.01 토글
- 근거: Andrychowicz C57, Huang core #2, CleanRL :127, SB3 :617-622
- 변경: networks.py:702 `mul_(0.1)` → `cfg.ACT_MEAN_INIT_SCALE`
- 등급: 학습 설정. 새 trunk에서만 의미 있음
- 위험: 침로 모드에서는 mean≈0이 이미 직진이라 효과가 작을 것 → 우선순위 낮음

### P13. 학습 장면 위험 분포 측정
- 근거: Hart CR-train §6.2, Semnani §III-C
- 겨냥: 원인 (2). 회피 신호가 희석되는지 확인
- 변경: diag_ckpt에서 risk_rw > 0 비율과 회피 가능 위험 비율을 집계함(vessel_gym.py:1436-1440 읽기만)
- 등급: 구현 수정. 단 diag_ckpt.py는 '학습기 본체(수정 전 담당 확인)' 목록에 있음
- 검사: n_ trunk 9.04M로 1회 측정. 커리큘럼은 결과를 본 뒤 별도로 제안

### P14. 증분형 침로 명령 `ACTION_MODE=course_inc`
- 근거: Waltz §4.3, Niu §3.7 Eq.9, Hart two-step §7.3, Sawada §4.1·§6, Krasowski §V-C
- 겨냥: 원인 (1) 구조적 해결
- 변경: env가 plan_course를 상태로 들고 a0·Δmax만큼 누적함. obs self에 (ψ_cmd − heading) 1차원 추가. **동시 수정 목록**(리뷰 보강):
  - config OBSERVATION_SIZE·SELF_SIZE
  - parse_obs
  - networks fc2 입력
  - Unity VesselAgent.cs·obs_utils(369D 빌드 트랩)
  - comm_codec p46 intent
  - check_reward_rank 행동 모드
- 등급: 논문 내용
- 체크포인트: 전부 비호환. 새 trunk·코덱·골든 케이스 필요
- 위험: 가장 큰 변경. P3–P5가 실패한 경우에만

### P15. ARPA 토큰에 쌍별 DCPA·TCPA 추가
- 근거: Hart two-step Abstract·§5.2, Hart CR §5.1, Sawada OZT
- 겨냥: 원인 (2)
- 변경(리뷰 정정): gym 경로에서는 `comm_pair_features`(vessel_gym.py:302)가 한 번 계산해 prel로 저장하고, update는 저장값을 씀 → **세 곳 미러 위험은 해당 없음.** 실제 수정 지점은 comm_pair_features·COMM_EXT_DIM, attn k/v·msg_encoder 입력(networks.py:1078). Unity 경로는 relpos_dim≠3이면 거부함
- 토글: `VESSEL_PAIR_RISK_FEAT=1`
- 등급: 논문 내용. 새 trunk 필요
- 위험: P6 결과를 본 뒤에만

### P16. target_kl 조기 종료 / clip 토글
- 근거: MAPPO Sug 4, Huang aux #3, CleanRL :307, SB3 ppo.py:262-271, Hsu App C.2
- 변경: `VESSEL_PPO_CLIP`(기본 0.2), `VESSEL_TARGET_KL`(기본 0). **`epsilon`·`target_kl`을 gym 스냅샷과 _cur_opt에 추가**(리뷰 정정: 현재 없음). 판정 방식을 명시함: SB3 = 현재 미니배치 kl > 1.5×target이면 break / CleanRL = epoch 끝 마지막 미니배치 값
- 등급: 학습 설정
- 위험: P2가 kl > 0.02를 보일 때만 시험. 증거 없이 켜면 근거 없는 튜닝이 됨

## 4. 추천 실행 순서

| 단계 | 할 일 | 어디서 | 이 단계가 정하는 것 |
|---|---|---|---|
| 1. 구현 | P2·P9·P13 계측. P3·P4·P5·P7·P16 토글은 기본 비트동일로 넣음. 스냅샷 키(entropy_bonus·logstd_min·noise_hold·minibatch_size·n_epoch·epsilon·target_kl)와 restore 경로. P1·P8 평가 래퍼·파일 이름 | Mac (`_dev_dyn`, 브랜치는 저자 지정) | 이후 실험을 '조용히 다른 실험' 없이 돌릴 수 있는지 |
| 2. Mac 검사 | `verify/test_golden.py --check` 3케이스, 텔레메트리 켠 골든 비교, 단위 assert(kl≈0·logstd.grad·hold 유지·KL=0 케이스). 1–2분 단일 프로세스만 | Mac | 기본값 비트동일 확인. 실패하면 머지 금지 |
| 3. Windows 무학습 | 미러 검증기 2종 ALL PASS. qb_ off/comm에 P1 평균 행동 열 + P8 last-K 재평가. qb_ comm에 P9 CIC. n_ trunk에 P13 위험 분포 | Windows | 평균 행동에서 순위가 바뀌는지, comm이 메시지를 읽는지, 장면 희석 정도 → 다음 단계 팔을 고르는 근거 |
| 4. Windows 1시드 trunk | 시드 43, 75분 trunk 동시 3건(q_ 단계 1 형식, 전부 OFF): A = ENT 0 / B = ENT 0 + LOGSTD_MIN −3.0 / C = NOISE_HOLD 4. P2 계측은 모두 켬. **관문·판정 기준은 실행 전에 사전등록** | Windows | trunk 관문(마지막 4창 도착 ≥ 30 %), logstd 추이, 평균−표본 격차 축소 여부 → σ 쪽 처방이 맞는지 |
| 5. 분기 | 4에서 통과한 팔만 off·comm 갈래로. P6(56 m)는 저자 override가 있을 때만 qc_로. P7·P11·P16은 P2 kl 증거가 있을 때만. P14·P15는 마지막 | Windows | 시드 1개 결과로 '시뮬 성공' 여부를 판단(저자). 성공하면 그 뒤에 시드 3개 확증 |

## 5. 안 가져올 것과 이유

- **팔마다 다른 위험 반경(OFF 56 / comm 300)**: 두 팔의 보상 함수가 달라 짝 비교가 무효가 됨 → 두 팔 동일(P6)만 권함
- **VO/TTC 행동 마스크(Sim2Sea, Krasowski)**: 성능을 규칙이 만들게 되어 '모방 아닌 학습'(노벨티 d)과 충돌함. 사후에 마스크만 씌우면 도착이 약 40 pt 떨어짐(Krasowski)
- **hybrid 규칙 조타(Fan 2020, Semnani 2020)**: 팔 구성·평가가 바뀌고 학습 정책이라는 주장이 약해짐
- **positive listening 보조손실(Eccles)·social influence 보상(Jaques)**: comm 팔에만 목적함수가 붙어 비대칭이 됨. Eccles 실험에서 영향 보상의 이득은 없었음
- **코덱 RL 미세조정·IB 정규화(IMAC·NDQ)**: Lin ae-rl-comm에서는 오히려 하락함. 동결 설계가 문헌의 지지를 받음
- **SAC 엔트로피 온도 자동조정**: off-policy 맥락임. Andrychowicz C34에서 PPO 엔트로피 제약은 이득이 없었음
- **Beta 정책(Hsu)**: 경로 전체를 교체해야 함. 균등 초기화는 mode 3 처방이지 drift 처방이 아님
- **상태의존 gSDE 전체 이식**: 분포 클래스·미러·골든을 전부 바꿔야 함. 핵심(재표집 간격)은 P5로 작게 이식
- **결정 간격 연장(SUBSTEPS)**: 시간 벌점·보상 항·판정 타이밍이 같이 바뀌어 보상 정의 변경과 같음. P5가 같은 저역통과 효과를 줌
- **advantage 정규화 범위·GAE λ 0.9·value clip·Adam eps 1e-5**: 영향이 작고 진단 원인과 연결되지 않음. 섞으면 효과 귀속만 흐려짐
- **best 체크포인트 선택(Sim2Sea, Gorsane)**: max 프로토콜이라 비교 불가 → P8 last-K로 대체
- **보상 λ를 관측에 넣는 방식(Meyer Taming)**: 새 실험이고 근거가 정적 장애물뿐임
- **QMIX·MADDPG·DQN 하이퍼파라미터**: 알고리즘이 달라 옮길 값이 없음. Wang & Zhao의 79.3 vs 52.7은 MADDPG vs DDPG라 통신 효과 근거로 인용 금지
- **링크 비용(DMCA)·DRU 잡음(DIAL)·학습 게이트(IC3Net)**: 팔·보상 변경임. IC3Net은 협력 과제에서 게이트를 1로 고정하는 편이 낫다고 함(현행 중립 게이트와 같은 방향)
- **다단계 의도 payload(Kim 2021)**: 초록만 읽어서 설정·수치를 인용할 수 없음

## 6. 저자 결정 필요 항목

1. 주 평가 지표를 평균 행동과 표본 행동 중 무엇으로 할지(P1). 결과 보기 전에 사전등록
2. P6(위험 비용 56 m) override 여부. 10-13 B 확정 조건 미충족 상태임
3. 단계 4 trunk 팔 구성(A/B/C)과 판정 기준 사전등록. 그리고 P3·P4·P5 토글 구현 승인
4. last-K 평균·짝 차이 보고 방식 채택(P8)
5. Woerner 지표 정의: 지연 정규화 /rdetect(2019) vs /(rdetect−rcpa)(박사논문), 큰 변침 문턱 30°(Woerner) vs 20°(Krasowski)
6. VO 마스크·hybrid 기각에 동의하는지(노벨티 d와의 관계)
7. Zotero 유형·날짜 수정(preprint 로 저장된 출판본 등)을 해도 되는지 — 중복·오타는 정리함

## 7. 레퍼런스 검증 기록

- 검증 항목은 주제별 합산 약 450건. **wrong 24건** → 본문에서 빼거나 정정값만 씀
  - 선박 6: Heiberg 타선 모델 위치, Waltz 시야 = 위험 계산 범위(**내용 오류**: CR은 시야 검사 없이 계산됨 → P6 근거에서 제외), Waltz Star 문제 위치(§6.4), Niu 공유 정책 위치, Niu 결정 간격 위치, Hart CR 관측(**6특징**, θ_obst 누락)
  - 로봇 2: Everett 2018 행동 공간 위치(§III-A), Semnani r_FMP(**세제곱근**)
  - PPO 5: MAPPO SMAC 평가 위치(Sec 4.3), SB3 엔트로피 부호(**추정치 = mean(−log_prob)**), SB3 logstd init(**PPO 경로는 0.0**, −2.0은 미사용 기본값), gSDE PPO 하이퍼파라미터(**Table 11 덮어쓰기**: Ant·Hopper log σ −1), Seyde 성능 인용 위치(Fig 2 caption)
  - 통신 3: ae-comm 메시지 구조 위치(App A.2), I2C δ percentile(**70·80 %가 어느 과제 값인지 표에서 안 나뉨**), IMAC 엔트로피 상한 위치(Prop 3)
  - 평가 8: 전부 절·쪽 위치 오류(Agarwal 2, Colas 2018 2, Patterson 1, Krasowski 3). 값은 맞음
- 포팅 스펙 리뷰: confirmed 9, wrong 9(P1·P3·P4·P5·P9·P10·P15·P16·워밍업 1200→2400, 전부 위 3절에 반영), **미확인 2**(Agarwal 'Sec 3 max 금지' 절 번호, Seyde Sec 5.3 저역통과 서술)
- 초록만 읽은 것: Kim 2021 Intention Sharing(OpenReview PDF가 403). 설정·수치 인용 금지
- 원문 미확보(비공개): Zhao & Roh 2019, Chun 2021, Woo & Kim 2020, Xu 2020·2022
- 그 밖에 미확인
  - 코드 미열람: Colas·Jordan 저장소
  - Sim2Sea 코드 공개 여부
  - gym-auv 논문과 코드 값이 다름(lam·net·sensor_range). 어느 쪽이 논문 결과를 냈는지 모름
  - Niu 식(10)의 수치는 PDF 추출에서 깨짐
  - Wei & Kuo는 계수·성공률 미기재 → 수치 인용 금지
- 도구: **Claude in Chrome은 이 세션에서 쓸 수 없었음.** 원문 확인은 curl·gh·pdftotext로 함
- Zotero 문제
  - 중복 정리함(10-26, 이번 세션이 넣은 항목만): 4RT2TJ9D·XA5F4XZ8·ASWVUCCB·H3MBT4BI 를 휴지통으로 옮김(복구 가능). 남긴 항목 JCQP8XZM·2EBPDXCM·NUUNU46P 를 해당 하위 폴더에 추가
  - 오타 수정함: CGB6R2UA 'Fanl' → Fan ('Rasheeda' 오타는 지운 H3MBT4BI 에 있었음)
  - JV837RZF 제목이 v1임(읽은 판은 v3)
  - 출판본인데 preprint 유형으로 저장됨: Andrychowicz(제목도 다름), MAPPO, Lowe, Jaques, Engstrom, Agarwal, Henderson, Patterson, Jordan, Gorsane 등
  - 날짜가 온라인 선공개일로 들어감: Sawada 2020-8-3, Woerner 2018-5-23(부제도 누락)
  - 미등록: Huang 37 블로그, Woerner 박사논문(DSpace URL로 추가 가능), Kim 2021, CleanRL·SB3 코드
  - Sawada 등 'PDF attached' 응답 뒤에도 첨부가 안 보임(동기화 지연 추정). semantic search DB 갱신은 안 돌림
- 더 볼 만한 누락 문헌: Schulman 2017 PPO, Pardo 2018 Time Limits(우리 TIMEOUT_BOOTSTRAP 0 + 시간 벌점 조합과 연결), Eberhard 2023 Pink Noise(P5 근거), Chou 2017 Beta, Zhao & Roh 2019(공유 정책 다선박 원조), Chan 2020 신뢰도 지표