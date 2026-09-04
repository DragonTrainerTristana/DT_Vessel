"""
프리즈-정책 완주 평가 (2026-08) — 학습 로그의 per-window goal%가 '동기화 종료 파도 × 고정-decision 창'
aliasing으로 요동하는 문제를 우회. 체크포인트를 로드해 다수 에피소드를 끝까지 돌려 outcome을 pooling.

핵심: 위상 dephase — 각 배마다 랜덤 burn-in(0~MAX_EP) 후부터 집계 시작 → 종료 파도 smear.
     충분히 길게(에이전트당 여러 에피소드) 돌려 안정적 outcome 분포 획득.

사용: VESSEL_MSG_DIM=6 VESSEL_USE_MOE=1 python eval_ckpt.py --ckpt vg_OFF_s43.pt --arm OFF
     (arm/dim/moe는 체크포인트 학습 시와 동일 env로 맞춰야 함)
     GPU 지정: --device cuda:1  (독립 프로세스 병렬로 여러 개 돌릴 땐 프로세스마다 다르게 줄 것)
"""
import os, sys, argparse
import numpy as _np
import torch
import config as cfg
import vessel_gym as vg
import networks as net
from networks import CNNPolicy
from vessel_gym_train import comm_gather, parse_obs, FrameStack, make_others_msg


def main():
    # ★2026-09-05 fix: 기본 콘솔(cp949)에서 print 한 줄 때문에 평가 전체가 죽던 것 방지.
    #   실측: msg_dim 스니핑 print 의 em dash 에서 UnicodeEncodeError → 몇 시간짜리 평가가
    #   결과 출력 직전에 통째로 날아갈 수 있음. 인코딩은 그대로 두고 에러 처리만 replace 로
    #   바꾸므로 숫자·동작은 불변(못 찍는 글자만 '?' 가 됨).
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--arm', default='OFF', choices=['OFF', 'ORACLE', 'ON', 'RANDOM'])
    ap.add_argument('--envs', type=int, default=256)
    ap.add_argument('--vessels', type=int, default=16)
    ap.add_argument('--max_partners', type=int, default=cfg.MAX_COMM_PARTNERS)
    # ★2026-08-27 재산정: MAX_EP 가 4500결정이 됨. 평균 에피소드는 ~1424결정(도착이 대부분)이라
    #   10000 이면 에이전트당 ~7 에피소드 완주, 그중 첫 1개는 아래 counted 게이트로 제외 → 실집계 ~6.
    ap.add_argument('--eval_decisions', type=int, default=10000)  # 에이전트당 ~6 에피소드 집계
    # burn-in 은 '스폰 직후 인위적 상태'를 흘려보내는 용도. 통계 정확성은 burnin 길이가 아니라
    # counted 게이트가 보장하므로(아래) 과하게 길 필요 없음.
    ap.add_argument('--burnin', type=int, default=1200)           # 초기 transient flush
    ap.add_argument('--seed', type=int, default=999)             # eval seed(학습과 분리)
    ap.add_argument('--ring', type=float, default=1.0)           # 스폰 링 스케일(학습과 동일해야 함). ★0.7→1.0
    ap.add_argument('--crossing', type=int, default=0)           # 2=대척 / 그 외=최소거리 랜덤 (학습과 동일해야 함)
    # ★ckpt 의 학습 arm 과 --arm 이 다르면 기본은 중단. 의도한 교차평가(예: ON 정책의 통신을 끊어 재기)만 이 플래그로 허용.
    ap.add_argument('--allow_arm_mismatch', action='store_true')
    # ★2026-09-05 fix: GPU 인덱스를 고를 수단이 없어 항상 cuda:0 에 몰렸음.
    #   무위험 속도개선이 '독립 프로세스 병렬(이 머신 ~6개)'인데, 6개가 전부 물리 GPU0 에
    #   4096 에이전트씩 올라가 메모리 경합·OOM 또는 직렬화된 속도가 됨. 미지정이면 기존 동작 그대로.
    ap.add_argument('--device', default=None, help='예: cuda:1 (미지정이면 cuda / cpu)')
    # ★2026-09-05 fix: 평가 창 끝 절단(censoring) 보정 — opt-in. 0=기존 동작(과거 숫자 재현).
    #   창 끝에 걸린 에피소드는 길이에 비례해 뽑히므로(length-biased) 긴 에피소드=timeout 이 더
    #   많이 잘려 TO% 과소·goal% 과대가 됨. 기본값을 바꾸면 과거 보고 숫자가 조용히 달라지므로
    #   기본은 끄고, 대신 아래에서 '잘린 개수'를 항상 출력함.
    ap.add_argument('--drain', type=int, default=0,
                    help='창 종료 후 진행 중 에피소드를 마감하는 데 쓸 최대 결정 수 (0=끄기, 기존 동작)')
    args = ap.parse_args()

    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    E, N = args.envs, args.vessels
    torch.manual_seed(args.seed)
    scr = os.path.dirname(os.path.abspath(__file__))
    # 상대 경로로 넘기면 VESSEL_CKPT_DIR(기본: 이 파일 옆 checkpoints/)에서 찾는다.
    ckpt_dir = os.environ.get('VESSEL_CKPT_DIR', os.path.join(scr, 'checkpoints'))
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ckpt_dir, args.ckpt)

    # ★2026-09-05 fix: 보상 계수를 리터럴로 박아두면 학습과 조용히 갈림.
    #   학습(vessel_gym_train.py:390~392)은 VESSEL_FARFIELD_COEF / VESSEL_PERPAIR_COEF 를 env 로 읽는데
    #   평가만 리터럴이라, 그 env 로 스윕한 ckpt 를 평가하면 *학습과 다른 보상함수*로 채점됨
    #   (outcome 은 살아남지만 epReward 비교·리턴 보고가 학습 목표와 다른 함수가 됨). 에러도 안 남.
    #   → 이름·기본값을 학습과 동일하게 맞추고, 기본값과 다르면 stdout 에 찍어 눈에 보이게 함.
    #   perpair_exp 는 학습이 리터럴 3.0 이므로 여기서도 리터럴 유지(eval 에만 env 를 열면 반대로 갈림).
    _ff = float(os.environ.get('VESSEL_FARFIELD_COEF', '0.0'))
    _pp = float(os.environ.get('VESSEL_PERPAIR_COEF', '-0.15'))
    if (_ff, _pp) != (0.0, -0.15):
        print(f"[eval] 보상계수 env override: farfield={_ff} perpair={_pp} "
              f"(학습 때와 같은 env 인지 확인할 것)", flush=True)
    env = vg.VesselBatchEnv(num_envs=E, n_vessels=N, device=dev, seed=args.seed,
                            ring_scale=args.ring, crossing=args.crossing, risk_range=vg.COMM_RANGE, reward_range=vg.COMM_RANGE,   # ★2026-08-30 학습과 동일 반경(200m)
                            farfield_coef=_ff, perpair_coef=_pp, perpair_exp=3.0)
    # ★ckpt 를 먼저 읽어 msg_ln(LayerNorm, 2026-08-31) 유무를 스니핑 → 옛/새 체크포인트 모두 strict 로드
    sd = torch.load(ckpt_path, map_location=dev)
    _sd = sd['model_state_dict'] if 'model_state_dict' in sd else sd
    os.environ['VESSEL_MSG_LN'] = '1' if any('msg_ln' in k for k in _sd) else '0'
    # ★msg_dim 도 ckpt 에서 스니핑 (2026-08-31): Fig4 는 MSG_DIM 2·4·6·8·10·12 를 훑는데
    #   cfg.MSG_DIM 은 `import config` 시점에 고정된다. VESSEL_MSG_DIM 을 안 주면 6 이외
    #   차원은 load_state_dict 에서 shape 불일치로 죽는다(6개 팔 중 4개). ckpt 가 진실이므로 거기서 읽는다.
    #   ⚠️make_others_msg(OFF/ORACLE 팔)는 vessel_gym_train 의 *모듈 전역* MSG_DIM 을 쓰므로 같이 맞춘다.
    #     안 맞추면 others_msg 폭이 틀린 채 ctr_actor 에 들어가 조용히 잘못된 평가가 된다.
    _mk = [k for k in _sd if k.endswith('msg_out.weight')]
    _msg_dim = int(_sd[_mk[0]].shape[0]) if _mk else cfg.MSG_DIM
    if _msg_dim != cfg.MSG_DIM:
        import vessel_gym_train as _vgt
        _vgt.MSG_DIM = _msg_dim
        print(f"[eval] ckpt msg_dim={_msg_dim} (cfg={cfg.MSG_DIM}) - ckpt 값으로 로드", flush=True)

    # ★2026-09-05 fix (blocker): 집계 방식 불일치가 조용히 통과하던 것을 막는다.
    #   문제: CNNPolicy.__init__ 은 msg_encoder(pos_ground용)와 attn(attention용)을 *조건 없이* 항상 만든다.
    #     그래서 USE_ATTENTION=1 로 학습한 체크포인트를 VESSEL_USE_ATTENTION 없이 평가해도
    #     load_state_dict(strict=True) 가 **경고 없이 통과**한다. 가중치는 다 맞는데 self.use_attention 만
    #     False 라, 학습 때와 *다른 집계 함수*(attention → pos_ground mean)로 정책이 굴러간다.
    #     comm-OFF 팔은 others_msg≡0 이라 영향이 없고 comm-ON 팔만 망가지는 비대칭이라 더 위험하다.
    #     msg_dim·msg_ln 은 키에서 스니핑되지만 집계 방식은 키로 구분이 불가능하다(둘 다 항상 존재).
    #   해결: 학습기가 체크포인트에 설정 스냅샷('cfg_snapshot')을 저장하고, 평가는 그걸 읽어
    #     networks 모듈 전역을 덮어쓴 뒤 CNNPolicy 를 만든다(전역은 __init__ 호출 시점에 읽히므로 유효).
    #     스냅샷이 없는 구 체크포인트는 키로 스니핑 가능한 것만 맞추고 나머지는 크게 경고한다.
    _snap = sd.get('cfg_snapshot') if isinstance(sd, dict) else None
    _sniff_cc = any(k.startswith('critic.') and 'glob_enc' in k for k in _sd)
    _sniff_sr = any(k.startswith('state_recon') for k in _sd)
    if _snap:
        net.USE_ATTENTION = bool(_snap.get('use_attention', net.USE_ATTENTION))
        net.POS_GROUND = bool(_snap.get('pos_ground', net.POS_GROUND))
        net.CENTRAL_CRITIC = bool(_snap.get('central_critic', _sniff_cc))
        net.STATE_RECON_COEF = float(_snap.get('state_recon_coef', 1.0 if _sniff_sr else 0.0))
        print(f"[eval] ckpt 설정 적용: attention={net.USE_ATTENTION} pos_ground={net.POS_GROUND} "
              f"central_critic={net.CENTRAL_CRITIC} state_recon={net.STATE_RECON_COEF} "
              f"msg_dim={_msg_dim}", flush=True)
        # 학습 arm 과 평가 arm 이 어긋나면 다른 실험을 재는 것이므로 즉시 실패시킨다.
        _ck_arm = _snap.get('arm') or (sd.get('arm') if isinstance(sd, dict) else None)
        if _ck_arm and _ck_arm != args.arm and not args.allow_arm_mismatch:
            raise SystemExit(f"[eval] ckpt 는 --arm {_ck_arm} 로 학습됐는데 평가는 --arm {args.arm} 임. "
                             f"다른 실험을 재게 되므로 중단함 (의도한 교차평가면 --allow_arm_mismatch).")
    else:
        net.CENTRAL_CRITIC = _sniff_cc
        net.STATE_RECON_COEF = 1.0 if _sniff_sr else 0.0
        print(f"[eval] [!] 이 체크포인트엔 cfg_snapshot 이 없음(2026-09-05 이전 학습). "
              f"central_critic={_sniff_cc} state_recon={_sniff_sr} 는 키로 스니핑했으나 "
              f"**집계 방식(attention/pos_ground)은 키로 알 수 없음** — 현재 env 값 "
              f"attention={net.USE_ATTENTION} pos_ground={net.POS_GROUND} 로 평가함. "
              f"학습 때와 다르면 조용히 틀린 숫자가 나오므로 학습 env 를 그대로 주고 돌릴 것.", flush=True)

    policy = CNNPolicy(_msg_dim, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES).to(dev)
    policy.load_state_dict(_sd)
    policy.eval()

    fs = FrameStack(E, N, dev)
    obs = env.reset()
    radar, goal, self_s, sit = parse_obs(obs); fs.reset_all(radar)

    counts = torch.zeros(5, device=dev)   # RUNNING/GOAL/vColl/oColl/TO
    total = 0

    def act(x, goal, self_s, sit):
        with torch.no_grad():
            if args.arm == 'ON':
                om, _ = comm_gather(policy, env, x, goal, self_s, sit, args.max_partners)
            else:
                om = make_others_msg(env, args.arm, E, N, dev)
            action, _, _, _ = policy.ctr_actor(x, goal, self_s, om, sit)
        return action

    import math, time
    t0 = time.time()
    DEG = vg.DEG
    BIG = 1e9
    eye = torch.eye(N, device=dev).unsqueeze(0) * BIG

    def wrap180(x):
        return (x + 180.0) % 360.0 - 180.0

    def nearest_sep():
        """각 배의 가장 가까운 타선 거리 [E,N] (step 전 위치 기준, 안전마진 지표)."""
        d = torch.cdist(env.pos, env.pos) + eye
        return d.min(dim=-1).values

    # 에피소드별 누적기 [E,N] (★step 전 유효 상태로 누적 → respawn 텔레포트 오염 없음)
    ep_fuel = torch.zeros(E, N, device=dev)
    ep_head = torch.zeros(E, N, device=dev)      # 총 heading 변화(궤적 굽음 = 덜 부드러움)
    ep_len = torch.zeros(E, N, device=dev)
    ep_minsep = torch.full((E, N), BIG, device=dev)   # 에피소드 중 최소 선박간 거리
    ep_reward = torch.zeros(E, N, device=dev)    # ★에피소드 누적 보상(연속 지표 — 효과크기 측정력↑)
    prev_head = env.heading.clone()
    # outcome별 지표 합계 (goal 에피소드 위주 비교)
    msum = {oc: dict(fuel=0.0, head=0.0, minsep=0.0, length=0.0, reward=0.0, n=0) for oc in range(1, 5)}
    all_minsep_sum = 0.0; all_minsep_n = 0
    # ★COLREGs 준수도 (보상 항과 동일 정의): max_risk>0.3 조우에서
    #   HeadOn(1)/GiveWay(3)/Overtaking(4) → 우현 권장 → 위반 = 좌현 변침량
    #   CrossingStandOn(2) → 침로유지 권장 → 위반 = |변침량|
    #   compliance = 1 - 위반 (0~1). 게이트 밖(위험 없음)은 집계 제외.
    comp_sum = 0.0; comp_n = 0; comp_bin = 0.0
    # ★상황별 준수율 (1=HeadOn, 2=CrossingStandOn, 3=CrossingGiveWay, 4=Overtaking)
    sit_ok = {k: 0.0 for k in (1, 2, 3, 4)}; sit_n = {k: 0 for k in (1, 2, 3, 4)}

    # ─── ★조우(encounter) 단위 COLREGs 준수도 — colregs_compliance_metric.tex 정의 구현 ───
    #   기존 스텝별 지표(colregs/colregsOK)는 *그대로 두고* 추가 산출한다(옛 run 과의 비교 보존).
    #   정의: 같은 COLREGs 상황이 연속 ENC_MIN_STEPS(5) 이상 유지된 구간 = 조우 1건.
    #     R_stb  = |{t : δ̄_t >  ε_s}| / |T_k|      ε_s = 0.05
    #     R_port = |{t : δ̄_t < -ε_p}| / |T_k|      ε_p = 0.10
    #     R_ck   = |{t : |δ̄_t| < ε_c}| / |T_k|     ε_c = 0.15
    #   판정: HeadOn·CrossingGiveWay → R_stb>0.5 ∧ R_port<0.2 | StandOn → R_ck>0.6 | Overtaking → R_stb>0.3
    #   C = 준수 조우 / 전체 조우. 조우 중 상대선과의 최소 통과거리를 함께 기록(tex 말미).
    #   ⚠️tex 는 δ̄ 를 '정규화 타각'이라고만 적어 실제/명령 구분이 없다 → 둘 다 산출해 함께 보고한다.
    #     (슬루 도입 후 둘이 다르다: 실제=상대 배가 보는 것, 명령=정책의 의도)
    EPS_S, EPS_P, EPS_C = 0.05, 0.10, 0.15
    ENC_MIN_STEPS = 5
    enc_sit = torch.zeros(E, N, dtype=torch.long, device=dev)   # burn-in 뒤 재초기화(아래)
    enc_skip = torch.zeros(E, N, dtype=torch.bool, device=dev)
    enc_len = torch.zeros(E, N, device=dev)
    enc_c = {'act': [torch.zeros(E, N, device=dev) for _ in range(3)],    # [stb, port, ck]
             'cmd': [torch.zeros(E, N, device=dev) for _ in range(3)]}
    enc_mind = torch.full((E, N), BIG, device=dev)
    enc_tot = {k: 0 for k in (1, 2, 3, 4)}
    enc_ok = {'act': {k: 0 for k in (1, 2, 3, 4)}, 'cmd': {k: 0 for k in (1, 2, 3, 4)}}
    enc_dsum = {k: 0.0 for k in (1, 2, 3, 4)}

    def enc_finalize(mask):
        """mask [E,N]: 방금 끝난 조우를 tex 기준으로 판정·집계 (5스텝 미만·burn-in 잔여는 제외)."""
        m = mask & (enc_len >= ENC_MIN_STEPS) & (~enc_skip)
        if not bool(m.any()):
            return
        for k in (1, 2, 3, 4):
            mk = m & (enc_sit == k)
            nk = int(mk.sum())
            if nk == 0:
                continue
            enc_tot[k] += nk
            enc_dsum[k] += float(enc_mind[mk].clamp(max=vg.RADAR_RANGE * 4).sum())
            L = enc_len[mk]
            for src in ('act', 'cmd'):
                stb = enc_c[src][0][mk] / L
                port = enc_c[src][1][mk] / L
                ck = enc_c[src][2][mk] / L
                if k in (1, 3):
                    ok = (stb > 0.5) & (port < 0.2)
                elif k == 2:
                    ok = ck > 0.6
                else:
                    ok = stb > 0.3
                enc_ok[src][k] += int(ok.sum())

    # burn-in: 집계 없이 위상 dephase
    for i in range(args.burnin):
        a = act(fs.get(), goal, self_s, sit)
        obs, _, done, _ = env.step(a)
        radar, goal, self_s, sit = parse_obs(obs); fs.push(radar, done)
        prev_head = torch.where(done, env.heading, prev_head)   # respawn 시 prev 갱신
        if i == 200:
            dps = 201 * E * N / (time.time() - t0)
            eta = (args.burnin + args.eval_decisions) * E * N / dps
            print(f"  [{os.path.basename(ckpt_path)}] {dps:.0f} dec/s, ETA ~{eta:.0f}s", flush=True)

    # ★2026-08-27 fix — burn-in 누적기 미리셋 버그.
    #   기존엔 ep_* 누적기를 burn-in *전에* 초기화하고 burn-in 중엔 누적도 리셋도 안 했다.
    #   그 결과 burn-in 경계를 걸치고 있던 에피소드가 '꼬리 구간만'으로 측정돼
    #   len 과소·minSep 과대·fuel/reward 과소 편향이 생겼고, 편향 크기가 에피소드 길이(=시간초과율)에
    #   비례해 팔마다 달랐다. 게다가 reset() 위상분산(step_count=U(0,MAX)) 탓에 '스폰 직후인데 곧
    #   시간초과'인 가짜 에피소드도 섞인다.
    #   → 누적기를 여기서 리셋하고, 경계를 걸친 첫 에피소드는 counted 게이트로 집계에서 제외한다.
    ep_fuel.zero_(); ep_head.zero_(); ep_len.zero_(); ep_reward.zero_()
    ep_minsep.fill_(BIG)
    counted = torch.zeros(E, N, dtype=torch.bool, device=dev)   # 종료를 한 번 본 뒤부터 집계
    # ★조우 누적기도 여기서 초기화 — burn-in 경계를 걸친 조우는 앞부분이 잘려 R_stb/R_ck 가 왜곡되므로
    #   현재 진행 중인 조우를 enc_skip 으로 표시해 첫 1건만 집계에서 뺀다(옛 counted 게이트와 같은 취지).
    enc_sit = env.situation.clone()
    enc_skip = env.situation > 0

    # ─── ★쌍(pair) 단위 조우 지표 (2026-08-31 사용자 승인) ───
    #   조우 = 선박 쌍 (i,j). i 관점 [e,i,j] 와 j 관점 [e,j,i] 를 각자 추적(각자 자기 역할로 판정).
    #   시작: near_risk>0.3 & 유효 상황 2스텝 연속(디바운스). ★역할은 시작 시점에 고정(COLREGs 실무 일치,
    #         argmax 재분류로 조우가 중앙값 1.2초로 조각나던 문제 해결).
    #   종료: CPA 통과(raw_tcpa<0) 3스텝 연속, 또는 거리>70m 5스텝 연속, 또는 어느 한 쪽 에피소드 종료.
    #   판정: tex 임계(R_stb>0.5∧R_port<0.2 / R_ck>0.6 / R_stb>0.3)를 쌍 창에 적용. 실제(슬루) 타각 기준.
    #   기록: 최소 통과거리, 첫 유의미 변침(|δ̄|>0.2)까지 시간, 무변침 여부.
    #   ★임계값은 결과 확인 전 고정(사전등록): 진입 0.3/디바운스 2/종료 3·5스텝/이탈 70m/변침 0.2/최소 5스텝.
    PAIR_ENTRY_RISK, PAIR_DEBOUNCE, PAIR_END_CPA, PAIR_END_FAR = 0.3, 2, 3, 5
    PAIR_FAR_DIST, PAIR_MANEUVER_THR, PAIR_MIN_LEN = 70.0, 0.2, 5
    _z3 = lambda: torch.zeros(E, N, N, device=dev)
    pr_active = torch.zeros(E, N, N, dtype=torch.bool, device=dev)
    pr_role = torch.zeros(E, N, N, dtype=torch.long, device=dev)
    pr_len = _z3(); pr_stb = _z3(); pr_port = _z3(); pr_ck = _z3()
    pr_mind = torch.full((E, N, N), BIG, device=dev)
    pr_fm = torch.full((E, N, N), -1.0, device=dev)     # 첫 유의미 변침 시점(조우 내 스텝, -1=무변침)
    pr_in = _z3(); pr_cpa = _z3(); pr_far = _z3()       # 진입 디바운스 / 종료 연속 카운터
    # ★침로변화 기반 준수 (2026-09-03 추가). 기존 R_stb 지표는 그대로 두고 *병기*한다.
    #   근거: COLREGs Rule 8(b) 는 '다른 선박이 쉽게 알아볼 만큼 큰 *침로 변경*'을 요구하지
    #   '타를 계속 잡고 있을 것'을 요구하지 않는다. 실무의 올바른 회피는 크게 한 번 틀고
    #   새 침로로 안정(steady up)하는 것이라, 타각 유지비율(R_stb>0.5)로는 준수가 위반으로 잡힌다.
    #   실측: 위험구간 우현타 비율이 HeadOn 69.8% / GiveWay 77.5% 인데 R_stb 기준 준수율은 8.7% / 22.1%.
    #   ★문턱을 하나 고르지 않고 여러 값에 대한 곡선으로 낸다(사후선택 방지).
    pr_hd0 = _z3()          # 조우 진입 시점의 자선 침로(deg)
    pr_dpsi = _z3()         # 진입 이후 부호 있는 누적 침로변화(+=우현)
    pr_dpsi_min = _z3()     # 그 누적값의 최솟값(좌현 최대 이탈 추적)
    # ★2026-09-03 수정: 조우 *중 최대 우현 변침*을 기록한다.
    #   기존엔 조우가 끝나는 순간의 Δψ 만 기록해서, 우현으로 크게 틀어 피한 뒤 원침로로
    #   복귀하면 Δψ≈0 → 위반으로 집계됐다. Rule 8(b) 가 요구하는 건 '회피 중 실제로 얼마나
    #   틀었나'지 통과 후 남은 각도가 아니다. 정면조우(서로 벌렸다 복귀)가 특히 과소평가됐다.
    pr_dpsi_max = _z3()     # 조우 중 최대 우현 변침(+)
    # ★2026-09-03 재수정: '목표 방위 대비 편차'로 잰다.
    #   침로변화만 보면 목표를 향해 도는 조타·장애물 우회·다른 배 회피가 전부 섞인다.
    #   실측 배경: headTravel 2005°/1126결정 = 결정당 1.78° → 조우 30~50결정이면 배경만 50~90°.
    #   그 위에서 'Stand-on 5° 이내 유지'를 요구하니 5%가 나왔다(규정 위반이 아니라 배경 잡음).
    #   obs 의 goal angle(자선 기준 목표 방위, ±180°)을 쓰면 '목표로 가는 조타'가 자동으로 빠진다.
    #     offset = -goal_angle  → +면 목표보다 우현으로 벗어나 있음(= 우현 회피 중)
    #   양보역할: 조우 중 max(offset) >= 문턱   /   유지역할: 조우 중 max|offset| <= 문턱
    pr_off_max = _z3()      # 조우 중 최대 우현 오프셋(목표 방위 대비)
    pr_off_abs = _z3()      # 조우 중 최대 |오프셋|
    pr_off_sum = _z3()      # 조우 중 |오프셋| 합
    pr_off_cnt = _z3()      # 조우 스텝 수 -> 평균|오프셋| (창 길이에 불변)
    # null 기준선: 조우가 없는 배의 같은 통계(배경 잡음 크기 확인용)
    null_off_abs = []
    # ★Rule 17 매칭 null 대조 (2026-09-04): '조우가 없는 배'가 같은 길이 창에서 침로를 얼마나 트는가.
    #   배는 조우와 무관하게 목표를 향해 계속 돈다 → 절대 문턱은 그 배경분을 못 뺀다
    #   (실측: 조우 없는 배의 |목표대비 오프셋| 중앙 9.6°, 10° 이내는 51.8%뿐).
    #   Rule 17(a)(i) '침로·속력 유지'는 *변화량* 규정이므로 |Δψ| 로 재고, 같은 길이 창의
    #   무조우 |Δψ| 분포를 기준선으로 삼는다. 길이 버킷별로 모아 길이 교란을 제거한다.
    NULLW_BUCKETS = ((0, 20), (20, 40), (40, 80), (80, 10 ** 9))
    null_dpsi = {i: [] for i in range(len(NULLW_BUCKETS))}   # 버킷 → [|Δψ|, ...]
    nq_h0 = torch.zeros(E, N, device=dev)      # 무조우 연속구간 시작 침로
    nq_len = torch.zeros(E, N, device=dev)     # 그 구간 길이(결정)
    nq_tgt = torch.zeros(E, N, device=dev)     # 이번에 채울 목표 길이(조우 길이 분포에서 표집)
    sd_dpsi = []                               # StandOn 조우의 (|Δψ|, 창길이)
    # ★규칙8 완전판 + 난이도 층화 (2026-09-04, 팔별 결과 보기 전 정의 확정)
    #   COLREGs Rule 8(b) 원문: "alteration of COURSE OR SPEED" — 침로만 보는 건 규정의 절반만 재는 것.
    #   감속 회피(Rule 8(e) '필요시 속력을 줄이거나 정지')는 다물체 상황의 정석 기동인데 침로 지표에선 0점.
    #   또 통신 팔은 위험 조우 자체가 7% 적다(분모가 다름) → 진입 DCPA 로 층화해 같은 난이도끼리 비교.
    pr_sp0 = _z3()          # 조우 진입 시 자선 속력비
    pr_spmin = _z3()        # 조우 중 최저 속력비
    pr_dcpa0 = _z3()        # 진입 시 DCPA(m) — 난이도 층화 키
    r8_res = {k: [] for k in (1, 2, 3, 4)}   # role → [(우현오프셋, 감속률, 진입DCPA), ...]
    DPSI_THRESHOLDS = (5.0, 10.0, 15.0, 20.0, 30.0)
    hd_res = {k: [] for k in (1, 2, 3, 4)}   # role → [순 침로변화(deg), ...]
    _pw0 = env._last_pw
    pr_skip = ((_pw0['near_risk'] > PAIR_ENTRY_RISK) & (_pw0['sit'] > 0)) if _pw0 is not None \
        else torch.zeros(E, N, N, dtype=torch.bool, device=dev)   # burn-in 을 걸친 쌍은 1회 제외
    pair_res = {k: [] for k in (1, 2, 3, 4)}            # role → [(ok, 최소거리, 첫변침스텝|-1), ...]

    def pair_finalize(mask):
        """mask [E,N,N]: 방금 끝난 쌍 조우를 판정·집계 (5스텝 미만·burn-in 잔여 제외)."""
        m = mask & (pr_len >= PAIR_MIN_LEN) & (~pr_skip)
        if not bool(m.any()):
            return
        for k in (1, 2, 3, 4):
            mk = m & (pr_role == k)
            if not bool(mk.any()):
                continue
            L = pr_len[mk]
            stb = pr_stb[mk] / L; port = pr_port[mk] / L; ck = pr_ck[mk] / L
            if k in (1, 3):
                ok = (stb > 0.5) & (port < 0.2)
            elif k == 2:
                ok = ck > 0.6
            else:
                ok = stb > 0.3
            for o, d_, f_ in zip(ok.tolist(), pr_mind[mk].tolist(), pr_fm[mk].tolist()):
                pair_res[k].append((o, d_, f_))
            if k == 2:   # 유지역할(R17): 목표 방위에서 얼마나 벗어났나
                hd_res[k] += (pr_off_sum / pr_off_cnt.clamp(min=1.0))[mk].tolist()
                # ★null 대조용: StandOn 조우의 순 침로변화 |Δψ| 와 창 길이를 같이 남긴다
                sd_dpsi.extend(zip(pr_dpsi[mk].abs().tolist(), pr_len[mk].tolist()))
            else:        # 양보역할: 목표 방위 대비 최대 우현 오프셋
                hd_res[k] += pr_off_max[mk].tolist()
            # ★규칙8: (우현 오프셋, 감속률, 진입DCPA) — 양보역할만 판정에 씀
            _drop = (1.0 - pr_spmin / pr_sp0.clamp(min=1e-3)).clamp(min=0.0)
            r8_res[k] += list(zip(pr_off_max[mk].tolist(), _drop[mk].tolist(), pr_dcpa0[mk].tolist()))

    # 평가: 완주 outcome pooling + 지표
    # ★2026-09-05 fix: 창 끝 절단(censoring).
    #   기존엔 eval_decisions 만큼 돌고 그냥 끝내서, 그 시점 진행 중이던 에피소드가 통째로 사라졌음
    #   (시작 쪽 절단은 counted 게이트로 막았는데 끝 쪽은 무처리). 창 끝에 걸릴 확률은 에피소드
    #   길이에 비례하므로 timeout 처럼 긴 에피소드가 더 많이 잘림 → TO% 과소·goal% 과대.
    #   더구나 편향 크기가 팔의 길이 분포에 비례해 팔마다 달라 두 팔 비교의 부호가 뒤집힐 수 있음.
    #   → 기본값(--drain 0)은 과거 숫자 재현을 위해 그대로 두고, 루프 뒤에 절단량을 항상 출력함.
    #     --drain K 를 주면 창 종료 후 최대 K 결정까지 더 돌려 진행 중이던 에피소드만 마감함.
    _pending = None          # 창 끝 시점에 아직 안 끝난 에이전트 [E,N] (drain 대상)
    _dstep = -1
    while True:
        _dstep += 1
        if _dstep >= args.eval_decisions:
            if args.drain <= 0:
                break
            if _pending is None:
                _pending = ep_len > 0
                print(f"   [drain] 창 끝 미완 {int(_pending.sum())}개 마감 시작 "
                      f"(최대 {args.drain}결정)", flush=True)
            if (not bool(_pending.any())) or (_dstep - args.eval_decisions) >= args.drain:
                break
        # ── step 전 유효 상태로 지표 누적 ──
        a = act(fs.get(), goal, self_s, sit)
        sr = env.speed / torch.clamp(env.max_speed, min=1e-6)
        turn01 = a[..., 0].abs().clamp(0, 1)
        ep_fuel += sr ** 2 + 0.5 * turn01 ** 2                    # 연료 프록시(보상 항과 동일 형태)
        dh = wrap180(env.heading - prev_head).abs()
        ep_head += dh                                            # 방향 변화 누적
        prev_head = env.heading.clone()
        sep = nearest_sep()
        ep_minsep = torch.minimum(ep_minsep, sep)
        ep_len += 1.0
        # ── ★COLREGs 준수도 (step 전 상황·행동으로 판정; 보상 게이트와 동일 max_risk>0.3) ──
        _pw = env._last_pw
        if _pw is not None:
            _mrisk = _pw['risk'].max(dim=-1).values                  # [E,N]
            _sit = env.situation                                     # [E,N] 0~4
            _rud = a[..., 0]                                         # [-1,1] >0=우현
            _gate = (_mrisk > 0.3) & (_sit > 0)                      # 조우 + 위험
            if bool(_gate.any()):
                _star = ((_sit == 1) | (_sit == 3) | (_sit == 4)).to(r_dtype := ep_fuel.dtype)
                _hold = (_sit == 2).to(r_dtype)
                _viol = _star * torch.clamp(-_rud, min=0.0) + _hold * _rud.abs()
                _comp = (1.0 - _viol).clamp(0.0, 1.0)                # 1=완전준수
                comp_sum += float(_comp[_gate].sum())
                comp_bin += float((_viol[_gate] < 0.1).float().sum())
                comp_n += int(_gate.sum())
                for _k in (1, 2, 3, 4):
                    _m = _gate & (_sit == _k)
                    if bool(_m.any()):
                        sit_ok[_k] += float((_viol[_m] < 0.1).float().sum())
                        sit_n[_k] += int(_m.sum())
        # ── ★조우 단위 누적 (tex 정의). step *직전* = 정책이 본 situation 과 현재 타각을 짝지음 ──
        _pwe = env._last_pw
        if _pwe is not None:
            _sn = env.situation
            _cont = (_sn == enc_sit) & (_sn > 0)
            _fin = (enc_sit > 0) & (~_cont)          # 방금 끝난 조우
            _new = (_sn > 0) & (~_cont)              # 방금 시작한 조우
            if bool(_fin.any()):
                enc_finalize(_fin)
            _rst = _fin | _new
            if bool(_rst.any()):
                _z = torch.zeros_like(enc_len)
                enc_len = torch.where(_rst, _z, enc_len)
                for _src in ('act', 'cmd'):
                    for _i in range(3):
                        enc_c[_src][_i] = torch.where(_rst, _z, enc_c[_src][_i])
                enc_mind = torch.where(_rst, torch.full_like(enc_mind, BIG), enc_mind)
                enc_skip = enc_skip & (~_rst)        # 새로 시작한 조우는 온전함
                enc_sit = torch.where(_new, _sn, torch.where(_fin, torch.zeros_like(enc_sit), enc_sit))
            _act_on = _sn > 0
            _f = _act_on.to(enc_len.dtype)
            enc_len = enc_len + _f
            for _src, _db in (('act', env.rudder / vg.MAX_TURN_RATE), ('cmd', a[..., 0])):
                enc_c[_src][0] = enc_c[_src][0] + (_act_on & (_db > EPS_S)).to(enc_len.dtype)
                enc_c[_src][1] = enc_c[_src][1] + (_act_on & (_db < -EPS_P)).to(enc_len.dtype)
                enc_c[_src][2] = enc_c[_src][2] + (_act_on & (_db.abs() < EPS_C)).to(enc_len.dtype)
            _dp = _pwe['dist'].gather(-1, env.danger_idx.unsqueeze(-1)).squeeze(-1)
            enc_mind = torch.where(_act_on, torch.minimum(enc_mind, _dp), enc_mind)

            # ── ★쌍 단위 조우 갱신 (역할 고정, CPA/이탈 종료) ──
            _nrk3 = _pwe['near_risk']; _sit3 = _pwe['sit']
            _d3 = _pwe['dist']; _rt3 = _pwe['raw_tcpa']
            _cand = (_nrk3 > PAIR_ENTRY_RISK) & (_sit3 > 0)
            pr_in = torch.where(_cand & (~pr_active), pr_in + 1.0, torch.zeros_like(pr_in))
            _start = (pr_in >= PAIR_DEBOUNCE) & (~pr_active)
            if bool(_start.any()):
                pr_active = pr_active | _start
                pr_role[_start] = _sit3[_start]
                pr_len[_start] = 0.0; pr_stb[_start] = 0.0; pr_port[_start] = 0.0; pr_ck[_start] = 0.0
                pr_mind[_start] = BIG; pr_fm[_start] = -1.0
                pr_hd0[_start] = env.heading.unsqueeze(-1).expand(-1, -1, N)[_start]
                pr_dpsi[_start] = 0.0; pr_dpsi_min[_start] = 0.0; pr_dpsi_max[_start] = 0.0
                pr_off_max[_start] = -999.0; pr_off_abs[_start] = 0.0
                _sr3 = sr.unsqueeze(-1).expand(-1, -1, N)
                pr_sp0[_start] = _sr3[_start]; pr_spmin[_start] = _sr3[_start]
                pr_dcpa0[_start] = _pwe['dcpa'][_start]
                pr_off_sum[_start] = 0.0; pr_off_cnt[_start] = 0.0
                pr_cpa[_start] = 0.0; pr_far[_start] = 0.0; pr_in[_start] = 0.0
            if bool(pr_active.any()):
                _nr3 = (env.rudder / vg.MAX_TURN_RATE).unsqueeze(-1)      # [E,N,1] → broadcast
                pr_len = pr_len + pr_active.to(pr_len.dtype)
                pr_stb = pr_stb + ((_nr3 > EPS_S) & pr_active).to(pr_len.dtype)
                pr_port = pr_port + ((_nr3 < -EPS_P) & pr_active).to(pr_len.dtype)
                pr_ck = pr_ck + ((_nr3.abs() < EPS_C) & pr_active).to(pr_len.dtype)
                _hnow = env.heading.unsqueeze(-1).expand(-1, -1, N)
                _dps = torch.remainder(_hnow - pr_hd0 + 180.0, 360.0) - 180.0   # [-180,180]
                pr_dpsi = torch.where(pr_active, _dps, pr_dpsi)
                pr_dpsi_min = torch.where(pr_active, torch.minimum(pr_dpsi_min, _dps), pr_dpsi_min)
                pr_dpsi_max = torch.where(pr_active, torch.maximum(pr_dpsi_max, _dps), pr_dpsi_max)
                _off = (-goal[..., 1] * 180.0).unsqueeze(-1).expand(-1, -1, N)   # [E,N,N] 목표 대비 우현 오프셋
                pr_off_max = torch.where(pr_active, torch.maximum(pr_off_max, _off), pr_off_max)
                pr_off_abs = torch.where(pr_active, torch.maximum(pr_off_abs, _off.abs()), pr_off_abs)
                pr_off_sum = torch.where(pr_active, pr_off_sum + _off.abs(), pr_off_sum)
                pr_off_cnt = torch.where(pr_active, pr_off_cnt + 1.0, pr_off_cnt)
                _sr3n = sr.unsqueeze(-1).expand(-1, -1, N)
                pr_spmin = torch.where(pr_active, torch.minimum(pr_spmin, _sr3n), pr_spmin)
                pr_mind = torch.where(pr_active, torch.minimum(pr_mind, _d3), pr_mind)
                _man = pr_active & (pr_fm < 0) & (_nr3.abs() > PAIR_MANEUVER_THR)
                pr_fm = torch.where(_man, pr_len, pr_fm)
                pr_cpa = torch.where(pr_active & (_rt3 < 0), pr_cpa + 1.0, torch.zeros_like(pr_cpa))
                pr_far = torch.where(pr_active & (_d3 > PAIR_FAR_DIST), pr_far + 1.0, torch.zeros_like(pr_far))
                # null 기준선: 활성 조우가 하나도 없는 배의 |오프셋| (배경 잡음)
                # ★20스텝마다·상한 5만개만 표집 (매 스텝 GPU→CPU 전송은 평가를 3배 느리게 만든다)
                _noenc = ~pr_active.any(dim=-1)                       # [E,N] 활성 조우 0
                if (_dstep % 20 == 0) and len(null_off_abs) < 50000:
                    if bool(_noenc.any()):
                        null_off_abs += (-goal[..., 1] * 180.0)[_noenc].abs().tolist()
                # ★매칭 null 창: 무조우 상태가 이어지는 동안 침로변화를 누적, 목표길이에 닿으면 기록.
                #   목표길이는 실제 조우 길이 분포(pr_len)에서 표집 → 창 길이 교란 제거.
                _h1 = env.heading                                     # [E,N]
                _new = _noenc & (nq_len <= 0)
                if bool(_new.any()):
                    nq_h0 = torch.where(_new, _h1, nq_h0)
                    # 관측된 조우 길이에서 표집(없으면 30). 버킷을 고르게 채우려 지수분포 근사
                    _t = torch.randint(8, 140, (E, N), device=dev).to(nq_tgt.dtype)
                    nq_tgt = torch.where(_new, _t, nq_tgt)
                nq_len = torch.where(_noenc, nq_len + 1.0, torch.zeros_like(nq_len))
                _hit = _noenc & (nq_len >= nq_tgt) & (nq_tgt > 0)
                if bool(_hit.any()):
                    _dp = (torch.remainder(_h1 - nq_h0 + 180.0, 360.0) - 180.0).abs()
                    for _bi, (_lo, _hi) in enumerate(NULLW_BUCKETS):
                        _bm = _hit & (nq_len >= _lo) & (nq_len < _hi)
                        if bool(_bm.any()) and len(null_dpsi[_bi]) < 60000:
                            null_dpsi[_bi] += _dp[_bm].tolist()
                    nq_len = torch.where(_hit, torch.zeros_like(nq_len), nq_len)
                _endp = pr_active & ((pr_cpa >= PAIR_END_CPA) | (pr_far >= PAIR_END_FAR))
                if bool(_endp.any()):
                    pair_finalize(_endp)
                    pr_active = pr_active & (~_endp)
                    pr_skip = pr_skip & (~_endp)

        # ── step ──
        obs, rew_step, done, outcome = env.step(a)
        ep_reward += rew_step
        # ── 종료 에피소드: 지표 기록 후 리셋 ──
        # ★2026-09-05 fix: drain 구간(_pending is not None)에서는 '창 끝에 걸려 있던' 에피소드만
        #   집계에 넣는다. 안 그러면 이미 마감된 배가 새로 시작한 에피소드까지 딸려 들어와
        #   창이 팔마다 다른 길이로 늘어난다. drain 이 꺼져 있으면 _pending is None → 기존 동작 그대로.
        _cnt_gate = counted if _pending is None else (counted & _pending)
        for oc in range(1, 5):
            mask = (outcome == oc) & _cnt_gate     # 경계 걸친 첫 에피소드 제외
            c = int(mask.sum())
            if c:
                counts[oc] += c; total += c
                msum[oc]['fuel'] += float(ep_fuel[mask].sum())
                msum[oc]['head'] += float(ep_head[mask].sum())
                msum[oc]['minsep'] += float(ep_minsep[mask].clamp(max=vg.RADAR_RANGE * 4).sum())
                msum[oc]['length'] += float(ep_len[mask].sum())
                msum[oc]['reward'] += float(ep_reward[mask].sum())
                msum[oc]['n'] += c
        term = (outcome != 0)
        if int(term.sum()):
            _rec = term & _cnt_gate
            if int(_rec.sum()):
                ms = ep_minsep[_rec].clamp(max=vg.RADAR_RANGE * 4)
                all_minsep_sum += float(ms.sum()); all_minsep_n += int(_rec.sum())
            ep_fuel = torch.where(term, torch.zeros_like(ep_fuel), ep_fuel)
            ep_head = torch.where(term, torch.zeros_like(ep_head), ep_head)
            ep_len = torch.where(term, torch.zeros_like(ep_len), ep_len)
            ep_minsep = torch.where(term, torch.full_like(ep_minsep, BIG), ep_minsep)
            ep_reward = torch.where(term, torch.zeros_like(ep_reward), ep_reward)
            counted = counted | term          # 기록 *후* 갱신 → 첫 종료는 제외, 이후부터 집계
            if _pending is not None:
                _pending = _pending & (~term)   # ★2026-09-05 fix: drain — 마감된 에이전트는 대기에서 제외
            # ★조우: 종료(충돌 포함)로 끊긴 조우도 5스텝 이상이면 집계한다.
            #   버리면 '충돌로 끝난 비준수 조우'가 통째로 빠져 준수율이 낙관 편향됨.
            enc_finalize(term & (enc_sit > 0))
            _z = torch.zeros_like(enc_len)
            enc_len = torch.where(term, _z, enc_len)
            for _src in ('act', 'cmd'):
                for _i in range(3):
                    enc_c[_src][_i] = torch.where(term, _z, enc_c[_src][_i])
            enc_mind = torch.where(term, torch.full_like(enc_mind, BIG), enc_mind)
            enc_sit = torch.where(term, torch.zeros_like(enc_sit), enc_sit)
            enc_skip = enc_skip & (~term)
            # ★쌍: i 또는 j 가 종료(도착·충돌·시간초과)한 쌍도 마감 — 충돌로 끝난 비준수 조우를 버리면
            #   준수율이 낙관 편향되므로 반드시 집계에 넣는다.
            _t3 = term.unsqueeze(-1) | term.unsqueeze(1)          # [E,N,N]
            _endt = pr_active & _t3
            if bool(_endt.any()):
                pair_finalize(_endt)
                pr_active = pr_active & (~_endt)
            pr_skip = pr_skip & (~_t3)
            pr_in = torch.where(_t3, torch.zeros_like(pr_in), pr_in)
        radar, goal, self_s, sit = parse_obs(obs); fs.push(radar, done)
        prev_head = torch.where(done, env.heading, prev_head)

    # ★2026-09-05 fix: 절단 실태를 항상 남김 — 팔마다 잘린 양이 다른지 눈으로 확인 가능해야 하고,
    #   출력에 안 남으면 사후 보정도 불가능했음.
    _op = (ep_len > 0) & counted        # 집계 대상(첫 종료를 이미 본 배)만이 실제 손실분
    _nop = int(_op.sum())
    _nall = int((ep_len > 0).sum())
    _oplen = float(ep_len[_op].mean()) if _nop else 0.0
    print(f"   [censored] 창 끝 미완 에피소드 {_nop}개 (전체 미완 {_nall}개, 평균 진행 {_oplen:.0f}결정), 집계 제외"
          + (" (drain 후 잔여)" if args.drain > 0 else " (--drain N 으로 마감 가능)"), flush=True)
    # ★2026-09-05 fix: 실행 파라미터를 stdout 에 남김. burnin 기본값(1200)과 문서에 기록된 실행
    #   명령(--burnin 1500)이 서로 달라, --burnin 을 준 실행과 안 준 실행의 숫자가 다른데 출력만
    #   보고는 어느 쪽인지 구분이 불가능했음. 기본값 자체는 과거 숫자 재현 때문에 바꾸지 않고,
    #   무슨 값으로 돌렸는지 기록만 남김.
    print(f"   [run] burnin={args.burnin} dec={args.eval_decisions} drain={args.drain} "
          f"ring={args.ring} crossing={args.crossing} seed={args.seed} "
          f"envs={E} vessels={N} maxp={args.max_partners} dev={dev}", flush=True)

    if total == 0:
        print(f"{os.path.basename(ckpt_path):26s} | no terminations"); return
    g, v, o, t = (float(counts[i]) / total * 100 for i in (1, 2, 3, 4))
    # goal 에피소드 기준 궤적 지표(공정 비교: 도달한 배들이 얼마나 부드럽고 안전하게 갔나)
    gm = msum[1]
    if gm['n'] > 0:
        gf = gm['fuel'] / gm['n']; gh = gm['head'] / gm['n']
        gs = gm['minsep'] / gm['n']; gl = gm['length'] / gm['n']
    else:
        gf = gh = gs = gl = float('nan')
    amin = all_minsep_sum / max(all_minsep_n, 1)
    # ★연속 지표: 전 에피소드 평균 보상 + COLREGs 준수도(연속 0~1, 이진 준수율)
    r_all = sum(msum[oc]['reward'] for oc in range(1, 5)) / max(total, 1)
    comp = comp_sum / max(comp_n, 1)
    compb = comp_bin / max(comp_n, 1) * 100
    print(f"{os.path.basename(ckpt_path):26s} | arm={args.arm:6s} | "
          f"goal={g:5.1f}%  vColl={v:5.1f}%  oColl={o:5.1f}%  TO={t:5.1f}%  | {total} eps || "
          f"[goal-ep] fuel={gf:6.1f}  headTravel={gh:6.0f}deg  minSep={gs:5.1f}m  len={gl:5.0f}  | allMinSep={amin:5.1f}m "
          f"|| epReward={r_all:8.2f}  colregs={comp:.4f}  colregsOK={compb:5.1f}%  (n={comp_n})"
          + " || " + " ".join(f"sit{k}={100*sit_ok[k]/max(sit_n[k],1):5.1f}%(n={sit_n[k]})"
                              for k in (1, 2, 3, 4)))

    # ─── ★조우 단위 COLREGs 준수도 (colregs_compliance_metric.tex) ───
    _NM = {1: 'HeadOn(R14)', 2: 'StandOn(R17)', 3: 'GiveWay(R15/16)', 4: 'Overtake(R13)'}
    _tot = sum(enc_tot.values())
    if _tot == 0:
        print("   [encounter-COLREGs] 조우 0건 (5스텝 이상 지속된 COLREGs 상황 없음)")
    else:
        for _src, _lab in (('act', '실제타각'), ('cmd', '명령타각')):
            _ok = sum(enc_ok[_src].values())
            _per = "  ".join(f"{_NM[k]}={100*enc_ok[_src][k]/max(enc_tot[k],1):5.1f}%(n={enc_tot[k]})"
                             for k in (1, 2, 3, 4))
            print(f"   [encounter-COLREGs/{_lab}] C={100*_ok/_tot:5.1f}%  ({_ok}/{_tot} 조우)  {_per}")
        _dm = sum(enc_dsum.values()) / _tot
        _dper = "  ".join(f"{_NM[k]}={enc_dsum[k]/max(enc_tot[k],1):5.1f}m" for k in (1, 2, 3, 4))
        print(f"   [encounter-minPass]  전체={_dm:5.1f}m   {_dper}")

    # ─── ★쌍 단위 조우 (역할 고정) — 2026-08-31 승인 지표 ───
    import statistics as _st
    _tp = sum(len(v) for v in pair_res.values())
    if _tp == 0:
        print("   [pair-COLREGs] 쌍 조우 0건")
    else:
        _okp = sum(1 for v in pair_res.values() for o, _, _ in v if o)
        _parts = []
        for k in (1, 2, 3, 4):
            v = pair_res[k]
            _parts.append(f"{_NM[k]}={100*sum(o for o,_,_ in v)/len(v):5.1f}%(n={len(v)})" if v
                          else f"{_NM[k]}=  -  (n=0)")
        print(f"   [pair-COLREGs/실제타각] C={100*_okp/_tp:5.1f}%  ({_okp}/{_tp} 조우)  " + "  ".join(_parts))
        _mds = [d_ for v in pair_res.values() for _, d_, _ in v]
        _fms = [f_ for v in pair_res.values() for _, _, f_ in v if f_ >= 0]
        _nofm = sum(1 for v in pair_res.values() for _, _, f_ in v if f_ < 0)
        _lensum = 0.0
        _fmtxt = f"{_st.median(_fms)*0.4:5.1f}s" if _fms else "  -  "
        print(f"   [pair-detail] 통과최소거리 중앙 {_st.median(_mds):5.1f}m | "
              f"첫 변침(|δ̄|>{PAIR_MANEUVER_THR})까지 중앙 {_fmtxt} | 무변침 조우 {100*_nofm/_tp:4.1f}%")
        # ── ★침로변화 기반 준수 (Rule 8(b) 근거). 양보역할(1,3,4)=우현으로 Δψ 이상, 유지역할(2)=|Δψ| 이하 ──
        _NMH = {1: 'HeadOn', 2: 'StandOn', 3: 'GiveWay', 4: 'Overtake'}
        for _thr in DPSI_THRESHOLDS:
            _parts, _ok_all, _n_all = [], 0, 0
            for k in (1, 2, 3, 4):
                v = hd_res[k]
                if not v:
                    _parts.append(f"{_NMH[k]}=  -  (n=0)"); continue
                arr = _np.asarray(v)
                okk = (arr >= _thr) if k in (1, 3, 4) else (_np.abs(arr) <= _thr)
                _ok_all += int(okk.sum()); _n_all += len(arr)
                _parts.append(f"{_NMH[k]}={100*okk.mean():5.1f}%(n={len(arr)})")
            if _n_all:
                print(f"   [pair-heading/Δψ>={_thr:4.0f}deg] C={100*_ok_all/_n_all:5.1f}%  " + "  ".join(_parts))
        _allv = _np.concatenate([_np.asarray(hd_res[k]) for k in (1, 3, 4) if hd_res[k]]) if any(hd_res[k] for k in (1,3,4)) else _np.array([0.0])
        _stv = _np.asarray(hd_res[2]) if hd_res[2] else _np.array([0.0])
        if null_off_abs:
            _nb = _np.asarray(null_off_abs)
            print(f"   [null-baseline] 조우 없는 배의 |목표대비 오프셋| 중앙 {_np.median(_nb):5.1f}deg  "
                  f"(<=5° {100*(_nb<=5).mean():4.1f}%  <=10° {100*(_nb<=10).mean():4.1f}%  <=20° {100*(_nb<=20).mean():4.1f}%)")
        print(f"   [pair-heading/raw] 양보역할 Δψ 중앙 {_np.median(_allv):+6.1f}deg (우현+) | "
              f"유지역할 |Δψ| 중앙 {_np.median(_np.abs(_stv)):5.1f}deg")
        # ── ★Rule 8 완전판: "침로 OR 속력" (2026-09-04, 팔별 결과 보기 전 정의 확정) ──
        #   Rule 8(b) 원문은 alteration of COURSE **OR SPEED**. 침로만 재면 규정의 절반만 재는 것이고,
        #   감속 회피(8(e) "필요하면 속력을 줄이거나 정지")를 한 배가 0점으로 집계된다.
        #   양보역할 준수 = (우현 오프셋 >= 문턱) OR (진입 대비 속력 감소 >= SPD_DROP)
        SPD_DROP = 0.15                       # 15% 이상 감속 = 유의미한 속력 회피
        _R8N = {1: 'HeadOn', 3: 'GiveWay', 4: 'Overtake'}
        for _thr in (5, 10, 20):
            _p, _oa, _na = [], 0, 0
            for k in (1, 3, 4):
                v = r8_res[k]
                if not v:
                    _p.append(f"{_R8N[k]}=  -  (n=0)"); continue
                _off_ = _np.asarray([x[0] for x in v]); _dr = _np.asarray([x[1] for x in v])
                ok = (_off_ >= _thr) | (_dr >= SPD_DROP)
                _oa += int(ok.sum()); _na += len(v)
                _only = int(((_off_ < _thr) & (_dr >= SPD_DROP)).sum())
                _p.append(f"{_R8N[k]}={100*ok.mean():5.1f}%(n={len(v)}, 속력만={100*_only/len(v):4.1f}%)")
            if _na:
                print(f"   [Rule8/침로or속력>={_thr:3.0f}deg] C={100*_oa/_na:5.1f}%  " + "  ".join(_p))
        # ── ★난이도 층화: 진입 DCPA 구간별 양보역할 준수 (분모 구성 차이 보정) ──
        #   통신 팔은 위험 조우 자체가 적다(분모가 다름) → 같은 진입 DCPA 구간끼리만 비교해야 공정.
        _DB = ((0, 10), (10, 20), (20, 35), (35, 10 ** 9))
        _gv = r8_res[3]
        if _gv:
            _o = _np.asarray([x[0] for x in _gv]); _d = _np.asarray([x[1] for x in _gv])
            _dc = _np.asarray([x[2] for x in _gv])
            _txt = []
            for _lo, _hi in _DB:
                _m = (_dc >= _lo) & (_dc < _hi)
                if int(_m.sum()) < 30:
                    _txt.append(f"{_lo}-{_hi if _hi < 10**8 else ''}m: n<30"); continue
                _ok = (_o[_m] >= 10) | (_d[_m] >= SPD_DROP)
                _txt.append(f"{_lo}-{_hi if _hi < 10**8 else ''}m={100*_ok.mean():5.1f}%(n={int(_m.sum())})")
            print(f"   [GiveWay/진입DCPA층화 Rule8@10deg] " + "  ".join(_txt))

        # ── ★Rule 17 매칭 null 대조 판정 (2026-09-04, 정의는 팔별 결과 보기 전 확정) ──
        #   유지역할 조우의 |Δψ| 를 *같은 길이 창*의 무조우 |Δψ| 분포와 비교.
        #   준수 = 그 배가 '조우가 없었을 때만큼만' 침로를 틀었다(= 배경 항해 수준).
        #   문턱은 null 분포의 백분위로 고정(절대 각도 아님) → 목표 방위·창 길이 교란 제거.
        if sd_dpsi and any(len(v) >= 50 for v in null_dpsi.values()):
            def _bucket(L):
                for _i, (_lo, _hi) in enumerate(NULLW_BUCKETS):
                    if _lo <= L < _hi:
                        return _i
                return len(NULLW_BUCKETS) - 1
            _nullq = {}
            for _i, v in null_dpsi.items():
                if len(v) >= 50:
                    _nullq[_i] = _np.percentile(_np.asarray(v), [50, 75, 90, 95])
            _res = {q: [0, 0] for q in range(4)}      # 백분위 인덱스 → [통과, 전체]
            _paired = []                               # (조우 |Δψ|, 같은 버킷 null 중앙)
            for _dp, _L in sd_dpsi:
                _b = _bucket(_L)
                if _b not in _nullq:
                    continue
                for _qi in range(4):
                    _res[_qi][1] += 1
                    if _dp <= _nullq[_b][_qi]:
                        _res[_qi][0] += 1
                _paired.append((_dp, _nullq[_b][0]))
            if _res[0][1]:
                _lbl = ('null중앙(P50)', 'P75', 'P90', 'P95')
                _txt = "  ".join(f"{_lbl[_qi]}={100*_res[_qi][0]/_res[_qi][1]:5.1f}%" for _qi in range(4))
                print(f"   [StandOn/null대조] n={_res[0][1]}  {_txt}")
                _a = _np.asarray([p[0] for p in _paired]); _c = _np.asarray([p[1] for p in _paired])
                print(f"   [StandOn/null대조] 유지역할 |Δψ| 중앙 {_np.median(_a):5.1f}deg  vs  "
                      f"무조우 같은길이 창 중앙 {_np.median(_c):5.1f}deg  (비 {_np.median(_a)/max(_np.median(_c),1e-9):4.2f}x)")
                _nb2 = {k: len(v) for k, v in null_dpsi.items()}
                print(f"   [StandOn/null대조] null 창 표본(길이버킷별) {_nb2}")


if __name__ == '__main__':
    main()
