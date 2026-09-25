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
# eval/ 로 내려온 뒤에도 Python/ 루트의 config·vessel_gym·networks·vessel_gym_train 을
# 찾으려면 sys.path 에 넣어야 함 (__init__.py 없음, PLAN.md 7단계 — unity-island/fidelity/astar_fig9와 동일 패턴).
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
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
    ap.add_argument('--ring', type=float, default=None)         # ★2026-09-10: None=체크포인트 스냅샷 값. 명시하면 override(로그에 남음)
    ap.add_argument('--crossing', type=int, default=None)       # ★2026-09-10: None=스냅샷 값. 구 ckpt(스냅샷 없음)는 명시 필수
    # ★ckpt 의 학습 arm 과 --arm 이 다르면 기본은 중단. 의도한 교차평가(예: ON 정책의 통신을 끊어 재기)만 이 플래그로 허용.
    ap.add_argument('--allow_arm_mismatch', action='store_true')
    # ★2026-09-10: comm_range 는 import 시점 고정. 스냅샷과 다르면 기본 중단(ckpt_io). 과거 숫자 재현 목적만 허용.
    ap.add_argument('--allow_comm_range_mismatch', action='store_true')
    ap.add_argument('--allow_sim_mismatch', action='store_true',
                    help='dyn_profile/obstacles/radar_range 가 스냅샷과 달라도 진행(스냅샷 값 강제 적용, 교차평가 전용)')
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
    # ★2026-09-25 의도·역할 통신(COMM_EXT) 절제·기전 진단. 전부 기본 off = 기존 출력 불변(조율 진단 줄만 추가).
    #   절제 이름을 섞지 말 것: msgzero(=--arm OFF --allow_arm_mismatch, latent+확장필드 전부 0) /
    #   latent-zero(--latent_zero, 확장필드 유지) / <그룹>-zero(--comm_groups 로 남길 그룹 지정) / field-shuffle.
    ap.add_argument('--comm_groups', default=None,
                    help="평가에 켤 확장필드 그룹(쉼표: state,role,intent / 'none'). 학습과 다르면 --allow_fields_mismatch 필요")
    ap.add_argument('--allow_fields_mismatch', action='store_true', help='의도한 필드 절제 평가만')
    ap.add_argument('--latent_zero', action='store_true', help='학습 latent 메시지만 0 (확장필드·위치는 유지)')
    ap.add_argument('--field_shuffle', action='store_true',
                    help='확장필드를 같은 env 의 다른 수신자 것으로 섞음(분포 유지, 정보 차단)')
    ap.add_argument('--traj_out', default=None,
                    help='궤적 덤프 경로(.pt). 앞 --traj_envs 개 env 의 위치·침로·속력·타각·명령·상황·목표·최대속력·종료를 결정마다 저장. '
                         "F5 '같은 조우 ON vs OFF' 는 --burnin 0 + 같은 --seed 로 돌려야 팔 간 초기 장면이 같다(run_repro.sh traj)")
    ap.add_argument('--traj_envs', type=int, default=4)
    args = ap.parse_args()

    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    E, N = args.envs, args.vessels
    torch.manual_seed(args.seed)
    # eval/ 로 내려온 뒤에도 기본 체크포인트 위치는 Python/checkpoints 그대로(깊이 +1 보정).
    scr = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    # 상대 경로로 넘기면 VESSEL_CKPT_DIR(기본: Python/checkpoints/)에서 찾는다.
    ckpt_dir = os.environ.get('VESSEL_CKPT_DIR', os.path.join(scr, 'checkpoints'))
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ckpt_dir, args.ckpt)

    # ★2026-09-10: 복원·env 생성은 ckpt_io 단일 구현으로. (예전 인라인 블록 ~100줄은 그리로 옮김)
    #   순서: 스냅샷 스니핑 → networks 전역 덮어쓰기 → CNNPolicy() → strict 로드. comm_range/arm 불일치는 기본 중단.
    #   env 는 스냅샷의 ring/crossing/perpair 를 쓴다. --ring/--crossing 을 주면 override 로 로그에 남는다.
    from ckpt_io import restore_policy, make_env_from_snapshot
    _cg = None
    if args.comm_groups is not None:
        _cg = () if args.comm_groups.strip().lower() == 'none' else tuple(
            g.strip().lower() for g in args.comm_groups.split(',') if g.strip())
    _r = restore_policy(ckpt_path, dev, arm=args.arm, max_partners=args.max_partners,
                        allow_arm_mismatch=args.allow_arm_mismatch,
                        allow_comm_range_mismatch=args.allow_comm_range_mismatch,
                        allow_sim_mismatch=args.allow_sim_mismatch, tag='[eval]',
                        comm_groups=_cg, allow_fields_mismatch=args.allow_fields_mismatch)
    policy = _r.policy
    args.max_partners = _r.max_partners
    env = make_env_from_snapshot(_r.snap, device=dev, num_envs=E, seed=args.seed, n_vessels=N,
                                 ring=args.ring, crossing=args.crossing, tag='[eval]')
    args.ring, args.crossing = env.ring_scale, env.crossing   # 뒤에서 [run] 로그가 찍는 값

    fs = FrameStack(E, N, dev)
    obs = env.reset()
    radar, goal, self_s, sit = parse_obs(obs); fs.reset_all(radar)

    counts = torch.zeros(5, device=dev)   # RUNNING/GOAL/vColl/oColl/TO
    total = 0

    # ★2026-09-25 절제 옵션 (ON 팔에서만 의미). 전용 CPU generator → 정책 샘플 RNG 와 분리.
    if (args.latent_zero or args.field_shuffle) and args.arm != 'ON':
        raise SystemExit('[eval] --latent_zero/--field_shuffle 은 --arm ON 에서만 의미가 있음')
    if args.field_shuffle and getattr(policy, 'relpos_dim', 3) <= 3:
        raise SystemExit('[eval] --field_shuffle 은 COMM_EXT 체크포인트에서만 (확장필드가 없음)')
    _shuf_gen = torch.Generator().manual_seed(args.seed + 7) if args.field_shuffle else None
    _zmsg = torch.zeros(E, N, _r.msg_dim, device=dev) if args.latent_zero else None
    if args.latent_zero or args.field_shuffle or args.comm_groups is not None:
        print(f"[eval] 절제: latent_zero={args.latent_zero} field_shuffle={args.field_shuffle} "
              f"comm_groups={list(net.COMM_GROUPS)} (학습과 다른 입력 — 절제 결과로만 보고)", flush=True)

    def act(x, goal, self_s, sit):
        with torch.no_grad():
            if args.arm == 'ON':
                om, _ = comm_gather(policy, env, x, goal, self_s, sit, args.max_partners,
                                    msg_override=_zmsg, ext_shuffle_gen=_shuf_gen)
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

    # ★2026-09-26 (T5) 평가 루프의 GPU→CPU 동기화 제거 — 실측 스텝당 ~100회(bool(any())·int(sum())·float(sum())·tolist)가
    #   호스트 바운드의 주범이었다. 루프 안 누적기는 전부 device 텐서로 두고 루프 뒤에 위 호스트 변수로 되돌린다
    #   (출력부는 글자 하나 안 바뀜). 비트동일 규칙:
    #   · 정수 카운트(0/1 합) → int64 마스크 합. 순서·dtype 무관하게 정확.
    #   · 연속 합: 옛 코드 `float(x[mask].sum())` = 행우선으로 압축된 float32 벡터를 reduce 커널로 합산 → 호스트 float64 누적.
    #     nonzero(동기화) 없이 같은 벡터를 만들려고 stable argsort 로 mask 의 True 원소를 행우선 순서로 모은 뒤
    #     (advanced index → 새 할당, 같은 numel → 같은 커널 구성·같은 덧셈 순서) 합산하고 .double() 로 device 에서 누적.
    #     슬라이스 길이(호스트 int)만 스텝당 2회(step 전·후)의 packed .tolist() 로 받는다.
    #   · 히스토그램은 torch.bincount 가 아니라 index_add_ — bincount 는 CUDA 에서 max() 로 동기화한다(실측 3회/호출).
    #   · 리스트 수집기(pair_res·null_*)는 device 텐서 리스트로 모아 루프 뒤 한 번에 .tolist() — 붙는 순서는 옛 append 순서 그대로.
    #   · 제어흐름·난수를 정하는 호스트값(pr_active.any()·_new.any() → torch.randint)은 그대로 호스트에서 판단하되 같은 전송에 싣는다.
    _f64 = dict(dtype=torch.float64, device=dev)
    _i64 = dict(dtype=torch.long, device=dev)
    _acc_comp_sum = torch.zeros((), **_f64)
    _acc_sit_n = torch.zeros(5, **_i64); _acc_sit_ok = torch.zeros(5, **_i64)      # [0] = 게이트 밖(미사용), 합 = comp_n / comp_bin
    _acc_msum = torch.zeros(5, 5, **_f64)                                           # [outcome, (fuel, head, minsep, length, reward)]
    _acc_all_minsep = torch.zeros((), **_f64)
    _ones_en = torch.ones(E * N, **_i64); _ones_enn = torch.ones(E * N * N, **_i64)

    def _hist5(key, ones):
        """key [n] long ∈ 0..4 → 개수 [5] (== torch.bincount(key, minlength=5) 값, 동기화 없음. 정수 원자합이라 정확)."""
        return torch.zeros(5, **_i64).index_add_(0, key, ones)

    def _compact(vals_flat, order, off, c):
        """stable argsort 압축: order[off:off+c] 가 가리키는 원소를 행우선 순서 그대로 새 텐서 [c] 로 뽑는다
        (== vals[mask] 의 원소·순서). 새 할당(정렬된 포인터)이라 .sum() 의 커널 구성이 옛 masked_select 결과와 같다 —
        큰 벡터의 슬라이스 뷰를 바로 합산하면 안 된다(reduce 커널이 포인터 정렬에 따라 머리 원소를 따로 더한다)."""
        return vals_flat[order[off:off + c]]

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
    _acc_enc_tot = torch.zeros(5, **_i64)                                   # [k] = 조우 수 (루프 뒤 enc_tot)
    _acc_enc_ok = {'act': torch.zeros(5, **_i64), 'cmd': torch.zeros(5, **_i64)}
    _acc_enc_dsum = torch.zeros(5, **_f64)

    def enc_prepare(mask):
        """mask [E,N]: 방금 끝난 조우를 tex 기준으로 판정·집계 (5스텝 미만·burn-in 잔여는 제외) — 옛 enc_finalize 의
        동기화 없는 판. 조우 수·판정 통과 수는 여기서 device 에 바로 누적(0/1 합 = 정확). 최소통과거리 합만 압축 길이가
        필요하므로 (압축용 벡터, order, 역할별 개수) 를 돌려주고 packed 전송 뒤 enc_apply 가 더한다.
        판정은 옛 코드처럼 mk 로 뽑은 뒤 나누지 않고 전체 텐서에서 나눈 뒤 mk 로 센다 — 원소별 나눗셈·비교라 값이 같고,
        mk 밖의 0/0=NaN 은 비교에서 False 인 데다 mk 로 걸러진다."""
        m = mask & (enc_len >= ENC_MIN_STEPS) & (~enc_skip)
        _z = torch.zeros_like(enc_sit)
        key = torch.where(m, enc_sit, _z).view(-1)                          # 0 = 집계 제외, 1..4 = 역할
        cnt = _hist5(key, _ones_en)                                         # cnt[k] == 옛 int(mk.sum())
        _acc_enc_tot.add_(cnt)
        for src in ('act', 'cmd'):
            stb = enc_c[src][0] / enc_len
            port = enc_c[src][1] / enc_len
            ck = enc_c[src][2] / enc_len
            ok = torch.where(enc_sit == 2, ck > 0.6,
                             torch.where(enc_sit == 4, stb > 0.3, (stb > 0.5) & (port < 0.2)))   # k∈{1,3}: 우현·좌현 기준
            _acc_enc_ok[src].add_(_hist5(torch.where(m & ok, enc_sit, _z).view(-1), _ones_en))
        return enc_mind.clamp(max=vg.RADAR_RANGE * 4).view(-1), torch.argsort(key, stable=True), cnt

    def enc_apply(prep, cnt):
        """prep = enc_prepare 반환값, cnt = 그 개수의 호스트 사본. 옛 `float(enc_mind[mk].clamp().sum())` 과 같은 압축 합."""
        vals, order, _ = prep
        off = cnt[0]
        for k in (1, 2, 3, 4):
            c = cnt[k]
            if c:
                _acc_enc_dsum[k].add_(_compact(vals, order, off, c).sum().double())
            off += c

    # ★2026-09-15: 진행 표시. 기존엔 burn-in 200 에서 ETA 한 줄만 찍고 끝까지 무출력이라
    #   몇 분짜리인지 몇 십 분짜리인지 알 수 없었다(실측: 그 ETA 25분 → 실제 38분).
    #   그 ETA 는 burn-in 첫 창 처리율로 한 번 계산한 값이라 뒤로 갈수록 낙관적이다.
    #   아래는 경과시간으로 매번 다시 추정한다. 순수 출력이므로 계산·난수·상태에 영향 없음.
    _prog_total = args.burnin + args.eval_decisions
    _prog_every = max(1, _prog_total // 20)

    def _progress(step_done, tag):
        el = time.time() - t0
        eta = el * (_prog_total - step_done) / max(step_done, 1)
        print(f"  [{os.path.basename(ckpt_path)}] {tag} {step_done}/{_prog_total} 결정 "
              f"({100.0 * step_done / _prog_total:.0f}%) 경과 {el / 60:.1f}분 ETA ~{eta / 60:.1f}분",
              flush=True)

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
        if (i + 1) % _prog_every == 0:
            _progress(i + 1, 'burn-in')

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

    # ─── ★조율 진단 (2026-09-25, 스펙 2026-09-25-comm-intent-design §5) ───
    #   저자 의도 '서로 역할·의도를 알면 상대가 보완 행동을 한다' 를 행동으로 잰다. 기전 설명용 — 사전등록 H1 판정 아님.
    #   모든 쌍 (i,j) 의 게이트된 기하 역할(접근·dcpa<24 m·≤COMM_RANGE — comm_pair_features 와 같은 정의)과
    #   실제 타각(쌍 지표와 같은 문턱: 우현 >0.05 / 좌현 <−0.10 / 유지 |·|<0.15)을 짝지어 센다. OFF 팔에도 같은 정의.
    #   거리대: near = 레이더 안(≤RADAR_RANGE), far = 레이더 밖~COMM_RANGE (far 의 양보선 우현 = 레이더 밖 조기 행동).
    _cidx = torch.arange(N, device=dev)
    _allj = torch.stack([torch.cat([_cidx[:i], _cidx[i + 1:]]) for i in range(N)]).unsqueeze(0).expand(E, N, N - 1)
    _bE = torch.arange(E, device=dev)[:, None, None]
    _upper = _allj > _cidx.view(1, N, 1)                        # 무순서 쌍 1회 (HeadOn·일치율)
    _CK = ('gwso', 'comp', 'gw_stb', 'so_hold', 'so_port', 'both_hold', 'ho', 'ho_both_stb', 'roled', 'agree')
    coord = {band: {k: 0.0 for k in _CK} for band in ('near', 'far')}
    _acc_coord = {band: torch.zeros(len(_CK), **_i64) for band in ('near', 'far')}   # _CK 순서, 루프 뒤 coord 로
    # 충돌 분해 (스펙 §5): 선박 충돌로 끝난 배마다 직전 결정의 '가장 가까운 상대' 와의 역할·실제 타각으로 분류
    _COLL_K = ('multi', 'gw_noact', 'gw_act_so_hold', 'gw_act_so_turn', 'so_act', 'so_hold_gw_noact',
               'so_hold_gw_act', 'ho_both_stb', 'ho_not_both', 'overtaking', 'no_role')
    coll_cat = {k: 0 for k in _COLL_K}
    _acc_coll = torch.zeros(len(_COLL_K), **_i64)                    # _COLL_K 순서, 루프 뒤 coll_cat 으로
    _cctx = {}                                                      # _coord_update 가 채우는 step 전 상태

    def _coord_update(a):
        f = vg.comm_pair_features(env, _allj, cfg.COMM_RANGE)   # [E,N,N-1,20], 역할 게이트 적용
        myr = f[..., 8:13].argmax(-1)
        thr = f[..., 13:18].argmax(-1)
        dij = torch.linalg.norm(env.pos[_bE, _allj] - env.pos.unsqueeze(2), dim=-1)
        # ★2026-09-25 리뷰 E2 수정: 샘플 명령(탐색 잡음·팔마다 다른 logstd 가 섞임)이 아니라 *실제 타각*(상대 배가 보는 것)으로,
        #   쌍 지표(tex)와 같은 문턱 EPS_S/EPS_P/EPS_C 를 쓴다. 우현 δ̄>0.05 · 좌현 δ̄<−0.10 · 유지 |δ̄|<0.15.
        _dn = env.rudder / vg.MAX_TURN_RATE                           # [E,N] 실제(슬루된) 정규화 타각
        ci = _dn.unsqueeze(-1).expand_as(dij)
        cj = _dn[_bE, _allj]
        stb_i = ci > EPS_S
        hold_i = (~stb_i) & (ci.abs() < EPS_C)
        stb_j, hold_j, port_j = cj > EPS_S, cj.abs() < EPS_C, cj < -EPS_P
        near = dij <= vg.RADAR_RANGE
        GW, SO, HO, OT, NO = vg.SIT_GIVEWAY, vg.SIT_STANDON, vg.SIT_HEADON, vg.SIT_OVERTAKING, vg.SIT_NONE
        agree = (((myr == GW) & (thr == SO)) | ((myr == SO) & (thr == GW)) | ((myr == HO) & (thr == HO))
                 | ((myr == OT) & (thr == NO)) | ((myr == NO) & (thr == OT)))
        roled = (myr != NO) | (thr != NO)
        for band, bm in (('near', near), ('far', ~near)):
            g = bm & (myr == GW) & (thr == SO)                     # i 양보선, j 유지선 (순서쌍이지만 방향이 정해져 1회)
            h = bm & (myr == HO) & (thr == HO) & _upper
            r = bm & roled & _upper
            # _CK 순서 (gwso comp gw_stb so_hold so_port both_hold ho ho_both_stb roled agree). 0/1 합 → int64 로 device 누적
            _acc_coord[band].add_(torch.stack([
                g.sum(), (g & stb_i & hold_j).sum(), (g & stb_i).sum(), (g & hold_j).sum(), (g & port_j).sum(),
                (g & hold_i & hold_j).sum(), h.sum(), (h & stb_i & stb_j).sum(), r.sum(), (r & agree).sum()]))
        # 충돌 분해용: 배 i 의 가장 가까운 상대 jn 과의 역할·행동 + 위험 상대 수(게이트 통과 역할이 붙은 상대)
        jn = dij.argmin(-1, keepdim=True)                               # [E,N,1]
        _g = lambda t: t.gather(-1, jn).squeeze(-1)                     # noqa: E731
        _cctx.update(myr=_g(myr), thr=_g(thr), stb_i=_g(stb_i), hold_i=_g(hold_i), stb_j=_g(stb_j),
                     hold_j=_g(hold_j), n_risky=((myr != NO) | (thr != NO)).sum(-1))

    # ─── ★궤적 덤프 (2026-09-25, --traj_out) — F5 '같은 조우 ON vs OFF' 용. 앞 traj_envs 개 env, 평가 창 전체 ───
    _traj = None
    if args.traj_out:
        _te = max(1, min(int(args.traj_envs), E))
        _traj = {k: [] for k in ('pos', 'heading', 'speed', 'rudder', 'cmd', 'situation', 'goal', 'max_speed', 'outcome')}

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
    _null_off_dev = []; _null_off_n = 0        # ★T5: device 조각 리스트 + 호스트 길이(옛 len(null_off_abs) 상한 검사용)
    # ★Rule 17 매칭 null 대조 (2026-09-04): '조우가 없는 배'가 같은 길이 창에서 침로를 얼마나 트는가.
    #   배는 조우와 무관하게 목표를 향해 계속 돈다 → 절대 문턱은 그 배경분을 못 뺀다
    #   (실측: 조우 없는 배의 |목표대비 오프셋| 중앙 9.6°, 10° 이내는 51.8%뿐).
    #   Rule 17(a)(i) '침로·속력 유지'는 *변화량* 규정이므로 |Δψ| 로 재고, 같은 길이 창의
    #   무조우 |Δψ| 분포를 기준선으로 삼는다. 길이 버킷별로 모아 길이 교란을 제거한다.
    NULLW_BUCKETS = ((0, 20), (20, 40), (40, 80), (80, 10 ** 9))
    null_dpsi = {i: [] for i in range(len(NULLW_BUCKETS))}   # 버킷 → [|Δψ|, ...]
    _null_dpsi_dev = {i: [] for i in range(len(NULLW_BUCKETS))}; _null_dpsi_n = {i: 0 for i in range(len(NULLW_BUCKETS))}
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
    # ★T5: 쌍 조우 레코드 [P,10] float32 (역할, ok, 최소거리, 첫변침, 평균|오프셋|, 최대우현오프셋, |Δψ|, 길이, 감속률, 진입DCPA)
    #   device 리스트로 모아 루프 뒤 한 번에 .tolist() → pair_res / hd_res / sd_dpsi / r8_res 를 옛 append 순서로 채운다.
    _pair_recs = []

    def _pr_state():
        """현재 commit 된 쌍 상태(활성 쌍 갱신이 바꾸는 텐서만). _pair_advance 가 같은 키의 후보 dict 를 돌려준다."""
        return dict(len=pr_len, stb=pr_stb, port=pr_port, ck=pr_ck, mind=pr_mind, fm=pr_fm,
                    dpsi=pr_dpsi, dpsi_min=pr_dpsi_min, dpsi_max=pr_dpsi_max, off_max=pr_off_max, off_abs=pr_off_abs,
                    off_sum=pr_off_sum, off_cnt=pr_off_cnt, spmin=pr_spmin, cpa=pr_cpa, far=pr_far)

    def _pair_advance(_pa, _pwe, sr, goal):
        """활성 쌍 _pa [E,N,N] 의 이번 결정 누적 — 옛 `if bool(pr_active.any()):` 블록 안의 순수 계산. 새 텐서만 만들어
        dict 로 돌려주고 이름 재바인딩(commit)은 호출자가 옛 코드와 같은 호스트 게이트(pr_active.any()) 뒤에 한다
        → 게이트가 False 인 스텝에는 옛 코드처럼 쌍 상태가 하나도 안 바뀐다(pr_cpa/pr_far 의 잔존값까지)."""
        _nr3 = (env.rudder / vg.MAX_TURN_RATE).unsqueeze(-1)      # [E,N,1] → broadcast
        _d3 = _pwe['dist']; _rt3 = _pwe['raw_tcpa']
        n_len = pr_len + _pa.to(pr_len.dtype)
        n_stb = pr_stb + ((_nr3 > EPS_S) & _pa).to(pr_len.dtype)
        n_port = pr_port + ((_nr3 < -EPS_P) & _pa).to(pr_len.dtype)
        n_ck = pr_ck + ((_nr3.abs() < EPS_C) & _pa).to(pr_len.dtype)
        _hnow = env.heading.unsqueeze(-1).expand(-1, -1, N)
        _dps = torch.remainder(_hnow - pr_hd0 + 180.0, 360.0) - 180.0   # [-180,180]
        n_dpsi = torch.where(_pa, _dps, pr_dpsi)
        n_dpsi_min = torch.where(_pa, torch.minimum(pr_dpsi_min, _dps), pr_dpsi_min)
        n_dpsi_max = torch.where(_pa, torch.maximum(pr_dpsi_max, _dps), pr_dpsi_max)
        _off = (-goal[..., 1] * 180.0).unsqueeze(-1).expand(-1, -1, N)   # [E,N,N] 목표 대비 우현 오프셋
        n_off_max = torch.where(_pa, torch.maximum(pr_off_max, _off), pr_off_max)
        n_off_abs = torch.where(_pa, torch.maximum(pr_off_abs, _off.abs()), pr_off_abs)
        n_off_sum = torch.where(_pa, pr_off_sum + _off.abs(), pr_off_sum)
        n_off_cnt = torch.where(_pa, pr_off_cnt + 1.0, pr_off_cnt)
        _sr3n = sr.unsqueeze(-1).expand(-1, -1, N)
        n_spmin = torch.where(_pa, torch.minimum(pr_spmin, _sr3n), pr_spmin)
        n_mind = torch.where(_pa, torch.minimum(pr_mind, _d3), pr_mind)
        _man = _pa & (pr_fm < 0) & (_nr3.abs() > PAIR_MANEUVER_THR)
        n_fm = torch.where(_man, n_len, pr_fm)
        n_cpa = torch.where(_pa & (_rt3 < 0), pr_cpa + 1.0, torch.zeros_like(pr_cpa))
        n_far = torch.where(_pa & (_d3 > PAIR_FAR_DIST), pr_far + 1.0, torch.zeros_like(pr_far))
        return dict(len=n_len, stb=n_stb, port=n_port, ck=n_ck, mind=n_mind, fm=n_fm,
                    dpsi=n_dpsi, dpsi_min=n_dpsi_min, dpsi_max=n_dpsi_max, off_max=n_off_max, off_abs=n_off_abs,
                    off_sum=n_off_sum, off_cnt=n_off_cnt, spmin=n_spmin, cpa=n_cpa, far=n_far)

    def pair_prepare(mask, S):
        """mask [E,N,N]: 방금 끝난 쌍 조우를 판정·집계 (5스텝 미만·burn-in 잔여 제외) — 옛 pair_finalize 의 동기화 없는 판.
        S = 쌍 상태 dict(_pr_state 또는 _pair_advance 후보). 역할별 개수는 packed 전송에 싣고, 호스트 사본이 오면
        pair_apply 가 레코드를 뽑는다."""
        m = mask & (S['len'] >= PAIR_MIN_LEN) & (~pr_skip)
        key = torch.where(m, pr_role, torch.zeros_like(pr_role)).view(-1)   # 0 = 집계 제외, 1..4 = 역할
        return key, torch.argsort(key, stable=True), _hist5(key, _ones_enn), S

    def pair_apply(prep, cnt):
        """prep = pair_prepare 반환값, cnt = 그 개수의 호스트 사본. [P,10] 레코드(역할 1,2,3,4 블록 순 — 옛 for k 루프와
        같은 순서, 블록 안은 행우선 = 옛 마스크 인덱싱 순서)를 device 리스트에 붙인다."""
        key, order, _, S = prep
        P = cnt[1] + cnt[2] + cnt[3] + cnt[4]
        if P == 0:
            return
        idx = order[cnt[0]:cnt[0] + P]
        L = S['len']
        # 옛 코드는 mk 로 뽑은 뒤 나눴고 여기선 전체에서 나눈 뒤 뽑는다 — 원소별 나눗셈·비교라 동일(mk 밖은 안 뽑힘)
        stb = S['stb'] / L; port = S['port'] / L; ck = S['ck'] / L
        ok = torch.where(pr_role == 2, ck > 0.6,                                     # 유지역할(R17)
                         torch.where(pr_role == 4, stb > 0.3, (stb > 0.5) & (port < 0.2)))   # 추월 / HeadOn·GiveWay
        off_avg = S['off_sum'] / S['off_cnt'].clamp(min=1.0)                        # 유지역할: 목표 방위 평균 이탈
        # ★규칙8: (우현 오프셋, 감속률, 진입DCPA) — 양보역할만 판정에 씀
        _drop = (1.0 - S['spmin'] / pr_sp0.clamp(min=1e-3)).clamp(min=0.0)
        cols = (pr_role.to(L.dtype), ok.to(L.dtype), S['mind'], S['fm'], off_avg, S['off_max'],
                S['dpsi'].abs(), L, _drop, pr_dcpa0)
        _pair_recs.append(torch.stack([c.reshape(-1)[idx] for c in cols], dim=1))   # [P,10]

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
        # ★2026-09-15: 진행 표시만. 루프 제어·집계에 관여하지 않음.
        if 0 < _dstep < args.eval_decisions and (args.burnin + _dstep) % _prog_every == 0:
            _progress(args.burnin + _dstep, '평가   ')
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
        #   ★T5: 옛 `if bool(_gate.any())` 게이트는 뺐다 — 마스크 합은 게이트가 비면 0 을 더할 뿐이고, 압축 합은
        #   아래 전송 뒤 길이 0 이면 건너뛴다(옛 코드와 같은 덧셈 순서).
        _pw = env._last_pw
        if _pw is not None:
            _mrisk = _pw['risk'].max(dim=-1).values                  # [E,N]
            _sit = env.situation                                     # [E,N] 0~4
            _rud = a[..., 0]                                         # [-1,1] >0=우현
            _gate = (_mrisk > 0.3) & (_sit > 0)                      # 조우 + 위험
            _star = ((_sit == 1) | (_sit == 3) | (_sit == 4)).to(r_dtype := ep_fuel.dtype)
            _hold = (_sit == 2).to(r_dtype)
            _viol = _star * torch.clamp(-_rud, min=0.0) + _hold * _rud.abs()
            _comp = (1.0 - _viol).clamp(0.0, 1.0)                # 1=완전준수
            _okv = _viol < 0.1
            _zs = torch.zeros_like(_sit)
            _sit_cnt = _hist5(torch.where(_gate, _sit, _zs).view(-1), _ones_en)   # [k] == 옛 int(_m.sum()); [0] = 게이트 밖
            _acc_sit_n.add_(_sit_cnt)
            _acc_sit_ok.add_(_hist5(torch.where(_gate & _okv, _sit, _zs).view(-1), _ones_en))
            _gorder = torch.argsort(_gate.view(-1).to(torch.int8), stable=True)   # False 앞, True 뒤(각각 행우선)
        # ── ★조율 진단·궤적 (2026-09-25): step 전 상태 + 이번 결정 ──
        if _dstep < args.eval_decisions:
            _coord_update(a)
            if _traj is not None:
                _traj['pos'].append(env.pos[:_te].clone()); _traj['heading'].append(env.heading[:_te].clone())
                _traj['speed'].append(env.speed[:_te].clone()); _traj['rudder'].append(env.rudder[:_te].clone())
                _traj['cmd'].append(a[:_te].clone()); _traj['situation'].append(env.situation[:_te].clone())
                _traj['goal'].append(env.goal[:_te].clone()); _traj['max_speed'].append(env.max_speed[:_te].clone())
        # ── ★조우 단위 누적 (tex 정의). step *직전* = 정책이 본 situation 과 현재 타각을 짝지음 ──
        _pwe = _pw                                   # env._last_pw 그대로(사이에 env 를 바꾸는 호출 없음 — comm_pair_features 는 읽기만)
        if _pwe is not None:
            _sn = env.situation
            _cont = (_sn == enc_sit) & (_sn > 0)
            _fin = (enc_sit > 0) & (~_cont)          # 방금 끝난 조우
            _new = (_sn > 0) & (~_cont)              # 방금 시작한 조우
            _encF = enc_prepare(_fin)                # 옛 enc_finalize(_fin): 수·판정은 device 누적, 최소거리 합은 전송 뒤 enc_apply
            # 리셋: 옛 `if bool(_rst.any())` 게이트는 뺐다 — where 는 마스크가 비면 항등
            _rst = _fin | _new
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
            _cand = (_nrk3 > PAIR_ENTRY_RISK) & (_sit3 > 0)
            pr_in = torch.where(_cand & (~pr_active), pr_in + 1.0, torch.zeros_like(pr_in))
            _start = (pr_in >= PAIR_DEBOUNCE) & (~pr_active)
            # 시작 쌍 초기화: 옛 `x[_start] = v` (in-place, `if _start.any()` 게이트) → where 재바인딩. 같은 위치에 같은 값,
            #   마스크가 비면 항등이라 게이트 불요. (in-place 를 버려야 아래 prepare 가 잡아둔 텐서가 뒤에 안 바뀐다)
            _z3f = torch.zeros_like(pr_len)
            pr_role = torch.where(_start, _sit3, pr_role)
            pr_len = torch.where(_start, _z3f, pr_len); pr_stb = torch.where(_start, _z3f, pr_stb)
            pr_port = torch.where(_start, _z3f, pr_port); pr_ck = torch.where(_start, _z3f, pr_ck)
            pr_mind = torch.where(_start, torch.full_like(pr_mind, BIG), pr_mind)
            pr_fm = torch.where(_start, torch.full_like(pr_fm, -1.0), pr_fm)
            pr_hd0 = torch.where(_start, env.heading.unsqueeze(-1), pr_hd0)
            pr_dpsi = torch.where(_start, _z3f, pr_dpsi); pr_dpsi_min = torch.where(_start, _z3f, pr_dpsi_min)
            pr_dpsi_max = torch.where(_start, _z3f, pr_dpsi_max)
            pr_off_max = torch.where(_start, torch.full_like(pr_off_max, -999.0), pr_off_max)
            pr_off_abs = torch.where(_start, _z3f, pr_off_abs)
            _sr3 = sr.unsqueeze(-1)
            pr_sp0 = torch.where(_start, _sr3, pr_sp0); pr_spmin = torch.where(_start, _sr3, pr_spmin)
            pr_dcpa0 = torch.where(_start, _pwe['dcpa'], pr_dcpa0)
            pr_off_sum = torch.where(_start, _z3f, pr_off_sum); pr_off_cnt = torch.where(_start, _z3f, pr_off_cnt)
            pr_cpa = torch.where(_start, _z3f, pr_cpa); pr_far = torch.where(_start, _z3f, pr_far)
            pr_in = torch.where(_start, _z3f, pr_in)
            pr_active = pr_active | _start
            # 활성 쌍 갱신 *후보* (commit 은 전송 뒤 호스트 게이트 안에서) + 종료 판정·집계 준비
            _adv = _pair_advance(pr_active, _pwe, sr, goal)
            _endp = pr_active & ((_adv['cpa'] >= PAIR_END_CPA) | (_adv['far'] >= PAIR_END_FAR))
            _pairP = pair_prepare(_endp, _adv)
            # null 기준선·매칭 null 창 — 호스트 게이트값과 압축 길이만 준비(순수 계산). 상태 commit 은 전송 뒤.
            _noenc = ~pr_active.any(dim=-1)                       # [E,N] 활성 조우 0
            _h1 = env.heading                                     # [E,N]
            _nqnew = _noenc & (nq_len <= 0)
            _nql = torch.where(_noenc, nq_len + 1.0, torch.zeros_like(nq_len))      # 옛 :567 의 새 nq_len
            # _nhit 는 randint 로 갱신되기 *전* nq_tgt 로 계산해도 같다: _nqnew 행은 _nql=1 < 8 ≤ 새 목표이고
            #   옛 목표는 0(→ nq_tgt>0 실패) 또는 ≥8 이라 어느 쪽이든 False.
            _nhit = _noenc & (_nql >= nq_tgt) & (nq_tgt > 0)
            _bkt = (_nql >= 20.0).long() + (_nql >= 40.0).long() + (_nql >= 80.0).long()   # NULLW_BUCKETS 인덱스 0..3
            _nkey = torch.where(_nhit, _bkt + 1, torch.zeros_like(_bkt)).view(-1)          # 0 = 미도달
            _norder = torch.argsort(_nkey, stable=True)
            # ★20스텝마다·상한 5만개만 표집 (매 스텝 GPU→CPU 전송은 평가를 3배 느리게 만든다)
            _samp = (_dstep % 20 == 0) and _null_off_n < 50000
            if _samp:
                _goff = (-goal[..., 1] * 180.0).abs()
                _oorder = torch.argsort(_noenc.view(-1).to(torch.int8), stable=True)
            # ── ★step 전 packed 전송 (스텝당 1회). 제어흐름·난수 게이트 2개 + 압축 길이들을 한 번에 ──
            _parts = [_sit_cnt, _encF[2], pr_active.any().reshape(1).long(), _nqnew.any().reshape(1).long(),
                      _pairP[2], _hist5(_nkey, _ones_en)]
            if _samp:
                _parts.append(_noenc.sum().reshape(1))
            _h = torch.cat(_parts).tolist()
            _sitc, _encc, _pa_any, _nq_any, _pairc, _nullc = _h[0:5], _h[5:10], _h[10], _h[11], _h[12:17], _h[17:22]
            _n_noenc = _h[22] if _samp else 0
            # ── 전송 뒤: 압축 합·commit — 옛 코드와 같은 순서 (comp → enc → 쌍 → null → 쌍 종료) ──
            _ng = E * N - _sitc[0]                                # == 옛 int(_gate.sum()) (게이트 ⇒ _sit∈1..4)
            if _ng:
                _acc_comp_sum.add_(_compact(_comp.view(-1), _gorder, E * N - _ng, _ng).sum().double())
            enc_apply(_encF, _encc)
            if _pa_any:                                           # == 옛 `if bool(pr_active.any())`: False 면 쌍·null 상태 전부 불변
                pr_len, pr_stb, pr_port, pr_ck = _adv['len'], _adv['stb'], _adv['port'], _adv['ck']
                pr_mind, pr_fm = _adv['mind'], _adv['fm']
                pr_dpsi, pr_dpsi_min, pr_dpsi_max = _adv['dpsi'], _adv['dpsi_min'], _adv['dpsi_max']
                pr_off_max, pr_off_abs, pr_off_sum, pr_off_cnt = _adv['off_max'], _adv['off_abs'], _adv['off_sum'], _adv['off_cnt']
                pr_spmin, pr_cpa, pr_far = _adv['spmin'], _adv['cpa'], _adv['far']
                # null 기준선: 활성 조우가 하나도 없는 배의 |오프셋| (배경 잡음)
                if _samp and _n_noenc:                            # == 옛 `if bool(_noenc.any())`
                    _null_off_dev.append(_compact(_goff.view(-1), _oorder, E * N - _n_noenc, _n_noenc))
                    _null_off_n += _n_noenc
                # ★매칭 null 창: 무조우 상태가 이어지는 동안 침로변화를 누적, 목표길이에 닿으면 기록.
                #   목표길이는 실제 조우 길이 분포(pr_len)에서 표집 → 창 길이 교란 제거.
                if _nq_any:                                       # == 옛 `if bool(_new.any())` — randint(기본 CUDA generator =
                    nq_h0 = torch.where(_nqnew, _h1, nq_h0)       #   정책 샘플과 공유)는 정확히 옛 조건·옛 위치(act 뒤, env.step 앞)에서만
                    # 관측된 조우 길이에서 표집(없으면 30). 버킷을 고르게 채우려 지수분포 근사
                    _t = torch.randint(8, 140, (E, N), device=dev).to(nq_tgt.dtype)
                    nq_tgt = torch.where(_nqnew, _t, nq_tgt)
                nq_len = _nql
                if _nullc[1] or _nullc[2] or _nullc[3] or _nullc[4]:      # == 옛 `if bool(_hit.any())`
                    _dpn = (torch.remainder(_h1 - nq_h0 + 180.0, 360.0) - 180.0).abs().view(-1)
                    _off = _nullc[0]
                    for _bi in range(len(NULLW_BUCKETS)):
                        _c = _nullc[_bi + 1]
                        # 옛 규칙 그대로: 붙이기 *전* 길이 < 60000 이면 버킷 전체를 붙인다(상한을 조금 넘겨 끝남)
                        if _c and _null_dpsi_n[_bi] < 60000:
                            _null_dpsi_dev[_bi].append(_compact(_dpn, _norder, _off, _c)); _null_dpsi_n[_bi] += _c
                        _off += _c
                    nq_len = torch.where(_nhit, torch.zeros_like(nq_len), nq_len)
                pair_apply(_pairP, _pairc)
                pr_active = pr_active & (~_endp)
                pr_skip = pr_skip & (~_endp)

        # ── step ──
        obs, rew_step, done, outcome = env.step(a)
        ep_reward += rew_step
        if _traj is not None and _dstep < args.eval_decisions:
            _traj['outcome'].append(outcome[:_te].clone())
        if _dstep < args.eval_decisions and _cctx:
            _cg_ = counted if _pending is None else (counted & _pending)
            _hit = (outcome == vg.OUT_COLLISION_VESSEL) & _cg_
            # ★T5: 옛 코드는 _hit 로 뽑은 뒤 분류했고 여기선 전체에서 분류해 마스크 합 — 0/1 합이라 동일, 동기화 없음
            _multi = _hit & (_cctx['n_risky'] >= 2)
            _one = _hit & ~(_cctx['n_risky'] >= 2)
            _m, _t = _cctx['myr'], _cctx['thr']
            _si, _hi, _sj, _hj = _cctx['stb_i'], _cctx['hold_i'], _cctx['stb_j'], _cctx['hold_j']
            GW, SO, HO, OT = vg.SIT_GIVEWAY, vg.SIT_STANDON, vg.SIT_HEADON, vg.SIT_OVERTAKING
            _acc_coll.add_(torch.stack([                                   # _COLL_K 순서
                _multi.sum(),
                (_one & (_m == GW) & (~_si)).sum(),                         # gw_noact
                (_one & (_m == GW) & _si & _hj).sum(),                      # gw_act_so_hold
                (_one & (_m == GW) & _si & (~_hj)).sum(),                   # gw_act_so_turn
                (_one & (_m == SO) & (~_hi)).sum(),                         # so_act
                (_one & (_m == SO) & _hi & (~_sj)).sum(),                   # so_hold_gw_noact
                (_one & (_m == SO) & _hi & _sj).sum(),                      # so_hold_gw_act
                (_one & (_m == HO) & _si & _sj).sum(),                      # ho_both_stb
                (_one & (_m == HO) & ~(_si & _sj)).sum(),                   # ho_not_both
                (_one & ((_m == OT) | ((_m == vg.SIT_NONE) & (_t == OT)))).sum(),   # overtaking
                (_one & (_m == vg.SIT_NONE) & (_t != OT)).sum()]))          # no_role
        # ── 종료 에피소드: 지표 기록 후 리셋 ──
        # ★2026-09-05 fix: drain 구간(_pending is not None)에서는 '창 끝에 걸려 있던' 에피소드만
        #   집계에 넣는다. 안 그러면 이미 마감된 배가 새로 시작한 에피소드까지 딸려 들어와
        #   창이 팔마다 다른 길이로 늘어난다. drain 이 꺼져 있으면 _pending is None → 기존 동작 그대로.
        _cnt_gate = counted if _pending is None else (counted & _pending)
        _okey = torch.where(_cnt_gate, outcome, torch.zeros_like(outcome)).view(-1)   # 0 = 진행 중 또는 게이트 밖
        _ocnt = _hist5(_okey, _ones_en)                                     # [oc] == 옛 int(mask.sum())
        _oorder2 = torch.argsort(_okey, stable=True)
        term = (outcome != 0)
        _rec = term & _cnt_gate
        _rorder = torch.argsort(_rec.view(-1).to(torch.int8), stable=True)
        # ★조우·쌍 마감 준비는 리셋(아래 where 재바인딩) *전* 값으로 — apply 는 전송 뒤(잡아둔 텐서는 안 바뀜)
        _encT = enc_prepare(term & (enc_sit > 0))
        _t3 = term.unsqueeze(-1) | term.unsqueeze(1)          # [E,N,N]
        _endt = pr_active & _t3
        _pairT = pair_prepare(_endt, _pr_state())
        # ── ★step 후 packed 전송 (스텝당 1회): outcome 별 개수 + 조우·쌍 역할별 개수 ──
        _h = torch.cat([_ocnt, _encT[2], _pairT[2]]).tolist()
        _oc, _encc, _pairc = _h[0:5], _h[5:10], _h[10:15]
        # counts 는 옛대로 float32 텐서에 정수를 더한다(0 을 더하면 항등이라 c==0 인 oc 도 같음)
        counts[1:].add_(_ocnt[1:].to(counts.dtype))
        _msc = ep_minsep.clamp(max=vg.RADAR_RANGE * 4).reshape(-1)
        _off = _oc[0]
        for oc in range(1, 5):
            c = _oc[oc]                                # (outcome == oc) & _cnt_gate — 경계 걸친 첫 에피소드 제외
            if c:
                total += c; msum[oc]['n'] += c
                for _fi, _fv in enumerate((ep_fuel, ep_head, _msc, ep_len, ep_reward)):
                    _acc_msum[oc, _fi].add_(_compact(_fv.reshape(-1), _oorder2, _off, c).sum().double())
            _off += c
        # 옛 `if int(term.sum())` 게이트는 뺐다 — 안의 where·마스크 합·prepare 는 term 이 비면 전부 항등/0
        _nrec = _oc[1] + _oc[2] + _oc[3] + _oc[4]      # == 옛 int(_rec.sum()) (outcome≠0 ⇔ oc∈1..4)
        if _nrec:
            _acc_all_minsep.add_(_compact(_msc, _rorder, E * N - _nrec, _nrec).sum().double()); all_minsep_n += _nrec
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
        enc_apply(_encT, _encc)
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
        pair_apply(_pairT, _pairc)
        pr_active = pr_active & (~_endt)
        pr_skip = pr_skip & (~_t3)
        pr_in = torch.where(_t3, torch.zeros_like(pr_in), pr_in)
        radar, goal, self_s, sit = parse_obs(obs); fs.push(radar, done)
        prev_head = torch.where(done, env.heading, prev_head)

    # ★T5: device 누적기 → 옛 호스트 변수 (여기서부터 출력부는 글자 하나 안 바뀜). 값은 정수(정확) 또는
    #   옛 코드와 같은 순서로 더한 float64 라 동일. 리스트 수집기는 device 조각을 옛 append 순서로 이어 붙인 것.
    _sn_, _so_ = _acc_sit_n.tolist(), _acc_sit_ok.tolist()
    for _k in (1, 2, 3, 4):
        sit_n[_k] = _sn_[_k]; sit_ok[_k] = float(_so_[_k])
    comp_n = sum(_sn_[1:]); comp_bin = float(sum(_so_[1:])); comp_sum = float(_acc_comp_sum)
    _ms_ = _acc_msum.tolist()
    for oc in range(1, 5):
        for _fi, _fn in enumerate(('fuel', 'head', 'minsep', 'length', 'reward')):
            msum[oc][_fn] = _ms_[oc][_fi]
    all_minsep_sum = float(_acc_all_minsep)
    _et_, _ed_ = _acc_enc_tot.tolist(), _acc_enc_dsum.tolist()
    _eo_ = {_src: _acc_enc_ok[_src].tolist() for _src in ('act', 'cmd')}
    for _k in (1, 2, 3, 4):
        enc_tot[_k] = _et_[_k]; enc_dsum[_k] = _ed_[_k]
        for _src in ('act', 'cmd'):
            enc_ok[_src][_k] = _eo_[_src][_k]
    for _band in ('near', 'far'):
        coord[_band] = {k: float(v) for k, v in zip(_CK, _acc_coord[_band].tolist())}
    coll_cat = dict(zip(_COLL_K, _acc_coll.tolist()))
    null_off_abs = torch.cat(_null_off_dev).tolist() if _null_off_dev else []
    for _bi in null_dpsi:
        null_dpsi[_bi] = torch.cat(_null_dpsi_dev[_bi]).tolist() if _null_dpsi_dev[_bi] else []
    for _row in (torch.cat(_pair_recs).tolist() if _pair_recs else []):
        _k = int(_row[0])
        pair_res[_k].append((bool(_row[1]), _row[2], _row[3]))
        if _k == 2:   # 유지역할(R17): 목표 방위에서 얼마나 벗어났나 / null 대조용 (|Δψ|, 창 길이)
            hd_res[2].append(_row[4]); sd_dpsi.append((_row[6], _row[7]))
        else:         # 양보역할: 목표 방위 대비 최대 우현 오프셋
            hd_res[_k].append(_row[5])
        r8_res[_k].append((_row[5], _row[8], _row[9]))

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

    # ── ★조율 진단 출력 (2026-09-25) — 기전 설명용, 사전등록 H1 판정 아님 ──
    for _band, _lbl in (('near', f'레이더 안 ≤{vg.RADAR_RANGE:g}m'), ('far', f'레이더 밖 {vg.RADAR_RANGE:g}-{cfg.COMM_RANGE:g}m')):
        _c = coord[_band]
        _g = _c['gwso']
        _pc = (lambda k: f"{100 * _c[k] / _g:5.1f}%") if _g > 0 else (lambda k: '  -  ')
        print(f"   [조율/{_lbl}] 양보-유지 쌍결정 n={int(_g)}  보완(양보 우현+유지 유지)={_pc('comp')}  "
              f"양보 우현={_pc('gw_stb')}  유지 유지={_pc('so_hold')}  유지 좌현={_pc('so_port')}  둘 다 유지={_pc('both_hold')}")
        _h = _c['ho']
        _r = _c['roled']
        print(f"   [조율/{_lbl}] HeadOn 쌍결정 n={int(_h)}  둘 다 우현="
              + (f"{100 * _c['ho_both_stb'] / _h:5.1f}%" if _h > 0 else '  -  ')
              + f"  | 역할 붙은 쌍 n={int(_r)}  역할 맞물림="
              + (f"{100 * _c['agree'] / _r:5.1f}%" if _r > 0 else '  -  '))
    _nc = sum(coll_cat.values())
    if _nc > 0:
        _p = lambda k: f"{100 * coll_cat[k] / _nc:4.1f}%"               # noqa: E731
        print(f"   [조율/충돌 분해] 선박충돌 배 n={_nc} (직전 결정의 가장 가까운 상대 기준, 실제 타각)  다선(위험 상대≥2)={_p('multi')}")
        print(f"   [조율/충돌 분해] 내가 양보선: 미행동={_p('gw_noact')} 행동+상대 유지={_p('gw_act_so_hold')} "
              f"행동+상대도 회피={_p('gw_act_so_turn')} | 내가 유지선: 회피={_p('so_act')} "
              f"고수+상대 미행동={_p('so_hold_gw_noact')} 고수+상대 행동={_p('so_hold_gw_act')}")
        print(f"   [조율/충돌 분해] HeadOn 둘 다 우현={_p('ho_both_stb')} 아님={_p('ho_not_both')} | 추월={_p('overtaking')} 역할 없음={_p('no_role')}")
    else:
        print("   [조율/충돌 분해] 집계 창 안 선박 충돌 없음")
    if _traj is not None and _traj['pos']:
        _out = {k: torch.stack(v).cpu() for k, v in _traj.items() if v}
        _out['meta'] = {'ckpt': os.path.basename(ckpt_path), 'arm': args.arm, 'comm_groups': list(net.COMM_GROUPS),
                        'latent_zero': bool(args.latent_zero), 'field_shuffle': bool(args.field_shuffle),
                        'seed': int(args.seed), 'dt_decision': 0.4, 'envs': int(_te), 'vessels': int(N),
                        'radar_range': float(vg.RADAR_RANGE), 'comm_range': float(cfg.COMM_RANGE)}
        torch.save(_out, args.traj_out)
        print(f"   [traj] {args.traj_out} ← {tuple(_out['pos'].shape)} (결정×env×선박×2)", flush=True)

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
