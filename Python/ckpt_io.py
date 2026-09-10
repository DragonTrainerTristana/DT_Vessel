"""ckpt_io.py — 체크포인트 저장·복원·평가 env 생성의 단일 구현 (2026-09-10).

왜 있나
  체크포인트를 굴리는 스크립트 9개 중 cfg_snapshot 을 읽는 건 eval_ckpt·eval_mixed 2개뿐이었고,
  나머지 7개 + runs/m2_ablation/diag/ 10개는 env 를 손으로 박아(VESSEL_USE_ATTENTION='0',
  VESSEL_COMM_RANGE='200', crossing=0) 학습과 다른 설정으로 쟀다 — 실제 런은 attention ON·300·crossing 2.
  측정 2회 무효의 직접 원인. eval_ckpt.py:83-180 의 복원 로직이 유일한 정본이었으나 main() 안
  인라인이라 재사용이 안 됐다 → 여기로 추출.

규약 (이 저장소의 진단·평가 전부에 적용)
  1. 체크포인트는 restore_policy() 로만 연다. VESSEL_* 를 스크립트에서 직접 세팅하지 않는다.
  2. 평가 env 는 make_env_from_snapshot() 으로만 만든다. 스냅샷과 다른 값은 명시 override 뿐이고 로그에 남는다.
  3. import 시점에 고정되는 값(comm_range·msg_dim 등)이 스냅샷과 어긋나면 기본은 **중단**한다.
     조용히 다른 실험을 재는 것보다 낫다. 의도한 교차평가만 allow_* 로 푼다.

순서가 중요하다 (eval_ckpt.py 2026-09-05~07 의 fix 를 그대로 보존)
  torch.load → msg_ln/msg_dim/head 스니핑 → networks 모듈 전역 덮어쓰기 → **그 다음** CNNPolicy() → strict 로드.
  CNNPolicy.__init__ 이 모듈 전역을 그 시점에 읽으므로 순서를 바꾸면 스냅샷이 무효가 된다.
"""
import os
import sys

import torch

import config as cfg
import networks as net
import vessel_gym as vg
from networks import CNNPolicy


# ──────────────────────────────────────────────────────────────────────────────
# 저장 쪽 — 학습기가 체크포인트에 넣는 설정 스냅샷
# ──────────────────────────────────────────────────────────────────────────────
def snapshot_config(*, arm, msg_dim, seed, n_envs, n_vessels, max_partners, trunc_boot,
                    comm_on_at=0, ring=None, crossing=None, rollout=None, trainer='gym'):
    """학습 설정 스냅샷. vessel_gym_train.py 의 _cfg_snapshot()(2026-09-05) 을 승격한 것.

    가중치에 흔적이 남지 않는 값(활성함수·집계 방식·token gain·보상 계수 등)을 전부 기록한다.
    평가·진단은 이걸 읽어 학습과 같은 조건으로 복원한다. 키 추가는 자유(구 로더는 모르는 키를 무시),
    키 삭제·의미 변경은 금지(구 체크포인트 복원이 깨진다).
    ring/crossing/rollout 은 gym 학습기 전용 — Unity(main.py) 는 None 으로 둔다.
    """
    return {
        'arm': arm, 'msg_dim': int(msg_dim), 'trainer': trainer,
        'use_attention': bool(cfg.USE_ATTENTION), 'pos_ground': bool(cfg.POS_GROUND),
        'central_critic': bool(cfg.CENTRAL_CRITIC), 'state_recon_coef': float(cfg.STATE_RECON_COEF),
        'use_moe': bool(cfg.USE_MOE), 'moe_shared': bool(cfg.MOE_SHARED), 'moe_width': float(cfg.MOE_WIDTH),
        'msg_ln': bool(net.MSG_LN),
        'comm_range': float(cfg.COMM_RANGE), 'max_partners': int(max_partners),
        'comm_on_at': int(comm_on_at),
        'ring': None if ring is None else float(ring), 'crossing': None if crossing is None else int(crossing),
        'vessels': int(n_vessels), 'envs': int(n_envs),
        'rollout': None if rollout is None else int(rollout), 'seed': int(seed),
        'msg_random_sd': float(cfg.MSG_RANDOM_SD) if arm == 'RANDOM' else None,
        # 레이더 인코더: 활성함수는 가중치에 안 남고, head 는 키로만 구분된다.
        'radar_act': 'leaky' if net._RADAR_LEAKY else 'relu',
        'radar_head': net._RADAR_HEAD,
        'radar_bottleneck_ch': int(net._RADAR_BOTTLENECK_CH),
        # 가중치에 흔적이 안 남는 값들 — 스냅샷이 유일한 근거
        'msg_token_gain': float(net._MSG_TOKEN_GAIN),
        'clip_per_module': os.environ.get('VESSEL_CLIP_PER_MODULE', '0') == '1',
        'msg_l2_coef': float(cfg.MSG_L2_COEF),
        'recon_ema_floor': float(net._RECON_EMA_FLOOR),
        'comm_telemetry': os.environ.get('VESSEL_COMM_TELEMETRY', '0') == '1',
        'agg_mode': net.AGG_MODE,
        'msg_gain': float(net.MSG_GAIN),
        'timeout_bootstrap': trunc_boot,
        # ★2026-09-10 추가: env 보상 계수. 학습기는 env 로 읽는데 스냅샷에 없어서 평가·진단이 각자 리터럴을 박았다.
        'farfield_coef': float(os.environ.get('VESSEL_FARFIELD_COEF', '0.0')),
        'perpair_coef': float(os.environ.get('VESSEL_PERPAIR_COEF', '-0.15')),
        'perpair_exp': 3.0,
        'radar_range': float(vg.RADAR_RANGE),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 복원 쪽
# ──────────────────────────────────────────────────────────────────────────────
class Restored:
    """restore_policy() 결과. policy 외에 '실제로 적용된 설정' 을 들고 다닌다."""
    __slots__ = ('policy', 'snap', 'raw', 'state_dict', 'msg_dim', 'arm', 'max_partners',
                 'effective', 'notes', 'path')

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def header(self):
        """결과 파일 첫 줄에 박을 한 줄 요약."""
        e = self.effective
        return (f"ckpt={os.path.basename(self.path)} arm={self.arm} msg_dim={self.msg_dim} "
                f"attention={e['use_attention']} pos_ground={e['pos_ground']} central_critic={e['central_critic']} "
                f"state_recon={e['state_recon_coef']} radar={e['radar_head']}/{e['radar_act']} "
                f"msg_ln={e['msg_ln']} token_gain={e['msg_token_gain']} agg={e['agg_mode']} msg_gain={e['msg_gain']} "
                f"comm_range={e['comm_range']} max_partners={self.max_partners} snapshot={'yes' if self.snap else 'NO'}")


def _say(tag, msg):
    print(f"{tag} {msg}", flush=True)


def restore_policy(ckpt_path, device, *, arm=None, max_partners=None,
                   allow_arm_mismatch=False, allow_comm_range_mismatch=False, tag='[ckpt]'):
    """체크포인트를 열어 학습 때와 같은 구조·설정으로 CNNPolicy 를 만든다.

    Args:
        ckpt_path: .pt 경로. 상대경로면 VESSEL_CKPT_DIR(기본 이 파일 옆 checkpoints/) 기준.
        device: 'cpu' | 'cuda' | 'cuda:1'
        arm: 평가 arm. None 이면 스냅샷의 arm 을 쓴다. 스냅샷과 다르면 기본 중단.
        max_partners: None 이면 스냅샷 값. 다르면 스냅샷 값으로 맞추고 경고(eval_ckpt 와 동일).
        allow_arm_mismatch: 의도한 교차평가(ON 정책의 통신을 끊어 재기 등)만 True.
        allow_comm_range_mismatch: comm_range 는 import 시점에 config/vessel_gym 에 고정된다.
            스냅샷과 다르면 relpos 정규화·파트너 선택·보상반경이 전부 달라지므로 기본 중단.
            True 면 경고만 (과거 숫자 재현 목적).
    Returns:
        Restored — .policy(eval 모드), .snap, .effective(실제 적용 설정), .notes(경고 목록)
    """
    notes = []
    scr = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.environ.get('VESSEL_CKPT_DIR', os.path.join(scr, 'checkpoints'))
    path = ckpt_path if os.path.isabs(ckpt_path) else os.path.join(ckpt_dir, ckpt_path)

    sd = torch.load(path, map_location=device)
    _sd = sd['model_state_dict'] if isinstance(sd, dict) and 'model_state_dict' in sd else sd

    # 1) msg_ln — CNNPolicy.__init__ 이 생성 시점에 env 를 읽으므로 먼저 맞춘다
    has_ln = any('msg_ln' in k for k in _sd)
    net.MSG_LN = has_ln          # 모듈 전역 (2026-09-10; 예전엔 env — import 후 env 는 효과 없음)

    # 2) msg_dim — cfg.MSG_DIM 은 import 시점 고정. ckpt 가 진실. make_others_msg 가 쓰는 모듈 전역도 동기화.
    _mk = [k for k in _sd if k.endswith('msg_out.weight')]
    msg_dim = int(_sd[_mk[0]].shape[0]) if _mk else cfg.MSG_DIM
    if msg_dim != cfg.MSG_DIM:
        import vessel_gym_train as _vgt
        _vgt.MSG_DIM = msg_dim
        _say(tag, f"ckpt msg_dim={msg_dim} (cfg={cfg.MSG_DIM}) - ckpt 값으로 로드")

    snap = sd.get('cfg_snapshot') if isinstance(sd, dict) else None
    sniff_cc = any(k.startswith('critic.') and 'glob_enc' in k for k in _sd)
    sniff_sr = any(k.startswith('state_recon') for k in _sd)

    # 3) 레이더 head 는 키로 확실히, 활성함수는 스냅샷으로만
    _rk = [k for k in _sd if k.endswith('radar_encoder.reduce.weight')]
    net._RADAR_HEAD = 'bottleneck' if _rk else 'flat'
    if _rk:
        net._RADAR_BOTTLENECK_CH = int(_sd[_rk[0]].shape[0])
    snap_act = (snap or {}).get('radar_act')
    if snap_act is not None:
        net._RADAR_LEAKY = (str(snap_act).lower() == 'leaky')
    elif os.environ.get('VESSEL_RADAR_ACT'):
        net._RADAR_LEAKY = os.environ['VESSEL_RADAR_ACT'].lower() == 'leaky'
        notes.append(f"스냅샷에 radar_act 없음 → env VESSEL_RADAR_ACT={os.environ['VESSEL_RADAR_ACT']} 사용. "
                     f"학습 때 값과 다르면 조용히 틀린 숫자")
    else:
        net._RADAR_LEAKY = False
        if snap is None:
            notes.append("스냅샷·env 둘 다 없어 radar_act=relu 가정")

    ck_arm = None
    if snap:
        # 4) 키로 구분 불가능한 구조 선택 + 가중치에 흔적 없는 값 — 모듈 전역 덮어쓰기
        net.USE_ATTENTION = bool(snap.get('use_attention', net.USE_ATTENTION))
        net.POS_GROUND = bool(snap.get('pos_ground', net.POS_GROUND))
        net.CENTRAL_CRITIC = bool(snap.get('central_critic', sniff_cc))
        net.STATE_RECON_COEF = float(snap.get('state_recon_coef', 1.0 if sniff_sr else 0.0))
        if snap.get('msg_token_gain') is not None:
            net._MSG_TOKEN_GAIN = float(snap['msg_token_gain'])
        # 집계 방식·이득·난수 sd: networks / vessel_gym_train 모듈 전역을 덮어쓴다 (env 는 import 시점에만 읽힘)
        if snap.get('agg_mode') is not None:
            net.AGG_MODE = str(snap['agg_mode']).lower()
        if snap.get('msg_gain') is not None:
            net.MSG_GAIN = float(snap['msg_gain'])
        if snap.get('msg_random_sd') is not None:
            import vessel_gym_train as _vgt
            _vgt.MSG_RANDOM_SD = float(snap['msg_random_sd'])
        # max_partners: 학습값이 진실
        ck_mp = snap.get('max_partners')
        if ck_mp is not None:
            if max_partners is not None and int(ck_mp) != int(max_partners):
                notes.append(f"max_partners 요청 {max_partners} ≠ 학습 {ck_mp} → 학습값으로 맞춤")
            max_partners = int(ck_mp)
        # comm_range: import 시점 고정값. 어긋나면 기본 중단.
        ck_cr = snap.get('comm_range')
        if ck_cr is not None and abs(float(ck_cr) - float(cfg.COMM_RANGE)) > 1e-6:
            msg = (f"ckpt 는 comm_range={ck_cr} 로 학습됐는데 현재 config/vessel_gym 은 {cfg.COMM_RANGE} 임. "
                   f"relpos 정규화·파트너 선택·보상반경이 전부 달라진다. "
                   f"VESSEL_COMM_RANGE={ck_cr:g} 를 주고 다시 실행할 것.")
            if not allow_comm_range_mismatch:
                raise SystemExit(f"{tag} 중단: {msg}")
            notes.append(msg + " (allow_comm_range_mismatch 로 진행)")
        # arm: 다르면 다른 실험
        ck_arm = snap.get('arm') or (sd.get('arm') if isinstance(sd, dict) else None)
        if arm is None:
            arm = ck_arm
        elif ck_arm and ck_arm != arm and not allow_arm_mismatch:
            raise SystemExit(f"{tag} 중단: ckpt 는 --arm {ck_arm} 로 학습됐는데 평가는 --arm {arm} 임. "
                             f"다른 실험을 재게 됨 (의도한 교차평가면 allow_arm_mismatch).")
    else:
        net.CENTRAL_CRITIC = sniff_cc
        net.STATE_RECON_COEF = 1.0 if sniff_sr else 0.0
        if arm is None:
            arm = sd.get('arm') if isinstance(sd, dict) else None
        notes.append(f"cfg_snapshot 없음(2026-09-05 이전 학습). central_critic={sniff_cc} state_recon={sniff_sr} 는 "
                     f"키로 스니핑했으나 **집계 방식(attention/pos_ground)은 키로 알 수 없음** — 현재 env 값 "
                     f"attention={net.USE_ATTENTION} pos_ground={net.POS_GROUND} 로 감. 학습 env 와 다르면 조용히 틀림")
    if max_partners is None:
        max_partners = int(cfg.MAX_COMM_PARTNERS)

    # 5) 이제 만든다 — 위 전역이 __init__ 에서 읽힌다
    policy = CNNPolicy(msg_dim, cfg.CONTINUOUS_ACTION_SIZE, cfg.FRAMES).to(device)
    policy.load_state_dict(_sd)
    policy.eval()

    effective = {
        'use_attention': bool(net.USE_ATTENTION), 'pos_ground': bool(net.POS_GROUND),
        'central_critic': bool(net.CENTRAL_CRITIC), 'state_recon_coef': float(net.STATE_RECON_COEF),
        'radar_head': net._RADAR_HEAD, 'radar_act': 'leaky' if net._RADAR_LEAKY else 'relu',
        'radar_bottleneck_ch': int(getattr(net, '_RADAR_BOTTLENECK_CH', 8)) if _rk else None,
        'msg_ln': bool(net.MSG_LN), 'msg_token_gain': float(net._MSG_TOKEN_GAIN),
        'agg_mode': net.AGG_MODE, 'msg_gain': float(net.MSG_GAIN),
        'msg_random_sd': (snap or {}).get('msg_random_sd'),
        'comm_range': float(cfg.COMM_RANGE), 'msg_dim': msg_dim, 'ckpt_arm': ck_arm,
    }
    r = Restored(policy=policy, snap=snap, raw=sd, state_dict=_sd, msg_dim=msg_dim, arm=arm,
                 max_partners=max_partners, effective=effective, notes=notes, path=path)
    _say(tag, r.header())
    for n in notes:
        _say(tag, f"[!] {n}")
    return r


# ──────────────────────────────────────────────────────────────────────────────
# 평가 env
# ──────────────────────────────────────────────────────────────────────────────
def make_env_from_snapshot(snap, *, device, num_envs, seed, n_vessels=None, ring=None, crossing=None,
                           farfield_coef=None, perpair_coef=None, perpair_exp=None, tag='[env]'):
    """학습 스냅샷과 같은 조건의 VesselBatchEnv. None 인 인자는 스냅샷에서, 스냅샷에도 없으면 학습기 기본값.

    기본값이 아닌 것(스냅샷과 다른 override)은 전부 로그에 찍는다. 스냅샷이 None 이면 ring·crossing 을 명시해야 한다.
    risk_range=reward_range=cfg.COMM_RANGE 는 학습기(vessel_gym_train.py:519)와 동일하게 고정.
    """
    snap = snap or {}
    src = {}

    def pick(name, given, snap_key, default):
        sv = snap.get(snap_key)
        if given is not None:
            # 명시값이 스냅샷과 같으면 'match'(조용), 다르면 'override'(경고)
            src[name] = 'match' if (sv is not None and float(given) == float(sv)) else 'override'
            return given
        if sv is not None:
            src[name] = 'snapshot'
            return sv
        src[name] = 'default'
        return default

    if not snap and (ring is None or crossing is None):
        raise SystemExit(f"{tag} 중단: 스냅샷이 없으면 ring·crossing 을 명시해야 함 (학습 조건을 모름)")

    n_vessels = int(pick('vessels', n_vessels, 'vessels', 16))
    ring = float(pick('ring', ring, 'ring', 1.0))
    crossing = int(pick('crossing', crossing, 'crossing', 0))
    ff = float(pick('farfield_coef', farfield_coef, 'farfield_coef', float(os.environ.get('VESSEL_FARFIELD_COEF', '0.0'))))
    pp = float(pick('perpair_coef', perpair_coef, 'perpair_coef', float(os.environ.get('VESSEL_PERPAIR_COEF', '-0.15'))))
    pe = float(pick('perpair_exp', perpair_exp, 'perpair_exp', 3.0))

    env = vg.VesselBatchEnv(num_envs=num_envs, n_vessels=n_vessels, device=device, seed=seed,
                            ring_scale=ring, crossing=crossing,
                            risk_range=cfg.COMM_RANGE, reward_range=cfg.COMM_RANGE,
                            farfield_coef=ff, perpair_coef=pp, perpair_exp=pe)
    used = (f"envs={num_envs} vessels={n_vessels} ring={ring} crossing={crossing} comm_range={cfg.COMM_RANGE} "
            f"farfield={ff} perpair={pp}^{pe} seed={seed}")
    ov = [k for k, v in src.items() if v == 'override']
    df = [k for k, v in src.items() if v == 'default']
    _say(tag, used + (f"  override={ov}" if ov else '') + (f"  default(스냅샷에 없음)={df}" if df else ''))
    for k in ov:
        sv = snap.get(k)
        if sv is not None:
            _say(tag, f"[!] {k}: 스냅샷 {sv} 대신 override 사용 — 학습과 다른 조건임")
    env.snapshot_sources = src
    return env


def describe(snap):
    """스냅샷 한 줄 요약(파일 헤더용)."""
    if not snap:
        return 'cfg_snapshot=NONE'
    keys = ('arm', 'msg_dim', 'use_attention', 'pos_ground', 'central_critic', 'state_recon_coef', 'use_moe',
            'msg_ln', 'comm_range', 'max_partners', 'ring', 'crossing', 'vessels', 'envs', 'seed',
            'radar_act', 'radar_head', 'msg_token_gain', 'agg_mode', 'msg_gain', 'recon_ema_floor',
            'clip_per_module', 'perpair_coef', 'farfield_coef')
    return ' '.join(f"{k}={snap[k]}" for k in keys if k in snap)


if __name__ == '__main__':
    # 간단 점검: 체크포인트 하나 열어서 헤더만 찍는다
    if len(sys.argv) < 2:
        print("사용: python ckpt_io.py <ckpt.pt> [device]")
        sys.exit(2)
    r = restore_policy(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'cpu')
    print(describe(r.snap))
