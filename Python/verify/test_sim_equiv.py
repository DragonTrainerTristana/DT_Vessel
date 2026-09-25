"""test_sim_equiv.py — vessel_gym 시뮬레이터 옛(커밋 da23ad0) vs 현재 코드 **비트동일** 검사 (2026-09-26).

왜 있나
  2026-09-26 perf 패치(호스트 동기 1회/스텝 · _radar 타깃 축 배치 · 리셋 행만 레이더 재계산 · 죽은 연산 제거)는
  "결과가 같은 float, 같은 난수 소비, 같은 상태" 여야 한다. test_golden 은 CPU·E=8·grid3x3 만 덮으므로
  이 테스트가 OBSTACLES=none · E∈{8,128,256} · CUDA 까지 옛 코드와 직접 대조한다.

방법
  참조 = git 오브젝트에서 그대로 읽은 옛 vessel_gym.py 를 별도 모듈로 exec (저장소에 사본을 두지 않음).
  같은 seed·같은 무작위 행동으로 reset + STEPS 스텝을 두 env 에 굴리며 매 스텝 obs/reward/done/outcome/situation/
  danger_idx/_last_pw(None 아닌 키)/내부 상태/generator state 를 비트 단위(float 는 int 로 view) 비교.
  + _respawn 단독(무작위 mask · all-False 조기 return · 단일 True · 전부 True) + _radar 부분 행 == 전체 행 슬라이스.

쓰는 법
  C:/Users/OSH/anaconda3/envs/mltest/python.exe verify/test_sim_equiv.py            # cpu + (cuda 가능 시) 전부
  ... --steps 100 --device cpu                                                        # 빠르게 / 한 장치만
  VESSEL_SIM_EQUIV_REF=<rev> 로 참조 커밋 변경 (기본 da23ad0 = 패치 직전).
"""
import argparse
import os
import subprocess
import sys
import time
import types
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(HERE)
sys.path.insert(0, PYROOT)

import torch  # noqa: E402

import vessel_gym as vg  # noqa: E402

REF_REV = os.environ.get('VESSEL_SIM_EQUIV_REF', 'da23ad0')
N_VESSELS = 16
SEED = 7
# 비교하는 env 내부 상태 (prev_far_risk 는 far-field 계수가 켜진 케이스에서만 — 꺼지면 현재 코드가 안 씀, _reward #8 주석)
STATE_ATTRS = ('pos', 'heading', 'speed', 'rudder', 'cmd_rudder', 'target_speed', 'max_speed', 'goal',
               'prev_dist', 'prev_rudder', 'prev_dcpa', 'prev_danger_idx', 'danger_idx', 'step_count',
               'situation', 'dropout_left', 'spawn_idx')
PW_KEYS = ('dist', 'risk', 'near_risk', 'sit', 'tcpa', 'raw_tcpa', 'dcpa', 'far_risk')


def load_ref():
    """옛 vessel_gym.py 를 git 오브젝트에서 읽어 독립 모듈로 만든다 (config 는 현재 것을 공유 — 같은 프로세스)."""
    src = subprocess.check_output(['git', 'show', f'{REF_REV}:Python/vessel_gym.py'], cwd=PYROOT)
    mod = types.ModuleType('vg_ref')
    mod.__file__ = os.path.join(PYROOT, f'vessel_gym@{REF_REV}.py')
    exec(compile(src, f'<{REF_REV}:Python/vessel_gym.py>', 'exec'), mod.__dict__)
    return mod


def _bits(t):
    """float 텐서를 같은 폭의 int 로 view → 부호 있는 0·NaN 페이로드까지 구분하는 비트 비교."""
    if t.dtype == torch.float32:
        return t.contiguous().view(torch.int32)
    if t.dtype == torch.float64:
        return t.contiguous().view(torch.int64)
    if t.dtype == torch.float16 or t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16)
    return t


def assert_bit_equal(name, a, b):
    if a is None and b is None:
        return
    assert a is not None and b is not None, f'{name}: 한쪽만 None (ref={a is not None}, new={b is not None})'
    assert a.shape == b.shape and a.dtype == b.dtype, f'{name}: shape/dtype {tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}'
    if not torch.equal(_bits(a), _bits(b)):
        diff = (_bits(a) != _bits(b))
        n = int(diff.sum())
        idx = diff.nonzero()[:5].tolist()
        raise AssertionError(f'{name}: {n}/{a.numel()} 원소 불일치, 예 {idx} ref={a[diff][:5].tolist()} new={b[diff][:5].tolist()}')


def make_pair(vg_ref, device, E, obst, rng_const=False, **kw):
    """같은 설정으로 (ref, new) env 를 만든다. OBSTACLES_MODE/RESPAWN_RNG_CONST 는 양쪽 모듈 전역을 같이 바꾼다."""
    for m in (vg_ref, vg):
        m.OBSTACLES_MODE = obst
        m.RESPAWN_RNG_CONST = rng_const
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        env_r = vg_ref.VesselBatchEnv(num_envs=E, n_vessels=N_VESSELS, device=device, seed=SEED,
                                      crossing=0, reward_range=200.0, **kw)
        env_n = vg.VesselBatchEnv(num_envs=E, n_vessels=N_VESSELS, device=device, seed=SEED,
                                  crossing=0, reward_range=200.0, **kw)
    assert_bit_equal('obstacles', env_r.obstacles, env_n.obstacles)
    return env_r, env_n


def compare_state(env_r, env_n, tag, far_on):
    for a in STATE_ATTRS:
        assert_bit_equal(f'{tag} {a}', getattr(env_r, a), getattr(env_n, a))
    if far_on:
        assert_bit_equal(f'{tag} prev_far_risk', env_r.prev_far_risk, env_n.prev_far_risk)
    assert torch.equal(env_r.gen.get_state(), env_n.gen.get_state()), f'{tag} generator state 불일치 (난수 소비가 다름)'


def run_config(vg_ref, device, E, obst, steps, rng_const=False, **kw):
    tag = f'[{device} E={E} obst={obst}' + (' rng_const' if rng_const else '') + (f' {kw}' if kw else '') + ']'
    t0 = time.perf_counter()
    env_r, env_n = make_pair(vg_ref, device, E, obst, rng_const=rng_const, **kw)
    far_on = env_n.farfield_coef > 0.0 or env_n.farpair_coef != 0.0
    obs_r = env_r.reset(); obs_n = env_n.reset()
    assert_bit_equal(f'{tag} reset obs', obs_r, obs_n)
    compare_state(env_r, env_n, f'{tag} reset', far_on)
    agen = torch.Generator(device='cpu').manual_seed(123)
    n_reset_steps = 0
    n_done = 0
    for s in range(steps):
        act = (torch.rand(E, N_VESSELS, 2, generator=agen) * 2 - 1).to(device)
        o_r, r_r, d_r, oc_r = env_r.step(act)
        o_n, r_n, d_n, oc_n = env_n.step(act)
        st = f'{tag} step {s}'
        assert_bit_equal(f'{st} obs', o_r, o_n)
        assert_bit_equal(f'{st} reward', r_r, r_n)
        assert_bit_equal(f'{st} done', d_r, d_n)
        assert_bit_equal(f'{st} outcome', oc_r, oc_n)
        for k in PW_KEYS:                       # 옛 코드는 모든 키가 텐서; 새 코드는 far_risk 만 계수 OFF 면 None
            v_n = env_n._last_pw.get(k)
            if v_n is not None:
                assert_bit_equal(f'{st} _last_pw[{k}]', env_r._last_pw[k], v_n)
        compare_state(env_r, env_n, st, far_on)
        if bool(d_r.any()):
            n_reset_steps += 1
            n_done += int(d_r.sum())
    # 같은 최종 상태에서 전체 _radar 직접 대조 (step 안에서는 새 코드가 부분 행 경로를 탔으므로 전체 경로도 확인)
    assert_bit_equal(f'{tag} final _radar()', env_r._radar(), env_n._radar())
    dt = time.perf_counter() - t0
    print(f'  PASS {tag} steps={steps} 리셋 있는 스텝 {n_reset_steps}/{steps} (done {n_done}) {dt:.1f}s', flush=True)
    return env_r, env_n


def check_pw_keys(env_n, far_on):
    """_last_pw 키 집합 불변 (far_risk 는 계수 꺼지면 None)."""
    assert tuple(env_n._last_pw.keys()) == PW_KEYS, tuple(env_n._last_pw.keys())
    assert (env_n._last_pw['far_risk'] is not None) == far_on


def check_respawn_unit(env_r, env_n, tag, far_on):
    """_respawn 단독: 같은 상태·같은 generator 에서 mask 4종. all-False 는 둘 다 조기 return + generator 불변."""
    E, N = env_n.E, env_n.N
    mgen = torch.Generator(device='cpu').manual_seed(99)
    masks = [
        ('random30', torch.rand(E, N, generator=mgen) < 0.3),
        ('allFalse', torch.zeros(E, N, dtype=torch.bool)),
        ('single', torch.zeros(E, N, dtype=torch.bool).index_put_((torch.tensor([E - 1]), torch.tensor([N - 1])), torch.tensor(True))),
        ('allTrue', torch.ones(E, N, dtype=torch.bool)),
        ('random70', torch.rand(E, N, generator=mgen) < 0.7),
    ]
    for name, m in masks:
        m = m.to(env_n.device)
        g_before = env_n.gen.get_state().clone()
        env_r._respawn(m, initial=False)
        env_n._respawn(m, initial=False)
        compare_state(env_r, env_n, f'{tag} _respawn({name})', far_on)
        if name == 'allFalse':
            assert torch.equal(g_before, env_n.gen.get_state()), f'{tag} all-False mask 에서 generator 가 소비됨'
        # col_any 를 넘기는 step() 경로도 같은 결과여야 함
        env_r._respawn(m, initial=False)
        env_n._respawn(m, initial=False, col_any=m.any(dim=0).tolist())
        compare_state(env_r, env_n, f'{tag} _respawn({name}, col_any)', far_on)
    print(f'  PASS {tag} _respawn 단독 (mask 5종 × col_any 유/무, all-False 조기 return·generator 불변)', flush=True)


def check_radar_rows(env_n, tag):
    """새 코드의 부분 행 _radar(pos[rows], heading[rows]) == 전체 _radar()[rows] (step 의 index_copy 경로 근거)."""
    E = env_n.E
    full = env_n._radar()
    rgen = torch.Generator(device='cpu').manual_seed(5)
    subsets = [torch.tensor([0]), torch.tensor([E - 1]), torch.arange(E)]
    for k in (1, 2, 3, 5, 7):
        if k < E:
            subsets.append(torch.randperm(E, generator=rgen)[:k].sort().values)
    for rows in subsets:
        rows = rows.to(env_n.device)
        part = env_n._radar(env_n.pos.index_select(0, rows), env_n.heading.index_select(0, rows))
        assert_bit_equal(f'{tag} _radar rows={rows.tolist()[:8]}', full.index_select(0, rows), part)
    print(f'  PASS {tag} _radar 부분 행 == 전체 슬라이스 ({len(subsets)} 부분집합)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--device', default='all', help='cpu | cuda | all')
    ap.add_argument('--threads', type=int, default=2, help='CPU 스레드 (공유 서버 배려)')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    vg_ref = load_ref()
    devices = []
    if args.device in ('all', 'cpu'):
        devices.append('cpu')
    if args.device in ('all', 'cuda'):
        if torch.cuda.is_available():
            devices.append('cuda')
        elif args.device == 'cuda':
            raise SystemExit('cuda 사용 불가')
    print(f'=== vessel_gym 비트동일 검사: ref={REF_REV} vs 현재 | torch {torch.__version__} | devices={devices} | steps={args.steps}')
    saved = (vg.OBSTACLES_MODE, vg.RESPAWN_RNG_CONST)
    try:
        for dev in devices:
            for E in ((8, 128) if dev == 'cpu' else (8, 128, 256)):
                for obst in ('none', 'grid3x3'):
                    env_r, env_n = run_config(vg_ref, dev, E, obst, args.steps)
                    check_pw_keys(env_n, far_on=False)
                    if E == 8:
                        check_respawn_unit(env_r, env_n, f'[{dev} E={E} obst={obst}]', far_on=False)
                        check_radar_rows(env_n, f'[{dev} E={E} obst={obst}]')
                    if dev == 'cuda' and E == 256:
                        check_radar_rows(env_n, f'[{dev} E={E} obst={obst}]')
                    del env_r, env_n
            # far-field·perpair 계수 ON: far_risk 가 계산·비교되는 경로 (+ prev_far_risk)
            env_r, env_n = run_config(vg_ref, dev, 8, 'grid3x3', min(args.steps, 150),
                                      farfield_coef=0.5, perpair_coef=-0.15, perpair_exp=3.0,
                                      farpair_coef=-0.1, farpair_exp=1.6)
            check_pw_keys(env_n, far_on=True)
            del env_r, env_n
            # RESPAWN_RNG_CONST=1 경로 (r_all 사전 추첨)
            env_r, env_n = run_config(vg_ref, dev, 8, 'none', min(args.steps, 150), rng_const=True)
            check_respawn_unit(env_r, env_n, f'[{dev} E=8 rng_const]', far_on=False)
            del env_r, env_n
    finally:
        vg.OBSTACLES_MODE, vg.RESPAWN_RNG_CONST = saved
        vg_ref.OBSTACLES_MODE, vg_ref.RESPAWN_RNG_CONST = saved
    print('=== ALL PASS (bit-identical) ===')


if __name__ == '__main__':
    main()
