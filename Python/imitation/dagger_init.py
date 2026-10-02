"""dagger_init.py — imitation start for the OFF trunk (2026-10-02).
Spec: docs/superpowers/specs/2026-10-02-imitation-init-design.md (preregistration §5).

Why: the learned OFF of the t_ batch (53.1 % goal, 42.2 % vessel collision) is close to a ship that never avoids
(goal 50.3 / 49.7), while the scripted vo56 with the same 56 m range reaches 90.0 / 9.3 (rule-ship eval 2026-10-01).
This script starts the policy from the vo56 teacher instead of from random weights. PPO then runs unchanged.

Steps (one seed):
  1. DAgger into the ControlActor. At every decision the teacher vo56 labels the current state
     (imitation/vo_teacher.teacher_action). Each ship's episode is driven by the teacher with prob beta, else by the
     student's sampled action; beta = 1 for the first --beta_hold decisions, then falls linearly to 0 at --beta_end.
     Labelled states go to a ring buffer; --sgd_per_dec minibatches per decision.
     Loss = weighted MSE(tanh(mean), clip(a*, +-0.995)), weights [rudder 1, thrust --w_thrust]. logstd is not trained.
  2. Critic warm-up with the actor frozen: --value_updates rollouts of the student (sampled actions, rollout --rollout),
     GAE + ValueNorm exactly as vessel_gym_train, critic-only parameters (the radar encoder shared with the actor is
     left alone).
  3. Save a step-0 checkpoint in the trainer's own format (the --init dict with model_state_dict / value_norm replaced
     and optimizer_state_dict removed → the trunk starts with a fresh Adam). The trunk then runs
     `vessel_gym_train.py --arm OFF --resume <this> --resume_at 0 --steps 9043968 ...` with no trainer change.
     cfg_snapshot gets an `init_imitation` record; a sidecar <save>.imitation.json holds the same record.

The teacher sees only ships within 56 m (radar range), so no information beyond what OFF can sense is used.
Run under the same VESSEL_* exports as the batch (restore_policy checks them against the --init snapshot).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import config as cfg  # noqa: E402
import vessel_gym as vg  # noqa: E402
from vessel_gym_train import (parse_obs, FrameStack, make_others_msg, batched_gae, ValueNorm,  # noqa: E402
                              build_global_feat, comm_gather)
from ckpt_io import restore_policy, make_env_from_snapshot  # noqa: E402
from vo_teacher import TEACHER, TEACHERS, Teacher  # noqa: E402

TARGET_CLIP = 0.995      # tanh(3) ≈ 0.9951 = largest reachable |action| (mean clamp ±3)
AGREE_TOL = 0.2          # rudder agreement: |tanh(mean) − a*| < 0.2


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _git_head():
    try:
        return subprocess.check_output(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def _beta(t, hold, end):
    if t < hold:
        return 1.0
    if t >= end:
        return 0.0
    return 1.0 - (t - hold) / float(end - hold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True, help='step-0 checkpoint made by vessel_gym_train.py --steps 0 (same seed)')
    ap.add_argument('--save', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--teacher', default='vo56', choices=sorted(TEACHERS),
                    help='★2026-10-02 vo56(기본, 0a3e70e 와 같음) | vo56h150 | vo300i. 라벨 단위는 VESSEL_ACTION_MODE 를 따름')
    ap.add_argument('--envs', type=int, default=128)
    ap.add_argument('--vessels', type=int, default=16)
    ap.add_argument('--decisions', type=int, default=3000)
    ap.add_argument('--beta_hold', type=int, default=500)
    ap.add_argument('--beta_end', type=int, default=2000)
    ap.add_argument('--buffer', type=int, default=64, help='ring buffer length in decisions')
    ap.add_argument('--sgd_per_dec', type=int, default=4)
    ap.add_argument('--mb', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--w_thrust', type=float, default=0.1)
    ap.add_argument('--value_updates', type=int, default=15)
    ap.add_argument('--rollout', type=int, default=32)
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--device', default=None)
    ap.add_argument('--smoke', action='store_true', help='tiny run for a code check only (numbers meaningless)')
    a = ap.parse_args()
    if a.smoke:
        a.envs, a.decisions, a.beta_hold, a.beta_end = 4, 30, 10, 20
        a.buffer, a.sgd_per_dec, a.mb, a.value_updates, a.rollout, a.log_every = 8, 2, 64, 1, 8, 10

    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed)
    t_start = time.time()

    r = restore_policy(a.init, dev, tag='[dagger]')        # 팔은 스냅샷 값(OFF 또는 ARPA@56·메시지 없음 ON)
    raw = r.raw
    if int(raw.get('steps', -1)) != 0:
        raise SystemExit(f"[dagger] --init must be a step-0 checkpoint (steps={raw.get('steps')}); "
                         f"make it with vessel_gym_train.py --steps 0")
    if int(raw.get('seed', -1)) != a.seed:
        raise SystemExit(f"[dagger] --init seed {raw.get('seed')} != --seed {a.seed}")
    if r.arm != 'OFF':
        _sn = r.snap or {}
        _no_msg = (float(_sn.get('comm_latent', 1.0)) == 0.0 and not _sn.get('comm_codec_sha256')
                   and _sn.get('partner_range') is not None and float(_sn['partner_range']) <= float(vg.RADAR_RANGE) + 1e-6)
        if not _no_msg:
            raise SystemExit('[dagger] --init 은 OFF 이거나 ARPA@레이더범위·메시지 없음 ON 체크포인트여야 함(공통 뿌리)')
    policy = r.policy
    policy.train()
    E, N = a.envs, a.vessels
    env = make_env_from_snapshot(r.snap, device=dev, num_envs=E, seed=a.seed, n_vessels=N, tag='[dagger]')
    print(f"[dagger] {r.header()}", flush=True)
    teacher = Teacher(a.teacher, env)
    print(f"[dagger] teacher={a.teacher} {TEACHERS[a.teacher]} action_mode={vg.ACTION_MODE} dyn={getattr(cfg, 'DYN_PROFILE', '?')} obst={getattr(cfg, 'OBSTACLES_MODE', '?')} "
          f"envs={E} vessels={N} D={a.decisions} beta hold/end={a.beta_hold}/{a.beta_end} buffer={a.buffer} "
          f"sgd={a.sgd_per_dec}x{a.mb} lr={a.lr} w_thrust={a.w_thrust} value_updates={a.value_updates} dev={dev}", flush=True)

    obs = env.reset()
    radar, goal, self_s, sit = parse_obs(obs)
    fs = FrameStack(E, N, dev)
    fs.reset_all(radar)
    om = make_others_msg(env, 'OFF', E, N, dev)
    gen = torch.Generator(device=dev).manual_seed(a.seed + 101)   # driver choice + minibatch draws (env.gen untouched)
    # ★2026-10-02 ON 경로(ARPA@56·메시지 없음 trunk, 스펙 2026-10-02-fig1-latent-design.md §2): 상대 필드가 comm_gather 로
    #   들어오므로 흉내도 PPO 업데이트와 같은 평가 경로(evaluate_actions)로 −log π(선생님 행동)을 줄인다(attention 까지 학습).
    #   메시지는 latent 0 이라 파트너 obs(px·pg·ps)는 결과에 영향 없음 → 0 텐서로 넘김. logstd 는 학습하지 않음.
    on = (r.arm == 'ON')
    K = int(r.max_partners or cfg.MAX_COMM_PARTNERS)

    def gather(x_, goal_, self_, sit_):
        if not on:
            return om, None
        with torch.no_grad():
            om_, parts = comm_gather(policy, env, x_, goal_, self_, sit_, K)
        return om_, parts

    # ── 1. DAgger ──
    ctr_params = [p for p in policy.ctr_actor.parameters()]
    if on:
        _ids, train_params = set(), []
        for mod in (policy.ctr_actor, getattr(policy, 'attn', None), getattr(policy, 'msg_encoder', None)):
            if mod is None:
                continue
            for n_, p_ in mod.named_parameters():
                if 'logstd' in n_ or id(p_) in _ids:
                    continue
                _ids.add(id(p_)); train_params.append(p_)
    else:
        train_params = ctr_params
    opt = torch.optim.Adam(train_params, lr=a.lr)
    cap = a.buffer * E * N
    xdim = fs.get().shape[-1]
    bx = torch.zeros(cap, xdim, device=dev, dtype=torch.float16)   # radar stack in [-0.5, 0.5] → fp16 is exact enough
    bg = torch.zeros(cap, goal.shape[-1], device=dev)
    bs = torch.zeros(cap, self_s.shape[-1], device=dev)
    bsit = torch.zeros(cap, device=dev, dtype=torch.long)
    ba = torch.zeros(cap, 2, device=dev)
    wvec = torch.tensor([1.0, a.w_thrust], device=dev)
    om_mb = torch.zeros(a.mb, 1, om.shape[-1], device=dev)
    if on:
        _om0, _p0 = gather(fs.get(), goal, self_s, sit)
        _prel_dim = _p0[4].shape[-1]
        _Kc = _p0[4].shape[2]
        bprel = torch.zeros(cap, _Kc, _prel_dim, device=dev)
        bpm = torch.zeros(cap, _Kc, 1, device=dev)
        bpsit = torch.zeros(cap, _Kc, device=dev, dtype=torch.long)
        bgf = torch.zeros(cap, N, 6, device=dev) if cfg.CENTRAL_CRITIC else None
        zpx = torch.zeros(a.mb, _Kc, xdim, device=dev)
        zpg = torch.zeros(a.mb, _Kc, goal.shape[-1], device=dev)
        zps = torch.zeros(a.mb, _Kc, self_s.shape[-1], device=dev)
    ptr, filled = 0, 0
    drive_t = torch.ones(E, N, device=dev, dtype=torch.bool)       # beta = 1 at t = 0
    win = {'loss': 0.0, 'nl': 0, 'agree': 0.0, 'na': 0}
    oc_s = torch.zeros(5, device=dev)    # outcomes of student-driven ships (RUNNING/GOAL/vColl/oColl/TO)
    oc_t = torch.zeros(5, device=dev)    # outcomes of teacher-driven ships
    last = {}
    for t in range(a.decisions):
        beta = _beta(t, a.beta_hold, a.beta_end)
        x = fs.get()
        om_t, parts = gather(x, goal, self_s, sit)
        with torch.no_grad():
            act_s, _, mean_s, _ = policy.ctr_actor(x, goal, self_s, om_t, sit)
            a_star, _ = teacher.act(env)
            tgt = a_star.clamp(-TARGET_CLIP, TARGET_CLIP)
            win['agree'] += float(((torch.tanh(mean_s[..., 0]) - tgt[..., 0]).abs() < AGREE_TOL).float().mean())
            win['na'] += 1
            a_exec = torch.where(drive_t.unsqueeze(-1), a_star, act_s)
            n_new = E * N
            idx = (torch.arange(n_new, device=dev) + ptr) % cap
            bx[idx] = x.reshape(n_new, -1).to(torch.float16)
            bg[idx] = goal.reshape(n_new, -1)
            bs[idx] = self_s.reshape(n_new, -1)
            bsit[idx] = sit.reshape(n_new)
            ba[idx] = tgt.reshape(n_new, 2)
            if on:
                bprel[idx] = parts[4].reshape(n_new, _Kc, -1)
                bpm[idx] = parts[3].reshape(n_new, _Kc, 1)
                bpsit[idx] = parts[5].reshape(n_new, _Kc)
                if bgf is not None:
                    bgf[idx] = build_global_feat(env).reshape(n_new, N, 6)
            ptr = (ptr + n_new) % cap
            filled = min(cap, filled + n_new)
        obs, _, done, outcome = env.step(a_exec)
        oc1 = F.one_hot(outcome.reshape(-1), 5).float()
        oc_t += (oc1 * drive_t.reshape(-1, 1).float()).sum(0)
        oc_s += (oc1 * (~drive_t).reshape(-1, 1).float()).sum(0)
        teacher.after_step(env, a_exec, done)
        new_drv = torch.rand(E, N, generator=gen, device=dev) < beta
        drive_t = torch.where(done, new_drv, drive_t)
        radar, goal, self_s, sit = parse_obs(obs)
        fs.push(radar, done)

        for _ in range(a.sgd_per_dec):
            mi = torch.randint(0, filled, (a.mb,), generator=gen, device=dev)
            if on:
                _, lp, *_ = policy.evaluate_actions(
                    bx[mi].float(), bg[mi], bs[mi], zpx, zpg, zps, bpm[mi], bprel[mi], torch.atanh(ba[mi]),
                    situation=bsit[mi], partner_situations=bpsit[mi], global_feat=bgf[mi] if bgf is not None else None)
                loss = -lp.mean()
            else:
                _, _, mean_b, _ = policy.ctr_actor(bx[mi].float().unsqueeze(1), bg[mi].unsqueeze(1), bs[mi].unsqueeze(1),
                                                   om_mb, bsit[mi].unsqueeze(1))
                pred = torch.tanh(mean_b.squeeze(1))
                loss = (((pred - ba[mi]) ** 2) * wvec).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, cfg.MAX_GRAD_NORM)
            opt.step()
            win['loss'] += float(loss)
            win['nl'] += 1

        if (t + 1) % a.log_every == 0 or t + 1 == a.decisions:
            def rates(oc):
                n = float(oc[1:].sum())
                return n, [100.0 * float(oc[i]) / n if n > 0 else float('nan') for i in (1, 2, 3, 4)]
            ns, rs = rates(oc_s)
            nt, rt = rates(oc_t)
            last = dict(dec=t + 1, beta=beta, loss=win['loss'] / max(1, win['nl']), agree=100.0 * win['agree'] / max(1, win['na']),
                        student_eps=ns, student_goal=rs[0], student_vColl=rs[1], student_oColl=rs[2], student_TO=rs[3],
                        teacher_eps=nt, teacher_goal=rt[0], teacher_vColl=rt[1])
            print(f"[dagger] dec={t + 1:5d} beta={beta:.2f} loss={last['loss']:.4f} agree={last['agree']:5.1f}% | "
                  f"student eps={ns:.0f} goal={rs[0]:5.1f}% vColl={rs[1]:5.1f}% oColl={rs[2]:4.1f}% TO={rs[3]:4.1f}% | "
                  f"teacher eps={nt:.0f} goal={rt[0]:5.1f}% vColl={rt[1]:5.1f}% | {(time.time() - t_start) / 60:.1f}min",
                  flush=True)
            win = {'loss': 0.0, 'nl': 0, 'agree': 0.0, 'na': 0}
            oc_s.zero_()
            oc_t.zero_()
    del bx, bg, bs, bsit, ba
    if on:
        del bprel, bpm, bpsit, bgf, zpx, zpg, zps

    # ── 2. critic warm-up (actor frozen) ──
    ctr_ids = {id(p) for p in policy.ctr_actor.parameters()}
    for p in ctr_params:
        p.requires_grad_(False)
    crit_params = [p for p in policy.critic.parameters() if id(p) not in ctr_ids]
    vopt = torch.optim.Adam(crit_params, lr=a.lr)
    vnorm = ValueNorm(dev)
    vnorm.load(raw.get('value_norm'))
    T = a.rollout
    vlog = {}
    for u in range(a.value_updates):
        B = {k: [] for k in ('x', 'goal', 'self', 'sit', 'om', 'gf', 'val', 'rew', 'done', 'trunc')}
        ocv = torch.zeros(5, device=dev)
        for _ in range(T):
            x = fs.get()
            om_t, _ = gather(x, goal, self_s, sit)
            with torch.no_grad():
                action, _, _, _ = policy.ctr_actor(x, goal, self_s, om_t, sit)
                gf = build_global_feat(env) if cfg.CENTRAL_CRITIC else None
                value = vnorm.denormalize(policy.critic(x, goal, self_s, om_t, sit, global_feat=gf).squeeze(-1))
            obs, reward, done, outcome = env.step(action)
            ocv += torch.bincount(outcome.reshape(-1), minlength=5)[:5].float()
            B['x'].append(x); B['goal'].append(goal); B['self'].append(self_s); B['sit'].append(sit); B['om'].append(om_t)
            if gf is not None:
                B['gf'].append(gf)
            B['val'].append(value); B['rew'].append(reward); B['done'].append(done.float())
            B['trunc'].append((outcome == vg.OUT_TIMEOUT).float() if cfg.TIMEOUT_BOOTSTRAP else torch.zeros_like(done.float()))
            radar, goal, self_s, sit = parse_obs(obs)
            fs.push(radar, done)
        om_t, _ = gather(fs.get(), goal, self_s, sit)
        with torch.no_grad():
            gf = build_global_feat(env) if cfg.CENTRAL_CRITIC else None
            last_v = vnorm.denormalize(policy.critic(fs.get(), goal, self_s, om_t, sit, global_feat=gf).squeeze(-1))
        S = {k: torch.stack(v) for k, v in B.items() if v}
        del B
        returns, _ = batched_gae(S['rew'], S['val'], S['done'], S['trunc'], last_v, cfg.DISCOUNT_FACTOR, cfg.GAE_LAMBDA)
        vnorm.update(returns)
        ret_n = vnorm.normalize(returns)

        def flat(t_):
            return t_.reshape(-1, *t_.shape[3:]) if t_.dim() > 3 else t_.reshape(-1)
        fx, fg, fsf, fsit, fret = flat(S['x']), flat(S['goal']), flat(S['self']), flat(S['sit']), flat(ret_n)
        fgf = flat(S['gf']) if 'gf' in S else None
        fom = flat(S['om'])
        M = fx.shape[0]
        vl, nv = 0.0, 0
        for _ in range(cfg.N_EPOCH):
            perm = torch.randperm(M, generator=gen, device=dev)
            for i in range(0, M - cfg.MINIBATCH_SIZE + 1, cfg.MINIBATCH_SIZE):
                mi = perm[i:i + cfg.MINIBATCH_SIZE]
                v = policy.critic(fx[mi].unsqueeze(1), fg[mi].unsqueeze(1), fsf[mi].unsqueeze(1), fom[mi].unsqueeze(1),
                                  fsit[mi].unsqueeze(1),
                                  global_feat=fgf[mi] if fgf is not None else None).squeeze(1).squeeze(-1)
                vloss = ((v - fret[mi]) ** 2).mean()
                vopt.zero_grad()
                (cfg.CRITIC_LOSS_WEIGHT * vloss).backward()
                torch.nn.utils.clip_grad_norm_(crit_params, cfg.MAX_GRAD_NORM)
                vopt.step()
                vl += float(vloss)
                nv += 1
        del S, fx, fg, fsf, fsit, fret, fgf, fom
        n_end = float(ocv[1:].sum())
        vlog = dict(update=u + 1, value_loss=vl / max(1, nv), ret_mean=vnorm.state()['mean'], ret_std=vnorm.state()['std'],
                    student_eps=n_end, student_goal=100.0 * float(ocv[1]) / n_end if n_end else float('nan'),
                    student_vColl=100.0 * float(ocv[2]) / n_end if n_end else float('nan'))
        print(f"[value] update {u + 1}/{a.value_updates} value_loss={vlog['value_loss']:.4f} "
              f"ret mean={vlog['ret_mean']:.1f} std={vlog['ret_std']:.1f} | student(sampled) eps={n_end:.0f} "
              f"goal={vlog['student_goal']:5.1f}% vColl={vlog['student_vColl']:5.1f}% | {(time.time() - t_start) / 60:.1f}min",
              flush=True)
    for p in ctr_params:
        p.requires_grad_(True)

    # ── 3. save (trainer format, step 0) ──
    info = dict(spec='docs/superpowers/specs/2026-10-02-imitation-init-design.md', code=_git_head(),
                teacher=dict(TEACHERS[a.teacher], name=a.teacher, action_mode=vg.ACTION_MODE,
                             source='imitation/vo_teacher.py (vo_action / vo_cost_intent; true pos/vel of ships <= R)'),
                init=os.path.basename(a.init), init_sha256=_sha256(a.init), seed=a.seed, envs=E, vessels=N,
                decisions=a.decisions, beta_hold=a.beta_hold, beta_end=a.beta_end, buffer=a.buffer,
                sgd_per_dec=a.sgd_per_dec, mb=a.mb, lr=a.lr, w_thrust=a.w_thrust, target_clip=TARGET_CLIP,
                value_updates=a.value_updates, rollout=a.rollout, smoke=bool(a.smoke),
                dagger_last=last, value_last=vlog, minutes=(time.time() - t_start) / 60.0)
    ck = dict(raw)
    ck['model_state_dict'] = policy.state_dict()
    ck['value_norm'] = vnorm.state()
    ck.pop('optimizer_state_dict', None)
    ck.update(arm=r.arm, comm_active=bool(raw.get('comm_active', False)), seed=a.seed, steps=0)
    snap = dict(ck.get('cfg_snapshot') or {})
    snap['init_imitation'] = info
    ck['cfg_snapshot'] = snap
    torch.save(ck, a.save)
    side = os.path.splitext(a.save)[0] + '.imitation.json'
    with open(side, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=1, default=str)
    print(f"[dagger] saved -> {a.save} (+ {os.path.basename(side)}) {info['minutes']:.1f}min", flush=True)


if __name__ == '__main__':
    main()
