"""diag_ckpt.py — 체크포인트 진단 단일 진입점 (2026-09-10).

왜 있나
  체크포인트를 열어 통신 지표를 재는 스크립트가 20개 넘게 있었고 전부 각자 설정을 박아 학습과 다른 조건으로
  쟀다(측정 2회 무효). 이제 진단은 여기서만 한다:

    restore_policy  →  make_env_from_snapshot  →  burn  →  게이트  →  comm_telemetry  →  JSON + CSV

  - 설정은 체크포인트 스냅샷에서만 온다. env 를 손으로 세팅하지 않는다 (ckpt_io 규약).
  - 지표 정의는 vessel_gym_train.comm_telemetry 하나다. 여기서 재구현하지 않는다.
  - **게이트를 통과 못 하면 숫자를 내지 않는다.** 조우율이 낮은 창(배들이 아직 안 만난 구간)에서 잰 숫자,
    학습과 다른 설정으로 잰 숫자가 두 번 보고됐다 철회된 게 이 파일이 생긴 이유다.

게이트
  1. 조우율(sit≠0 비율) ≥ --min_sit_rate (기본 5%). 평가 창은 약 10%.
  2. 설정 == 스냅샷 — restore_policy 가 comm_range/arm 불일치에서 중단하므로 여기 도달했으면 통과.
  3. --expect_vcoll 을 주면 창의 vColl% 이 그 값의 ±--expect_tol(상대) 안이어야 함 (평가 수치와 대조).

출력
  --out X.json : 헤더(적용 설정·env·창·조우율·게이트) + outcome + 텔레메트리 평균 + 조우/비조우 분리 메시지 통계
  같은 이름 .csv : 한 줄 (열 = 헤더 몇 개 + COMM_TELE_COLS + split 통계) — 여러 ckpt 를 표로 붙이기 용

쓰는 법
  python diag_ckpt.py --ckpt ff_ON_s43.pt --burn 1000 --collect 900 --out ff_ON_s43.json
  python diag_ckpt.py --ckpt ff_ON_s43.pt --expect_vcoll 1.1        # 평가 vColl 과 대조
  OFF 팔도 돌아간다(outcome·조우율만, 통신 지표는 정의되지 않음).
"""
import argparse
import json
import os
import sys
import time

import torch

import config as cfg
import vessel_gym as vg
from ckpt_io import restore_policy, make_env_from_snapshot, describe
from vessel_gym_train import (comm_gather, parse_obs, FrameStack, make_others_msg,
                              comm_telemetry, COMM_TELE_COLS, msg_stats)

OUT_NAMES = {1: 'goal', 2: 'vColl', 3: 'oColl', 4: 'TO'}


def _mean_rows(rows):
    """텔레메트리 행(dict) 리스트 → 열별 평균 (nan 무시)."""
    import math
    out = {}
    for k in COMM_TELE_COLS:
        vals = [float(r[k]) for r in rows if k in r and not math.isnan(float(r[k]))]
        out[k] = sum(vals) / len(vals) if vals else float('nan')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default=None, help='예: cuda:1 (미지정이면 cuda / cpu)')
    ap.add_argument('--envs', type=int, default=32)
    ap.add_argument('--vessels', type=int, default=None, help='None=스냅샷')
    ap.add_argument('--seed', type=int, default=999, help='진단 env 시드(학습·평가와 분리)')
    ap.add_argument('--burn', type=int, default=1000, help='집계 전 굴릴 결정 수. 조우가 정상상태(~10%)가 되는 지점')
    ap.add_argument('--collect', type=int, default=900)
    ap.add_argument('--ring', type=float, default=None, help='None=스냅샷. 주면 override(로그)')
    ap.add_argument('--crossing', type=int, default=None, help='None=스냅샷')
    ap.add_argument('--max_partners', type=int, default=None, help='None=스냅샷')
    ap.add_argument('--arm', default=None, choices=[None, 'OFF', 'ORACLE', 'ON', 'RANDOM'], help='None=스냅샷')
    ap.add_argument('--allow_arm_mismatch', action='store_true')
    ap.add_argument('--allow_comm_range_mismatch', action='store_true')
    ap.add_argument('--min_sit_rate', type=float, default=0.05)
    ap.add_argument('--expect_vcoll', type=float, default=None, help='평가에서 얻은 vColl%% (예: 1.1)')
    ap.add_argument('--expect_tol', type=float, default=0.5, help='상대 허용오차 (0.5 = ±50%%)')
    ap.add_argument('--tele_every', type=int, default=50, help='텔레메트리 측정 간격(결정)')
    ap.add_argument('--out', default=None, help='JSON 경로. 없으면 stdout 만')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass

    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    t0 = time.time()

    # ── 1. 복원 (불일치는 여기서 중단) ──
    r = restore_policy(args.ckpt, dev, arm=args.arm, max_partners=args.max_partners,
                       allow_arm_mismatch=args.allow_arm_mismatch,
                       allow_comm_range_mismatch=args.allow_comm_range_mismatch, tag='[diag]')
    env = make_env_from_snapshot(r.snap, device=dev, num_envs=args.envs, seed=args.seed, n_vessels=args.vessels,
                                 ring=args.ring, crossing=args.crossing, tag='[diag]')
    E, N = env.E, env.N
    policy = r.policy
    K = r.max_partners
    gen = torch.Generator().manual_seed(args.seed + 100003)      # 텔레메트리 전용 CPU RNG (학습기와 같은 규약)

    fs = FrameStack(E, N, dev)
    obs = env.reset()
    radar, goal, self_s, sit = parse_obs(obs)
    fs.reset_all(radar)

    def act(x, goal, self_s, sit):
        with torch.no_grad():
            if r.arm == 'ON':
                om, _ = comm_gather(policy, env, x, goal, self_s, sit, K)
            else:
                om = make_others_msg(env, r.arm, E, N, dev)
            a, _, _, _ = policy.ctr_actor(x, goal, self_s, om, sit)
        return a

    # ── 2. burn ──
    for _ in range(args.burn):
        obs, _, done, _ = env.step(act(fs.get(), goal, self_s, sit))
        radar, goal, self_s, sit = parse_obs(obs)
        fs.push(radar, done)

    # ── 3. collect ──
    counts = torch.zeros(5, dtype=torch.long)
    sit_hits = sit_total = 0
    tele_rows = []
    split = {'enc': [], 'non': []}          # msg 텐서 조각 (조우 중 / 비조우)
    for t in range(args.collect):
        x = fs.get()
        s2 = sit.reshape(E, N)
        enc = (s2 != 0)
        sit_hits += int(enc.sum()); sit_total += E * N
        if r.arm == 'ON' and t % args.tele_every == 0:
            with torch.no_grad():
                tele_rows.append(comm_telemetry(policy, env, x, goal, self_s, sit, K, gen, vg.RADAR_RANGE))
                msg = policy.msg_actor(x, goal, self_s, sit).reshape(E * N, -1)
                em = enc.reshape(-1)
                split['enc'].append(msg[em].cpu()); split['non'].append(msg[~em].cpu())
        obs, _, done, outcome = env.step(act(x, goal, self_s, sit))
        for oc in range(1, 5):
            counts[oc] += int((outcome == oc).sum())
        radar, goal, self_s, sit = parse_obs(obs)
        fs.push(radar, done)

    # ── 4. 게이트 ──
    sit_rate = sit_hits / max(1, sit_total)
    total = int(counts[1:].sum())
    rates = {OUT_NAMES[oc]: (float(counts[oc]) / total * 100.0 if total else float('nan')) for oc in range(1, 5)}
    gates = {'sit_rate': {'value': sit_rate, 'min': args.min_sit_rate, 'pass': sit_rate >= args.min_sit_rate},
             'config_match': {'pass': True, 'note': 'restore_policy 가 comm_range/arm 불일치에서 중단함'}}
    if args.expect_vcoll is not None:
        v = rates['vColl']
        ok = total > 0 and abs(v - args.expect_vcoll) <= args.expect_tol * max(args.expect_vcoll, 1e-9)
        gates['vcoll'] = {'value': v, 'expect': args.expect_vcoll, 'tol': args.expect_tol, 'pass': bool(ok)}
    passed = all(g['pass'] for g in gates.values())

    header = {
        'ckpt': os.path.basename(r.path), 'applied': r.header(), 'snapshot': describe(r.snap),
        'env_sources': getattr(env, 'snapshot_sources', {}),
        'envs': E, 'vessels': N, 'seed': args.seed, 'burn': args.burn, 'collect': args.collect,
        'sit_rate': sit_rate, 'n_episodes': total, 'outcomes_pct': rates, 'gates': gates, 'gates_passed': passed,
        'notes': r.notes, 'elapsed_s': round(time.time() - t0, 1),
    }
    print(f"[diag] 창: burn={args.burn} collect={args.collect} 조우율={sit_rate*100:.2f}% 에피소드={total} "
          f"goal={rates['goal']:.1f}% vColl={rates['vColl']:.2f}% oColl={rates['oColl']:.2f}% TO={rates['TO']:.2f}%")
    for k, g in gates.items():
        print(f"[diag] 게이트 {k:12s} {'PASS' if g['pass'] else '★FAIL'}  {g}")
    if not passed:
        print("[diag] ★게이트 실패 — 지표를 내지 않음. 창(burn)·설정을 고쳐서 다시 돌릴 것.")
        if args.out:
            json.dump(header, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        sys.exit(3)

    # ── 5. 지표 ──
    rec = dict(header)
    if r.arm == 'ON':
        rec['telemetry'] = _mean_rows(tele_rows)
        rec['telemetry_n'] = len(tele_rows)
        sp = {}
        for g in ('enc', 'non'):
            if split[g]:
                m = torch.cat(split[g], 0)
                sp[g] = dict(n=int(m.shape[0]), **msg_stats(m))
        rec['msg_split'] = sp
        print(f"[diag] 텔레메트리 {len(tele_rows)}회 평균:")
        for k in COMM_TELE_COLS:
            print(f"         {k:14s} {rec['telemetry'][k]:.4f}")
        for g, d in sp.items():
            print(f"[diag] msg[{g}] n={d['n']} " + ' '.join(f"{k}={v:.4f}" for k, v in d.items() if k != 'n'))
    else:
        rec['telemetry'] = None
        print(f"[diag] arm={r.arm}: 통신 지표는 정의되지 않음 (outcome·조우율만)")

    if args.out:
        json.dump(rec, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        csvp = os.path.splitext(args.out)[0] + '.csv'
        cols = ['ckpt', 'arm', 'sit_rate', 'n_episodes', 'goal', 'vColl', 'oColl', 'TO']
        vals = [rec['ckpt'], r.arm, f"{sit_rate:.5f}", total] + [f"{rates[k]:.4f}" for k in ('goal', 'vColl', 'oColl', 'TO')]
        if rec.get('telemetry'):
            cols += list(COMM_TELE_COLS); vals += [f"{rec['telemetry'][k]:.6g}" for k in COMM_TELE_COLS]
            for g in ('enc', 'non'):
                for k, v in rec['msg_split'].get(g, {}).items():
                    cols.append(f'{g}_{k}'); vals.append(f"{v:.6g}" if isinstance(v, float) else str(v))
        with open(csvp, 'w', encoding='utf-8') as f:
            f.write(','.join(cols) + '\n' + ','.join(map(str, vals)) + '\n')
        print(f"[diag] → {args.out}, {csvp}")


if __name__ == '__main__':
    main()
