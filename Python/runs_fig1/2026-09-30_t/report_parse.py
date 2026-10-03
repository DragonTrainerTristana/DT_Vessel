"""report_parse.py — eval_*.txt 파서 + 사전등록 판정표 (2026-09-30, 스펙 2026-09-30-reward-v3-decode-sweep-design.md §7).

  python report_parse.py --dir <out dir> --prefix t_ --off off --arms a6,a8,a2,a4 --seeds 43,44,45 [--noise offb] [--extra_off s_off=<dir>,x_off=<dir>]
  python report_parse.py --dir C:/work/DT_Vessel/Python/_repro_out_s_stageA --prefix s_ --off off --arms a6,c6 --seeds 43,44,45 --off_dir C:/work/DT_Vessel/Python/_repro_out_s

지표(방향): goal↑ vColl↓ oColl↓ TO↓ taskret↑ fuel↓ head↓ dcpa(pair-detail 중앙)↑ C(레이더 안 실제타각)↑ rks_end↑ rks_v2↑ epReward↑
통과 = 같은 시드 짝 3/3 방향 일치 그리고 |평균차| > N.  N = max(|offb−off| 지표별 평균차, off 시드 범위, 고정 N: vColl 12.1 pp · dcpa 1.5 m).
stdlib 만 씀(mltest python 으로 실행 가능). 숫자는 파일에 있는 값만, 없으면 None.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

DIR_UP = {'goal': 1, 'vColl': -1, 'oColl': -1, 'TO': -1, 'taskret': 1, 'fuel': -1, 'head': -1, 'dcpa': 1, 'C': 1,
          'rks_end': 1, 'rks_v2': 1, 'epReward': 1, 'C_standon': 1, 'C_giveway': 1, 'len': -1, 'minPass': 1,
          'fuel_per_m': -1, 'fleet_fuel': -1, 'head_per_len': -1, 'rks_v2x': 1, 'both_v2': 1, 'safe_v2': 1}
FIXED_N = {'vColl': 12.1, 'dcpa': 1.5}
PRIMARY = ('goal', 'vColl', 'dcpa', 'C', 'rks_v2', 'fuel', 'head')      # 저자 6 지표 + COLREGs 두 판정


def _f(m, i=1):
    return float(m.group(i)) if m else None


def parse_eval(path):
    s = open(path, encoding='utf-8', errors='replace').read()
    d = {'file': os.path.basename(path)}
    m = re.search(r"goal=\s*([\d.]+)%\s+vColl=\s*([\d.]+)%\s+oColl=\s*([\d.]+)%\s+TO=\s*([\d.]+)%\s+\|\s*(\d+) eps", s)
    if m:
        d.update(goal=float(m.group(1)), vColl=float(m.group(2)), oColl=float(m.group(3)), TO=float(m.group(4)), eps=int(m.group(5)))
        d['taskret'] = 1.5 * d['goal'] - 3.0 * (d['vColl'] + d['oColl']) - 0.5 * d['TO']
    m = re.search(r"\[goal-ep\] fuel=\s*([\d.]+)\s+headTravel=\s*([\d.]+)deg\s+minSep=\s*([\d.]+)m\s+len=\s*([\d.]+)", s)
    if m:
        d.update(fuel=float(m.group(1)), head=float(m.group(2)), minSep=float(m.group(3)), len=float(m.group(4)))
    d['epReward'] = _f(re.search(r"epReward=\s*(-?[\d.]+)", s))
    m = re.search(r"\[encounter-COLREGs/실제타각\] C=\s*([\d.]+)%.*?HeadOn\(R14\)=\s*([\d.]+)%.*?StandOn\(R17\)=\s*([\d.]+)%.*?GiveWay\(R15/16\)=\s*([\d.]+)%.*?Overtake\(R13\)=\s*([\d.]+)%", s)
    if m:
        d.update(C=float(m.group(1)), C_headon=float(m.group(2)), C_standon=float(m.group(3)), C_giveway=float(m.group(4)), C_overtake=float(m.group(5)))
    d['minPass'] = _f(re.search(r"\[encounter-minPass\]\s+전체=\s*([\d.]+)m", s))
    d['dcpa'] = _f(re.search(r"\[pair-detail\] 통과최소거리 중앙\s*([\d.]+)m", s))
    m = re.search(r"\[role-promise\] judged n=(\d+)\s+roleKeptSafe=\s*([\d.]+)%\s+both_comply=\s*([\d.]+)%\s+safe=\s*([\d.]+)%", s)
    if m:
        d.update(rp_n=int(m.group(1)), rks_end=float(m.group(2)), both_end=float(m.group(3)), safe_end=float(m.group(4)))
    m = re.search(r"\[role-promise/v2\] judged n=(\d+)\s+roleKeptSafe=\s*([\d.]+)%\s+both_comply=\s*([\d.]+)%\s+safe=\s*([\d.]+)%.*?resolved=(\S+)", s)
    if m:
        d.update(rp2_n=int(m.group(1)), rks_v2=float(m.group(2)), both_v2=float(m.group(3)), safe_v2=float(m.group(4)))
        try:
            rv = int(m.group(5)); d['resolved_v2'] = rv
            # 해소 제외 판정: (성공 − 해소)/(판정 − 해소)  (해소는 항상 성공)
            succ = d['rks_v2'] / 100.0 * d['rp2_n']
            d['rks_v2x'] = 100.0 * (succ - rv) / (d['rp2_n'] - rv) if d['rp2_n'] > rv else None
        except ValueError:
            d['resolved_v2'] = None
    m = re.search(r"\[fuel-diag\] fuel_per_progress_m=\s*([\d.]+)\s+headTravel_per_len=\s*([\d.]+)deg\s+fleet_fuel_per_arrival=\s*([\d.]+)", s)
    if m:
        d.update(fuel_per_m=float(m.group(1)), head_per_len=float(m.group(2)), fleet_fuel=float(m.group(3)))
    return d


def load_runs(dir_, prefix, arm, seeds):
    out = {}
    for sd in seeds:
        p = os.path.join(dir_, f"eval_{prefix}{arm}_s{sd}.txt")
        if os.path.exists(p):
            out[sd] = parse_eval(p)
    return out


def load_runs_any(dir_, prefix, arm):
    """★2026-10-01 풀 버그 수정: 풀 팔은 자기 시드 전부(eval_{prefix}{arm}_s*.txt)를 읽음.
    전에는 t_ 시드 목록(43,46,47)을 그대로 써서 s_off·x_off(시드 43–45)는 s43 하나만 들어갔음."""
    out = {}
    for p in sorted(glob.glob(os.path.join(dir_, f"eval_{prefix}{arm}_s*.txt"))):
        m = re.search(r'_s(\d+)\.txt$', p)
        if m:
            out[int(m.group(1))] = parse_eval(p)
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def judge(off, comm, noise, seeds, metrics):
    """off/comm/noise: {seed: dict}. Returns per-metric judgement dict."""
    res = {}
    for k in metrics:
        sgn = DIR_UP[k]
        wins, pairs = 0, 0
        diffs = []
        for sd in seeds:
            a, b = comm.get(sd, {}).get(k), off.get(sd, {}).get(k)
            if a is None or b is None:
                continue
            pairs += 1
            diffs.append(a - b)
            if sgn * (a - b) > 0:
                wins += 1
        md = mean(diffs)
        offv = [off[sd].get(k) for sd in seeds if sd in off and off[sd].get(k) is not None]
        rng = (max(offv) - min(offv)) if len(offv) >= 2 else None
        nz = None
        if noise:
            nd = [noise[sd].get(k) - off[sd].get(k) for sd in seeds if sd in noise and sd in off
                  and noise[sd].get(k) is not None and off[sd].get(k) is not None]
            nz = abs(mean(nd)) if nd else None
        N = max([x for x in (rng, nz, FIXED_N.get(k)) if x is not None] or [0.0])
        ok = (pairs >= 3 and wins == pairs and md is not None and sgn * md > N)
        res[k] = dict(wins=wins, pairs=pairs, mean_diff=md, N=N, off_range=rng, noise=nz, pass_=ok,
                      comm_mean=mean([comm[sd].get(k) for sd in seeds if sd in comm]),
                      off_mean=mean([off[sd].get(k) for sd in seeds if sd in off]))
    return res


def fmt(v, nd=1):
    if v is None:
        return '·'
    return f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--off_dir', default=None, help='OFF eval 파일이 다른 폴더에 있을 때(Stage A: s_off 는 s_ out)')
    ap.add_argument('--prefix', default='t_')
    ap.add_argument('--off', default='off')
    ap.add_argument('--arms', default='a6,a8,a2,a4')
    ap.add_argument('--noise', default=None, help='재분기 OFF 팔 이름(예 offb)')
    ap.add_argument('--seeds', default='43,44,45')
    ap.add_argument('--extra_off', default='', help='추가 OFF 풀: 이름=폴더,이름=폴더 (goal·vColl·dcpa 의 최선 OFF 풀; goal≥50 만)')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(',')]
    arms = a.arms.split(',')
    off = load_runs(a.off_dir or a.dir, a.prefix, a.off, seeds)
    noise = load_runs(a.dir, a.prefix, a.noise, seeds) if a.noise else None
    comm = {arm: load_runs(a.dir, a.prefix, arm, seeds) for arm in arms}
    cols = ['goal', 'vColl', 'oColl', 'TO', 'taskret', 'fuel', 'head', 'len', 'dcpa', 'minPass', 'C', 'C_standon', 'C_giveway',
            'rks_end', 'rks_v2', 'rks_v2x', 'epReward', 'fuel_per_m', 'fleet_fuel']
    print('## 시드별 값')
    print('| run | ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * (len(cols) + 1))
    allruns = [(f"{a.prefix}{a.off}", off)] + ([(f"{a.prefix}{a.noise}", noise)] if noise else []) + [(f"{a.prefix}{arm}", comm[arm]) for arm in arms]
    for nm, runs in allruns:
        for sd in seeds:
            if sd in runs:
                r = runs[sd]
                print(f"| {nm}_s{sd} | " + ' | '.join(fmt(r.get(c), 0 if c in ('fuel', 'head', 'len', 'fleet_fuel') else (2 if c == 'fuel_per_m' else 1)) for c in cols) + ' |')
        ms = {c: mean([runs[sd].get(c) for sd in seeds if sd in runs]) for c in cols}
        print(f"| **{nm} 평균** | " + ' | '.join(fmt(ms[c], 0 if c in ('fuel', 'head', 'len', 'fleet_fuel') else (2 if c == 'fuel_per_m' else 1)) for c in cols) + ' |')
    metrics = ['goal', 'vColl', 'oColl', 'TO', 'taskret', 'fuel', 'head', 'dcpa', 'C', 'C_standon', 'rks_end', 'rks_v2', 'rks_v2x', 'epReward', 'fuel_per_m', 'fleet_fuel']
    out = {}
    for arm in arms:
        res = judge(off, comm[arm], noise, seeds, metrics)
        out[arm] = res
        print(f"\n## {a.prefix}{arm} vs {a.prefix}{a.off} (같은 시드 짝, 통과 = 3/3 & |평균차| > N)")
        print('| 지표 | 방향 | comm 평균 | off 평균 | 평균차 | 승/짝 | N (off 범위 / 재분기 / 고정) | 통과 |')
        print('|---|---|---|---|---|---|---|---|')
        for k in metrics:
            r = res[k]
            arrow = '↑' if DIR_UP[k] > 0 else '↓'
            print(f"| {k} | {arrow} | {fmt(r['comm_mean'])} | {fmt(r['off_mean'])} | {fmt(r['mean_diff'], 2)} | {r['wins']}/{r['pairs']} | "
                  f"{fmt(r['N'], 2)} ({fmt(r['off_range'], 2)} / {fmt(r['noise'], 2)} / {fmt(FIXED_N.get(k), 1)}) | {'**통과**' if r['pass_'] else '불통과'}{' ★주' if k in PRIMARY else ''} |")
        prim = [k for k in PRIMARY if res.get(k, {}).get('pass_')]
        print(f"주 지표 통과: {len(prim)}/{len(PRIMARY)} → {prim}")
    if a.extra_off:
        print("\n## 최선 OFF 풀(goal ≥ 50 인 건강한 OFF 만; goal·vColl·dcpa): comm 3시드 전부가 풀 최대를 넘어야 함")
        pool = {f"{a.prefix}{a.off}": off}
        for item in a.extra_off.split(','):
            nm, d_ = item.split('=')
            pre, arm_ = nm.rsplit('_', 1)
            pool[nm] = load_runs_any(d_, pre + '_', arm_)
        for nm, runs in pool.items():
            print(f"- 풀 구성: {nm} 시드 {sorted(runs)} → goal {fmt(mean([r.get('goal') for r in runs.values()]))} "
                  f"vColl {fmt(mean([r.get('vColl') for r in runs.values()]))} dcpa {fmt(mean([r.get('dcpa') for r in runs.values()]))}")
        pool_ok = {arm: {} for arm in arms}
        for k in ('goal', 'vColl', 'dcpa'):
            sgn = DIR_UP[k]
            best, who = None, None
            for nm, runs in pool.items():
                mg = mean([r.get('goal') for r in runs.values()])
                if mg is None or mg < 50.0:
                    continue
                mv = mean([r.get(k) for r in runs.values()])
                if mv is not None and (best is None or sgn * (mv - best) > 0):
                    best, who = mv, nm
            for arm in arms:
                vals = [comm[arm][sd].get(k) for sd in seeds if sd in comm[arm]]
                ok = best is not None and len(vals) == 3 and all(v is not None and sgn * (v - best) > 0 for v in vals)
                pool_ok[arm][k] = ok
                print(f"- {k}: 풀 최선 = {who} {fmt(best, 2)} → {a.prefix}{arm} 3시드 {[fmt(v) for v in vals]} {'통과' if ok else '불통과'}")
        print("\n## 사전등록 최종 판정 (스펙 §7: 3/3 & |평균차| > N, 그리고 goal·vColl·dcpa 는 최선 OFF 풀도 넘어야)")
        for arm in arms:
            fin = [k for k in PRIMARY if out[arm].get(k, {}).get('pass_') and pool_ok[arm].get(k, True)]
            print(f"- {a.prefix}{arm}: 주 지표 최종 통과 {len(fin)}/{len(PRIMARY)} → {fin}")
            out[arm]['_final_pass'] = fin
            out[arm]['_pool_ok'] = pool_ok[arm]
    if a.json:
        json.dump({'off': off, 'noise': noise, 'comm': comm, 'judge': out}, open(a.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
