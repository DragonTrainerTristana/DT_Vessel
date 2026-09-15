"""장거리 항해 비용 합성 — measure_regimes.py 가 잰 국면별 계수로 임의 거리·밀도를 계산.

모델
----
항해를 두 국면으로 나눔.
  T (순항) : 결정당 연료 fT, 결정당 목표접근 sT
  E (조우) : 결정당 연료 fE, 결정당 목표접근 sE, 1회 평균 지속 durE
조우는 진행거리에 비례해 발생: 진행 1 m 당 λ 회 (밀도 ρ0 에서 측정). 밀도 r 배면 λ·r.

거리 D 항해에 대해
  E 국면이 감당하는 진행거리 : pE = min(D, λ·r·D · durE · sE)
  T 국면이 감당하는 진행거리 : pT = D − pE
  결정수  = pE/sE + pT/sT
  연료    = (pE/sE)·fE + (pT/sT)·fT
  항해시간 = 결정수 × 0.4 s      (거리·속도가 같은 축척이라 심=실제)

왜 이게 합법인가
---------------
vessel_gym 은 무기억 환경임(누적 상태 없음). 그래서 장거리 항해는 짧은 구간의 독립적
이어붙이기와 동치이고, 위 합성은 근사가 아니라 그 성질의 직접 결과임.
단, **밀도 r 은 설계 입력이지 외삽 대상이 아님** — 면적을 키우면서 척수를 안 늘리면 r<1 임.

자체 검증
--------
측정 조건(D = 338.7 m, r = 1)을 넣으면 실측 도착 결정수(883.3, metrics_v2.txt)가 재현돼야 함.
안 맞으면 모델이 틀린 것이므로 아래 [검증] 줄을 반드시 확인할 것.
"""
import argparse
import json
import os

DECS = 0.4          # s per decision
VS = 0.2            # VESSEL_SCALE — 심 1 m = 실제 5 m, 시간은 1:1
D0 = 338.7          # 측정 조건 여정 (m, 심)
L0 = 883.3          # 그때 실측 도착 결정수 (metrics_v2.txt)

ROUTES = [('Busan-Taiwan', 1590)]


def compose(c, D, r=1.0):
    """c=regime json, D=심 항해거리(m), r=밀도배율 → dict"""
    fT, sT = c['T']['fuel_per_dec'], c['T']['sog_per_dec']
    fE, sE = c['E']['fuel_per_dec'], c['E']['sog_per_dec']
    lam, dur = c['encounter']['enter_per_m'], c['encounter']['mean_dur_dec']
    # 진행 1 m 당 조우 진입 λ·r, 1회당 진행거리 dur·sE
    pE = min(D, lam * r * D * dur * max(sE, 1e-9))
    pT = max(D - pE, 0.0)
    decE = pE / max(sE, 1e-9)
    decT = pT / max(sT, 1e-9)
    dec = decE + decT
    return dict(D=D, r=r, dec=dec, decE=decE, decT=decT,
                fuel=decE * fE + decT * fT,
                days=dec * DECS / 86400, frac_E=decE / max(dec, 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--on', default='')
    ap.add_argument('--off', default='')
    ap.add_argument('--dens', type=float, default=1.0,
                    help='밀도 배율 r. 면적만 키우고 척수 그대로면 r<1')
    args = ap.parse_args()

    arms = []
    for lab, p in (('Comm ON', args.on), ('Comm OFF', args.off)):
        if p and os.path.exists(p):
            arms.append((lab, json.load(open(p, encoding='utf-8'))))
    if not arms:
        raise SystemExit('regime json 이 없음 — measure_regimes.py 먼저 돌릴 것')

    print('=== 측정된 국면별 계수 ===')
    hdr = f"{'arm':9s} {'T fuel/dec':>11s} {'T sog':>8s} {'E fuel/dec':>11s} {'E sog':>8s} " \
          f"{'λ(/m)':>9s} {'조우지속':>8s} {'E비율':>7s} {'표본(dec)':>11s}"
    print(hdr)
    for lab, c in arms:
        print(f"{lab:9s} {c['T']['fuel_per_dec']:11.4f} {c['T']['sog_per_dec']:8.4f} "
              f"{c['E']['fuel_per_dec']:11.4f} {c['E']['sog_per_dec']:8.4f} "
              f"{c['encounter']['enter_per_m']:9.5f} {c['encounter']['mean_dur_dec']:8.1f} "
              f"{c['frac_E']:6.1%} {c['agent_decisions']:11,d}")

    print('\n=== [검증] 측정 조건 재현 — D=338.7m, r=1 → 실측 883.3 결정이어야 함 ===')
    for lab, c in arms:
        v = compose(c, D0, 1.0)
        err = 100 * (v['dec'] - L0) / L0
        flag = 'OK' if abs(err) < 25 else '★불일치 — 모델 재검토'
        print(f"  {lab:9s} 합성 {v['dec']:7.1f} 결정   실측 {L0:.1f}   오차 {err:+6.1f}%   {flag}")

    print(f"\n=== 대만 항로 합성 (밀도 배율 r={args.dens}) ===")
    for nm, km in ROUTES:
        D = km * 1000 * VS
        print(f"  {nm}  실제 {km:,} km = 심 {D/1000:.0f} km")
        res = {}
        for lab, c in arms:
            v = compose(c, D, args.dens)
            res[lab] = v
            print(f"    {lab:9s} 결정 {v['dec']:12,.0f}  step {v['dec']*10:13,.0f}  "
                  f"연료 {v['fuel']:12,.0f}  항해 {v['days']:6.2f}일  (조우구간 {v['frac_E']:.1%})")
        if len(res) == 2:
            a, b = res['Comm OFF'], res['Comm ON']
            print(f"    {'차이':9s} 연료 {100*(b['fuel']-a['fuel'])/a['fuel']:+6.2f}%   "
                  f"시간 {(b['days']-a['days'])*24:+6.2f}시간 "
                  f"({100*(b['days']-a['days'])/a['days']:+.2f}%)")


if __name__ == '__main__':
    main()
