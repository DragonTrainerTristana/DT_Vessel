"""build_t_report.py — t_ 배치 최종 보고서 HTML 조립 → Edge headless PDF. base anaconda python(markdown) 으로 실행.

  python build_t_report.py --pdf "C:/Users/OSH/Dropbox/Private_Paper_Project/2026-09-30_보상v3_t배치결과.pdf"
읽는 것: _repro_out_t/report/*.md · _primary_table.md · _secondary_table.md · runs/2026-09-30_t/rootcause_2026-09-30.md · 그림 png.
숫자는 위 파일에 있는 값만 씀(이 스크립트는 조립만).
"""
import argparse
import html
import os
import re
import subprocess
import sys

import markdown

sys.stdout.reconfigure(encoding='utf-8')
O = r'C:\work\DT_Vessel_v3\Python\_repro_out_t'
RP = os.path.join(O, 'report')
HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'

CSS = """
@page { size: A4; margin: 11mm 10mm; }
body { font-family: "Malgun Gothic", sans-serif; font-size: 9.2pt; line-height: 1.42; color: #111; }
h1 { font-size: 15pt; margin: 0 0 3px; }
h2 { font-size: 12pt; margin: 14px 0 5px; border-bottom: 2px solid #333; padding-bottom: 2px; page-break-after: avoid; }
h3 { font-size: 10.3pt; margin: 9px 0 3px; page-break-after: avoid; }
.sub { color: #555; font-size: 8.5pt; margin-bottom: 6px; }
table { border-collapse: collapse; margin: 4px 0 7px; font-size: 7.6pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 1px 3px; text-align: right; }
th { background: #eee; }
td:first-child, th:first-child { text-align: left; white-space: nowrap; }
.goal { border: 2px solid #b00; background: #fff4f4; padding: 5px 10px; margin: 6px 0; }
.warn { border: 2px solid #c60; background: #fff7ec; padding: 5px 10px; margin: 6px 0; }
.fig { page-break-inside: avoid; margin: 6px 0 10px; }
.fig img { width: 100%; }
.cap { font-size: 8.3pt; color: #333; margin-top: 2px; }
ul, ol { margin: 3px 0 3px 18px; padding: 0; } li { margin: 1px 0; }
code { font-size: 8pt; background: #f3f3f3; }
.pb { page-break-before: always; }
@page wide { size: A4 landscape; margin: 9mm 9mm; }
.wide { page: wide; }
.wide table { font-size: 6.9pt; }
.wide th { white-space: normal; }
.fig img { max-height: 245mm; object-fit: contain; }
"""


def md(text, demote=0):
    """markdown → html. 2칸 들여쓴 목록을 4칸으로(파이썬 markdown 중첩 규칙), 헤더 강등."""
    lines = []
    is_list = lambda x: bool(re.match(r'^\s*([-*]|\d+\.) ', x))
    for ln in text.splitlines():
        m = re.match(r'^( +)([-*]|\d+\.) ', ln)
        if m:
            ln = ' ' * (len(m.group(1)) * 2) + ln[len(m.group(1)):]
        if demote and re.match(r'^#{1,5} ', ln):
            ln = '#' * demote + ln
        if is_list(ln) and lines and lines[-1].strip() and not is_list(lines[-1]) and not lines[-1].lstrip().startswith('|'):
            lines.append('')
        lines.append(ln)
    return markdown.markdown('\n'.join(lines), extensions=['tables', 'sane_lists'])


def rd(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def fig(path, cap):
    return (f'<div class="fig"><img src="file:///{html.escape(os.path.abspath(path)).replace(os.sep, "/")}">'
            f'<div class="cap">{html.escape(cap)}</div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', required=True)
    a = ap.parse_args()
    rc = rd(os.path.join(HERE, 'rootcause_2026-09-30.md'))
    rc = rc.split('\n', 1)[1] if rc.startswith('# ') else rc          # 파일 제목 줄 제거
    rc = re.sub(r'^- 조사 원자료.*\n', '', rc, flags=re.M)
    FIX = '[보고서 작성 때 정정]'
    rc = re.sub(r'^- 주의\(10-01\).*\n', '', rc, flags=re.M)
    subs = [
        ('56 m 밖이라 OFF 는 이걸 못 봄. 그래서 이 비용은 통신 배만 줄일 수 있음.',
         '그중 56 m 레이더 밖 몫(약 42–44 %)은 OFF 가 못 봄. 그중 통신으로 보이는 31–33 % 는 통신 배만 줄일 수 있고 10–11 % 는 아무도 못 봄 [보고서 작성 때 문구 보정 — 인과 확신도 중간].'),
        ('목표 근처 선회 함정에 더 자주 빠짐(', '목표 근처 선회 함정에 더 자주 빠짐[확신도 중간 — 보고서 작성 때 덧붙임](' ),
        ('## 원인 (확인됨)', '## 관찰과 기전 (확신도는 항목별 표기 — 모두 확정된 인과는 아님) [보고서 작성 때 정정 — 원문 제목 "원인 (확인됨)"]'),
        ('| oColl · TO | – | 9/12 |', '| oColl | 1/12 | 8/12 (동률 1) ' + FIX + ' |\n| TO | 1/12 | 9/12 ' + FIX + ' |'),
        ('   - 최선 OFF 풀 (`_primary_table.md` 38–41행): goal·vColl 통과, dcpa 불통과\n     - offb 는 풀에 없음. 넣으면 goal 도 불통과 (a6_s47 55.8 < offb 평균 55.9)',
         '   - 최선 OFF 풀 ' + FIX + ': 원문(풀 버그 상태)은 \'goal·vColl 통과, dcpa 불통과 · offb 는 풀에 없음, 넣으면 goal 도 불통과\'. 풀 버그(옛 OFF 를 시드 43 하나로만 읽음·x_off 누락) 수정 뒤 t_off·s_off·x_off 각 3시드 기준 → goal 불통과(s_off 56.53 > a6_s47 55.8), vColl 통과, dcpa 불통과'),
        ('  - a6 이탈 결정: #6 이 목표 직진 시 −0.569 인데 실제 −0.310 → 의도적 회피임\n  - OFF 이탈 결정: −0.682 vs −0.655 → 우연한 이탈임',
         '  - a6 이탈 결정: #6 이 목표 직진 시 −0.569 인데 실제 −0.310 → 의도적 회피로 보임\n  - OFF 이탈 결정: −0.682 vs −0.655 → 우연한 이탈로 보임\n  - ' + FIX + ' 위 두 줄은 s43 한 시드 반사실임. 3시드 근거는 아래 \'주변 배 무게중심 반대쪽 이탈 비율\' (3/3, 두 OFF 갈래 모두 넘음)'),
        ('11. **a8·a2·a4 판정은 비공식**\n    - scratch 에서 다시 돌린 값이고 최선 OFF 풀 검사 없음\n    - 스펙상 a2·a4 는 사다리 보고용',
         '11. **a8·a2·a4 판정은 비공식**\n    - scratch 에서 다시 돌린 값이고 최선 OFF 풀 검사 없음\n    - 스펙상 a2·a4 는 사다리 보고용\n    - [보고서 작성 때 덧붙임] 공식 판정은 이 보고서의 판정표 2 (eval2 원문, 풀 버그 수정 뒤 최선 OFF 풀 포함)'),
    ]
    for a_, b_ in subs:
        assert rc.count(a_) == 1, a_[:40]
        rc = rc.replace(a_, b_)
    parts = [
        '<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>2026-09-30 t_ 배치 결과</title>',
        f'<style>{CSS}</style></head><body>',
        '<h1>2026-09-30 t_ 배치 결과 — 보상 v3 · 복원형 latent 사다리(a2·a4·a6·a8)</h1>',
        '<div class="sub">작성 2026-10-01 · 코드 feat/reward-v3 3aa0636 · 스펙 2026-09-30-reward-v3-decode-sweep-design.md(사전등록) · '
        '평가 256 env × 10000 결정, seed 999 · 뿌리 9.04M → 갈래 16.06M · 시드 43·46·47</div>',
        '<div class="goal">' + md(rd(os.path.join(RP, '00_goal.md'))) + '</div>',
        md(rd(os.path.join(RP, '01_summary.md'))),
        md(rd(os.path.join(RP, 'summary_table.md'))),
        md(rd(os.path.join(RP, '01c_deviations.md'))),
        '<div class="wide"><h2>판정표 1 — 주 처치 t_a6 대 t_off (사전등록)</h2><p>이 표 안의 <b>통과</b> = 3/3 &amp; |평균차| &gt; N 만 본 것(풀 검사 전). 최종 판정은 표 끝 \'사전등록 최종 판정\' 줄 — 도착·선박 충돌·통과거리는 최선 OFF 풀도 넘어야 함.</p>',
        md(rd(os.path.join(O, '_primary_table.md')), demote=1),
        '<h2 class="pb">판정표 2 — 보조 t_a8 · 사다리 t_a2·t_a4 대 t_off (사다리는 스펙상 확정 주장 없음)</h2><p>판정표 1 과 같음: 표 안의 <b>통과</b> 는 풀 검사 전, 최종은 끝의 \'사전등록 최종 판정\' 줄.</p>',
        md(rd(os.path.join(O, '_secondary_table.md')), demote=1) + '</div>',
        '<h2 class="pb">목표 2 — reward 곡선</h2>',
        md(rd(os.path.join(RP, 'goal2_train.md')), demote=1),
        fig(os.path.join(O, '_reward_v3.png'), '그림 1. 결정당 reward(학습 로그) — raw(옅음) + 중심 이동평균 20 update, 0–16.06M 전체와 분기 뒤 확대. '
            '첫 update(+0.16 근처)는 32결정·사건 0 이라 비교 불가. 분기 뒤 통신 곡선이 대체로 위지만 s43 초반·s47 후반에 OFF 아래 구간이 있음'),
        fig(os.path.join(O, '_ep_return.png'), '그림 2. 에피소드 반환(학습 로그, 시계 잘린 첫 에피소드 포함) — 같은 판정 기준을 적용함(위 표)'),
        '<h2 class="pb">원인 분석 (조사 4갈래 · 반박 검증 · 종합)</h2>',
        md(rc, demote=1),
        md(rd(os.path.join(RP, 'oldckpt.md')), demote=0),
        '<div class="warn">' + md(rd(os.path.join(RP, '05_measure.md'))) + '</div>',
        md(rd(os.path.join(RP, 'evalcurve.md')), demote=1),
        fig(os.path.join(RP, 'evalcurve.png'), '그림 3. 고정 평가 학습곡선 — ⚠ 평가 창 2000 결정이라 긴 에피소드·시간초과가 빠진 편향 값(모든 팔이 부풀려짐, 에피소드가 긴 a6 가 약 2배 더). 비교·판정에 쓰지 말 것'),
        md(rd(os.path.join(RP, '06_appendix.md'))),
        '</body></html>',
    ]
    out_html = os.path.join(RP, 't_report.html')
    with open(out_html, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(parts))
    print('html', out_html)
    pdf = os.path.abspath(a.pdf)
    cmd = [EDGE, '--headless', '--disable-gpu', '--no-pdf-header-footer', '--run-all-compositor-stages-before-draw',
           f'--print-to-pdf={pdf}', 'file:///' + out_html.replace(os.sep, '/')]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    print('edge rc', r.returncode, (r.stderr or '')[-300:])
    print('pdf', pdf, os.path.getsize(pdf) if os.path.exists(pdf) else 'MISSING')


if __name__ == '__main__':
    main()
