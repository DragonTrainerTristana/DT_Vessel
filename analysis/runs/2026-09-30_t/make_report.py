"""make_report.py — report_parse.py 출력(markdown) + 그림 → HTML (Edge headless 로 PDF). stdlib 만.

  python make_report.py --title "..." --md <parse.md> [--md2 <...>] --figs a.png,b.png --notes notes.md --out report.html
markdown 은 표(| … |)·헤더(#)·리스트(-)·문단만 변환(간단 변환기).
"""
import argparse
import html
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
CSS = """
@page { size: A4; margin: 12mm 11mm; }
body { font-family: "Malgun Gothic", sans-serif; font-size: 9.4pt; line-height: 1.42; color: #111; }
h1 { font-size: 15pt; margin: 0 0 4px; } h2 { font-size: 12pt; margin: 14px 0 5px; border-bottom: 2px solid #333; padding-bottom: 2px; }
h3 { font-size: 10.5pt; margin: 9px 0 3px; }
table { border-collapse: collapse; margin: 4px 0 6px; font-size: 8pt; }
th, td { border: 1px solid #999; padding: 1px 4px; text-align: right; white-space: nowrap; } th { background: #eee; }
td:first-child, th:first-child { text-align: left; }
.goal { border: 2px solid #b00; background: #fff4f4; padding: 6px 10px; margin: 6px 0; }
.box { background: #f4f4f4; border-left: 4px solid #555; padding: 5px 9px; margin: 5px 0; }
img { width: 100%; margin: 3px 0; } .pb { page-break-before: always; }
ul { margin: 3px 0 3px 17px; padding: 0; } li { margin: 1px 0; }
"""


def md_to_html(md):
    out, in_tab, in_ul = [], False, False
    for line in md.splitlines():
        if line.startswith('|'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if all(re.fullmatch(r'-{3,}', c) for c in cells):
                continue
            if not in_tab:
                out.append('<table>'); in_tab = True; tag = 'th'
            else:
                tag = 'td'
            out.append('<tr>' + ''.join(f'<{tag}>{inline(c)}</{tag}>' for c in cells) + '</tr>')
            continue
        if in_tab:
            out.append('</table>'); in_tab = False
        if line.startswith('- '):
            if not in_ul:
                out.append('<ul>'); in_ul = True
            out.append(f'<li>{inline(line[2:])}</li>'); continue
        if in_ul:
            out.append('</ul>'); in_ul = False
        m = re.match(r'^(#{1,3}) (.*)', line)
        if m:
            out.append(f'<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>'); continue
        if line.strip():
            out.append(f'<p>{inline(line)}</p>')
    if in_tab:
        out.append('</table>')
    if in_ul:
        out.append('</ul>')
    return '\n'.join(out)


def inline(t):
    t = html.escape(t)
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'`(.+?)`', r'<code>\1</code>', t)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--title', required=True)
    ap.add_argument('--sub', default='')
    ap.add_argument('--goal', default='', help='저자 목표 박스 markdown 파일')
    ap.add_argument('--md', action='append', default=[], help='markdown 파일(여러 번)')
    ap.add_argument('--figs', default='', help='그림 경로(쉼표)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    parts = [f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{html.escape(a.title)}</title><style>{CSS}</style></head><body>',
             f'<h1>{html.escape(a.title)}</h1>', f'<div class="sub">{inline(a.sub)}</div>']
    if a.goal and os.path.exists(a.goal):
        parts.append('<div class="goal">' + md_to_html(open(a.goal, encoding='utf-8').read()) + '</div>')
    for i, m in enumerate(a.md):
        if os.path.exists(m):
            parts.append(md_to_html(open(m, encoding='utf-8').read()))
    for f in [x for x in a.figs.split(',') if x]:
        if os.path.exists(f):
            parts.append(f'<div class="pb"></div><img src="{html.escape(os.path.abspath(f))}">')
    parts.append('</body></html>')
    open(a.out, 'w', encoding='utf-8').write('\n'.join(parts))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
