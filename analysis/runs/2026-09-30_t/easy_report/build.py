# 쉬운 설명 PDF 빌드: make_figs.py 로 만든 SVG 를 report_template.html 에 넣어 report.html 생성 → Chrome headless 로 PDF.
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
html = open(os.path.join(HERE, 'report_template.html'), encoding='utf-8').read()
for key in ('fig_curves', 'fig_standon', 'fig_scripted'):
    html = html.replace('{{' + key + '}}', open(os.path.join(HERE, key + '.svg'), encoding='utf-8').read())
out_html = os.path.join(HERE, 'report.html')
open(out_html, 'w', encoding='utf-8').write(html)
pdf = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'report.pdf')
chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
subprocess.run([chrome, '--headless', '--disable-gpu', '--no-pdf-header-footer',
                f'--print-to-pdf={pdf}', 'file://' + out_html], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(pdf)
