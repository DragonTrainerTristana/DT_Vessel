#!/bin/bash
# 2026-10-26 A–Z 진단 보고서: Markdown → HTML(pandoc) → PDF(Chrome headless). 새 학습·평가 없음.
# 사용: bash make_pdf.sh <입력.md> <출력.pdf>
set -euo pipefail
IN="$1"; OUT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; HTML="${OUT%.pdf}.html"
CSS="$(cd "$(dirname "$0")" && pwd)/report.css"
pandoc "$IN" -f gfm -t html5 -s --metadata title="$(head -1 "$IN" | sed 's/^# //')" --css "$CSS" --embed-resources -o "$HTML"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$OUT" "file://$HTML" 2>/dev/null
echo "$OUT"
