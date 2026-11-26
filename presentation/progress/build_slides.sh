#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
project_root="$(cd ../.. && pwd)"
compiler="$project_root/Robotics/piper-handover/tools/tectonic/tectonic"
mkdir -p build
if [ -x "$compiler" ]; then
  export XDG_CACHE_HOME="$project_root/Robotics/piper-handover/tools/tectonic/cache"
  "$compiler" --keep-logs --outdir build slides.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --outdir build slides.tex
elif command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build slides.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build slides.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build slides.tex
else
  printf '%s\n' 'Install Tectonic or a TeX distribution with Beamer.' >&2
  exit 1
fi
cp build/slides.pdf DH307_research_meeting.pdf
cp build/slides.pdf DH307_progress_update.pdf
