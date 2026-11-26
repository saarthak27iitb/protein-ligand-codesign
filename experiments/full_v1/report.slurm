#!/bin/bash
#SBATCH --job-name=dh307-full-report
#SBATCH --partition=small
#SBATCH --account=robot-perception-and-learning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/full-report-%j.out
set -euo pipefail
cd "${DH307_ROOT:?}"
export PYTHONPATH="$DH307_CODE_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec envs/flowr/bin/python -u -m research.report --root runs/full_v1 --manifest data/full_v1/manifest.json --config "$DH307_CODE_ROOT/experiments/full_v1/config.json"
