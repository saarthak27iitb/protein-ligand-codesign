#!/bin/bash
#SBATCH --job-name=dh307-full-splits
#SBATCH --partition=small
#SBATCH --account=robot-perception-and-learning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/full-splits-%j.out
set -euo pipefail
cd "${DH307_ROOT:?}"
export PYTHONPATH="$DH307_CODE_ROOT:$DH307_CODE_ROOT/vendor/flowr"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec envs/flowr/bin/python -u -m research.stage_data --stage finalize --config "$DH307_CODE_ROOT/experiments/full_v1/config.json" --output "$PWD/data/full_v1"
