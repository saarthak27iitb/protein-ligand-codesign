#!/bin/bash
#SBATCH --job-name=dh307-flowr-preflight
#SBATCH --partition=serial
#SBATCH --account=robot-perception-and-learning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/flowr-preflight-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PATH="$PWD/envs/flowr/bin:$PATH"
export PYTHONPATH="$PWD/vendor/flowr" FLOWR_NUM_WORKERS=0 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec envs/flowr/bin/python -u scripts/flowr_preflight.py
