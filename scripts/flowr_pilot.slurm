#!/bin/bash
#SBATCH --job-name=dh307-flowr-pilot
#SBATCH --partition=gpu
#SBATCH --account=robot-perception-and-learning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=rmgpu003
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/flowr-pilot-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled
export FLOWR_NUM_WORKERS=0
export PYTHONPATH="$PWD/vendor/flowr"
export PATH="$PWD/envs/flowr/bin:$PATH"
export MPLCONFIGDIR="$PWD/runs/matplotlib-${SLURM_JOB_ID}"
envs/flowr/bin/python scripts/capture_provenance.py
exec envs/flowr/bin/python scripts/flowr_pilot.py \
  --mode "${PILOT_MODE:-independent}" --seed "${PILOT_SEED:-42}" \
  --output "runs/flowr-pilot-${SLURM_JOB_ID}"
