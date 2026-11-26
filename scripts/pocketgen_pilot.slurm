#!/bin/bash
#SBATCH --job-name=dh307-pocketgen-pilot
#SBATCH --partition=gpu
#SBATCH --account=robot-perception-and-learning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=rmgpu003
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pocketgen-pilot-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_HOME="$PWD/data/torch-cache"
export PYTHONPATH="$PWD/vendor/PocketGen"
export PATH="$PWD/envs/pocketgen/bin:$PATH"
exec envs/pocketgen/bin/python scripts/pocketgen_pilot.py \
  --seed "${PILOT_SEED:-42}" --output "runs/pocketgen-pilot-${SLURM_JOB_ID}"
