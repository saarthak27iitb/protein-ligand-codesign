#!/bin/bash
#SBATCH --job-name=dh307-candidate-bank
#SBATCH --partition=gpu
#SBATCH --account=robot-perception-and-learning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=rmgpu003
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/candidate-bank-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PATH="$PWD/envs/flowr/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
envs/flowr/bin/python scripts/capture_provenance.py
exec envs/flowr/bin/python scripts/build_candidate_bank.py \
  --output "runs/candidate-bank-${SLURM_JOB_ID}"
