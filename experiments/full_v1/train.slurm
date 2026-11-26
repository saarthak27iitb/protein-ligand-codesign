#!/bin/bash
#SBATCH --job-name=dh307-full-rl
#SBATCH --partition=gpu
#SBATCH --account=robot-perception-and-learning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=rmgpu003
#SBATCH --mem=96G
#SBATCH --time=3-23:00:00
#SBATCH --requeue
#SBATCH --output=logs/full-%x-%j.out
set -euo pipefail
cd "${DH307_ROOT:?Set DH307_ROOT using research.submit}"
export PYTHONPATH="$DH307_CODE_ROOT:$DH307_CODE_ROOT/vendor/flowr"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=4 FLOWR_NUM_WORKERS=0 WANDB_MODE=disabled
export PATH="$DH307_ROOT/envs/flowr/bin:$PATH"
exec envs/flowr/bin/python -u -m research.run \
 --setting "${SETTING:?}" --seed "${POLICY_SEED:?}" \
 --config "$DH307_CODE_ROOT/experiments/full_v1/config.json" \
 --manifest "$DH307_ROOT/data/full_v1/manifest.json" \
 --output "$DH307_ROOT/runs/full_v1/$SETTING/seed_$POLICY_SEED"
