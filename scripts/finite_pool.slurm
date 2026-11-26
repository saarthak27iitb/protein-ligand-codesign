#!/bin/bash
#SBATCH --job-name=dh307-controller-pilot
#SBATCH --partition=serial
#SBATCH --account=robot-perception-and-learning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/controller-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${BANK:?}" "${EXPERIMENT_MODE:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
envs/flowr/bin/python scripts/capture_provenance.py
if [ "$EXPERIMENT_MODE" = game-and-joint ]; then
  modes=(adversarial joint)
else
  modes=("$EXPERIMENT_MODE")
fi
for mode in "${modes[@]}"; do
  envs/flowr/bin/python scripts/finite_pool_experiments.py \
    --bank "$BANK" --mode "$mode" --output "runs/${mode}-${SLURM_JOB_ID}"
done
