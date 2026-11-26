#!/bin/bash
#SBATCH --job-name=dh307-score-pilot
#SBATCH --partition=serial
#SBATCH --account=robot-perception-and-learning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/score-pilot-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PATH="$PWD/envs/flowr/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec envs/flowr/bin/python scripts/score_pair.py \
  --protein vendor/PocketGen/examples/2p16/2p16.pdb \
  --ligand vendor/PocketGen/examples/2p16/2p16_ligand.sdf \
  --output "runs/scoring-pilot-${SLURM_JOB_ID}"
