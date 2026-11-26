#!/bin/bash
#SBATCH --job-name=dh307-rl-fragment-pilot
#SBATCH --partition=gpu
#SBATCH --account=robot-perception-and-learning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=rmgpu003
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rl-fragment-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PATH="$PWD/envs/flowr/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
envs/flowr/bin/python scripts/capture_provenance.py
exec envs/flowr/bin/python scripts/rl_fragment.py \
  --protein vendor/PocketGen/examples/2p16/2p16.pdb \
  --ligand vendor/PocketGen/examples/2p16/2p16_ligand.sdf \
  --output "runs/rl-fragment-${SLURM_JOB_ID}"
