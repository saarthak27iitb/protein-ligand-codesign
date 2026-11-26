#!/bin/bash
#SBATCH --job-name=dh307-full-data
#SBATCH --partition=small
#SBATCH --account=robot-perception-and-learning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/full-data-%j.out
set -euo pipefail
cd "${DH307_ROOT:?Set DH307_ROOT to this checkout}"
export PYTHONPATH="$PWD:$PWD/vendor/flowr"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FLOWR_NUM_WORKERS=0
exec envs/flowr/bin/python -u -m research.stage_data --stage export --config experiments/full_v1/config.json --output data/full_v1
