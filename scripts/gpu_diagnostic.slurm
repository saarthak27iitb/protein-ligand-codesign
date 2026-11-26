#!/bin/bash
#SBATCH --job-name=dh307-gpu-check
#SBATCH --partition=gpu
#SBATCH --account=robot-perception-and-learning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/gpu-check-%j.out
set -eu
cd "${SLURM_SUBMIT_DIR:?}"
hostname
nvidia-smi
envs/pocketgen/bin/python - <<'PY'
import os, torch
print('torch:', torch.__version__, 'compiled CUDA:', torch.version.cuda, flush=True)
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)
print('SLURM_JOB_GPUS:', os.environ.get('SLURM_JOB_GPUS'), flush=True)
print('CUDA available:', torch.cuda.is_available(), flush=True)
torch.cuda.init()
print(torch.ones(2, device='cuda'), flush=True)
PY
