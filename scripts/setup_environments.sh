#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
task_root="$PWD"
export MAMBA_ROOT_PREFIX="$task_root/.mamba"
export CONDA_PKGS_DIRS="$task_root/.mamba/pkgs"
bin/micromamba create -y -p "$task_root/envs/flowr" -f configs/flowr-env.yml
envs/flowr/bin/python -m pip install -r configs/flowr-pip.txt
bin/micromamba create -y -p "$task_root/envs/pocketgen" -f configs/pocketgen-env.yml
envs/pocketgen/bin/python -m pip install -r configs/pocketgen-pip.txt
envs/pocketgen/bin/python -m pip install --only-binary=:all: \
  torch_scatter torch_cluster torch_sparse torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
envs/flowr/bin/python -m pip freeze > reports/flowr-pip-freeze.txt
envs/pocketgen/bin/python -m pip freeze > reports/pocketgen-pip-freeze.txt
