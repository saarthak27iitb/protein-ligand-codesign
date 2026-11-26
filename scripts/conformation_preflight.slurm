#!/bin/bash
#SBATCH --job-name=dh307-conformer-check
#SBATCH --partition=serial
#SBATCH --account=robot-perception-and-learning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/conformer-check-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD/scripts"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
envs/flowr/bin/python -u - <<'PY'
import os
from pathlib import Path
from make_conformations import make_conformations
from mutation_lattice import residues
protein=Path('vendor/PocketGen/examples/2p16/2p16.pdb')
manifest=make_conformations(protein,Path('runs')/('conformer-check-'+os.environ['SLURM_JOB_ID']),
                            frames=1,steps_per_frame=10,platform_name='CPU',minimization_steps=10)
before=residues(protein)
after=residues(manifest['files'][0])
assert {k:v[0][17:20] for k,v in before.items()} == {k:v[0][17:20] for k,v in after.items()}
print('Fixed-sequence MD preparation and short integration passed')
PY
