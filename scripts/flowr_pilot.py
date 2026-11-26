"""Small, reproducible upstream inference pilot, not RL or co-design training."""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('SLURM_JOB_ID'):
    from capture_provenance import capture
    capture()
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['independent', 'scaffold', 'functional-group', 'linker'], default='independent')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--targets', type=int, default=2)
parser.add_argument('--samples', type=int, default=4)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if min(args.targets, args.samples, args.steps) < 1:
    parser.error('targets, samples and steps must be positive')
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
status = {'stage': 'initializing', 'purpose': 'inference pilot only', 'mode': args.mode, 'seed': args.seed}
status_path = args.output / 'status.json'
def save_status():
    status_path.write_text(json.dumps(status, indent=2) + '\n')
save_status()
start = time.monotonic()
try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable: run this pilot through Slurm')
    source = list((ROOT / 'data' / 'spindr').rglob('val.smol'))
    if len(source) != 1:
        raise RuntimeError(f'Expected one validation file, found {source}')
    # Official .smol container is a pickled list of serialized complexes.
    raw = source[0].read_bytes()
    systems = pickle.loads(raw)
    if not isinstance(systems, list) or not all(isinstance(x, bytes) for x in systems):
        raise RuntimeError('Unexpected upstream .smol container format')
    selected = systems[:args.targets]
    if len(selected) != args.targets:
        raise RuntimeError('Insufficient validation targets')
    subset_dir = args.output / 'input'
    subset_dir.mkdir()
    (subset_dir / 'val.smol').write_bytes(pickle.dumps(selected))
    out = args.output / 'samples'
    out.mkdir()
    command = [sys.executable, '-m', 'flowr.gen.generate_from_smol',
               '--arch', 'pocket', '--pocket_noise', 'fix', '--ckpt_path', str(ROOT/'checkpoints'/'flowr_noHs.ckpt'),
               '--data_path', str(subset_dir), '--dataset', 'spindr', '--dataset_split', 'val',
               '--save_dir', str(out), '--gpus', '1', '--mp_index', '1', '--num_workers', '0',
               '--batch_cost', str(args.samples), '--sample_n_molecules_per_target', str(args.samples),
               '--max_sample_iter', '0', '--integration_steps', str(args.steps), '--seed', str(args.seed),
               '--categorical_strategy', 'uniform-sample']
    flags = {'scaffold': '--scaffold_inpainting', 'functional-group': '--func_group_inpainting',
             'linker': '--linker_inpainting'}
    if args.mode in flags:
        command.append(flags[args.mode])
    status.update(stage='sampling', command=command, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0), validation_sha256=hashlib.sha256(raw).hexdigest(),
                  targets=args.targets, samples_requested_per_target=args.samples,
                  steps=args.steps, target_serialization_sha256=[hashlib.sha256(x).hexdigest() for x in selected])
    save_status()
    env = dict(os.environ, PYTHONPATH=str(ROOT/'vendor'/'flowr'), WANDB_MODE='disabled')
    with (args.output/'inference.log').open('w') as log:
        subprocess.run(command, cwd=ROOT/'vendor'/'flowr', env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors
    predictions = torch.load(out/'predictions_multi_1.pt', map_location='cpu', weights_only=False)
    rows = []
    valid_smiles = []
    for i, molecules in enumerate(predictions['gen_ligs']):
        with Chem.SDWriter(str(out/f'target_{i}.sdf')) as writer:
            for j, mol in enumerate(molecules):
                row = {'target': i, 'candidate': j, 'valid': False}
                try:
                    if mol is None:
                        raise ValueError('No molecule returned')
                    Chem.SanitizeMol(mol)
                    smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
                    row.update(valid=True, connected=len(Chem.GetMolFrags(mol)) == 1,
                               smiles=smi, qed=QED.qed(mol), molecular_weight=Descriptors.MolWt(mol))
                    valid_smiles.append(smi)
                    writer.write(mol)
                except Exception as exc:
                    row['error'] = str(exc)
                rows.append(row)
    (args.output/'molecules.json').write_text(json.dumps(rows, indent=2)+'\n')
    status.update(stage='complete', generated=len(rows), sanitized=sum(x['valid'] for x in rows),
                  unique_valid=len(set(valid_smiles)), seconds=time.monotonic()-start,
                  interpretation='Pilot chemistry metrics only; no affinity, RL, co-design or benchmark claim')
    save_status()
except Exception as exc:
    status.update(stage='failed', error=str(exc), seconds=time.monotonic()-start)
    save_status()
    raise
