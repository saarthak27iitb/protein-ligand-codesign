"""Run released FLOWR on a specified protein/ligand reference pair."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('SLURM_JOB_ID'):
    from capture_provenance import capture
    capture()
parser = argparse.ArgumentParser()
parser.add_argument('--protein', type=Path, required=True)
parser.add_argument('--ligand', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mode', choices=['independent','scaffold','functional-group','linker','substructure'], default='independent')
parser.add_argument('--substructure', help='Retained fragment SMILES for substructure mode')
parser.add_argument('--samples', type=int, default=4)
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
if min(args.samples,args.steps) < 1:
    parser.error('samples and steps must be positive')
if args.mode == 'substructure' and not args.substructure:
    parser.error('substructure mode needs --substructure')
args.output = args.output.resolve()
args.output.mkdir(parents=True,exist_ok=False)
status = {'stage':'initializing','seed':args.seed,'mode':args.mode,
          'protein':str(args.protein.resolve()),'reference_ligand':str(args.ligand.resolve()),
          'purpose':'pair-interface pilot; frozen FLOWR, no training'}
start = time.monotonic()
def save():
    (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
save()
try:
    import torch
    from rdkit import Chem
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable')
    samples = args.output/'generated'
    samples.mkdir()
    cmd = [sys.executable,'-m','flowr.gen.generate_from_pdb',
           '--pdb_file',str(args.protein.resolve()),'--ligand_file',str(args.ligand.resolve()),
           '--arch','pocket','--pocket_type','holo','--dataset','spindr',
           '--data_path',str(args.output),'--ckpt_path',str(ROOT/'checkpoints'/'flowr_noHs.ckpt'),
           '--save_dir',str(samples),'--cut_pocket','--pocket_cutoff','6',
           '--gpus','1','--num_workers','1','--batch_cost',str(args.samples),
           '--sample_n_molecules_per_target',str(args.samples),'--max_sample_iter','0',
           '--integration_steps',str(args.steps),'--seed',str(args.seed),
           '--categorical_strategy','uniform-sample']
    flags = {'scaffold':'--scaffold_inpainting','functional-group':'--func_group_inpainting',
             'linker':'--linker_inpainting','substructure':'--substructure_inpainting'}
    if args.mode in flags:
        cmd.append(flags[args.mode])
    if args.mode == 'substructure':
        cmd.extend(['--substructure',args.substructure])
    status.update(stage='generating',command=cmd)
    save()
    env = dict(os.environ,PYTHONPATH=str(ROOT/'vendor'/'flowr'),FLOWR_NUM_WORKERS='0',WANDB_MODE='disabled')
    with (args.output/'generation.log').open('w') as log:
        subprocess.run(cmd,cwd=ROOT/'vendor'/'flowr',env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    paths = list(samples.glob('samples_unfiltered_*.pt'))
    if len(paths) != 1:
        raise RuntimeError(f'Expected one predictions artifact, got {paths}')
    predictions = torch.load(paths[0],map_location='cpu',weights_only=False)
    records = []
    for i,mol in enumerate(predictions['gen_ligs']):
        row = {'index':i,'valid':False}
        try:
            if mol is None:
                raise ValueError('No molecule')
            Chem.SanitizeMol(mol)
            if len(Chem.GetMolFrags(mol)) != 1:
                raise ValueError('Disconnected molecule')
            if args.mode == 'substructure' and not mol.HasSubstructMatch(Chem.MolFromSmiles(args.substructure)):
                raise ValueError('Retained substructure missing')
            path = args.output/f'candidate_{i}.sdf'
            with Chem.SDWriter(str(path)) as writer:
                writer.write(mol)
            row.update(valid=True,path=str(path),smiles=Chem.MolToSmiles(Chem.RemoveHs(mol)))
        except Exception as exc:
            row['error'] = str(exc)
        records.append(row)
    (args.output/'candidates.json').write_text(json.dumps(records,indent=2)+'\n')
    if not any(r['valid'] for r in records):
        raise RuntimeError('No valid connected candidates')
    status.update(stage='complete',seconds=time.monotonic()-start,generated=len(records),valid=sum(r['valid'] for r in records))
    save()
except Exception as exc:
    status.update(stage='failed',seconds=time.monotonic()-start,error=str(exc))
    save()
    raise
