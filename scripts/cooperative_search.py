"""Frozen conditional-model alternating search with an explicit shared proxy reward.

This is the first co-design baseline, not cooperative RL training or a unified model.
All proposals, invalid candidates, evaluations and acceptance decisions are retained.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from score_pair import score

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--protein',type=Path,required=True)
parser.add_argument('--ligand',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--rounds',type=int,default=2)
parser.add_argument('--ligand-proposals',type=int,default=4)
parser.add_argument('--pocket-proposals',type=int,default=2)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--steps',type=int,default=100)
parser.add_argument('--ligand-only',action='store_true')
args=parser.parse_args()
if min(args.rounds,args.ligand_proposals,args.pocket_proposals,args.steps)<1:
    parser.error('budgets must be positive')
args.output=args.output.resolve()
args.output.mkdir(parents=True,exist_ok=False)
start=time.monotonic()
status={'stage':'initializing','algorithm':'frozen alternating shared-reward search',
        'training':False,'ligand_only':args.ligand_only,'seed':args.seed,
        'limitation':'No foldability/stability assessment; example-based integration pilot only'}
history=[]
counts={'flowr_calls':0,'pocketgen_calls':0,'ligand_proposals':0,'pocket_proposals':0,'score_calls':0}
def save():
    (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
def run(model,script,argv,folder):
    env=dict(os.environ)
    env.update(PATH=str(ROOT/'envs'/model/'bin')+os.pathsep+env['PATH'],
               PYTHONPATH=str(ROOT/'vendor'/('flowr' if model=='flowr' else 'PocketGen')),
               TORCH_HOME=str(ROOT/'data'/'torch-cache'),FLOWR_NUM_WORKERS='0',WANDB_MODE='disabled')
    with folder.open('w') as log:
        subprocess.run([str(ROOT/'envs'/model/'bin'/'python'),str(ROOT/'scripts'/script)]+argv,
                       cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
def evaluate(protein,ligand,label):
    counts['score_calls']+=1
    metrics=score(protein,ligand,args.output/label,seed=args.seed)
    strain=metrics['mmff94s_strain_kcal_mol']
    feasible=(metrics['connected'] and metrics['clashing_atom_pairs_0_4A']<=5 and
              strain is not None and metrics['mmff_minimization_converged'] and strain<=100 and
              0.8*reference_atoms<=metrics['heavy_atoms']<=1.2*reference_atoms)
    utility=-metrics['vina_score_input_pose']-0.01*strain if feasible else None
    return metrics,utility
save()
try:
    from rdkit import Chem
    mol=next(iter(Chem.SDMolSupplier(str(args.ligand),removeHs=False)))
    if mol is None:
        raise ValueError('Invalid reference ligand')
    reference_atoms=mol.GetNumHeavyAtoms()
    protein,ligand=args.protein.resolve(),args.ligand.resolve()
    metrics,utility=evaluate(protein,ligand,'initial_score')
    if utility is None:
        raise RuntimeError('Reference pair fails the predefined feasibility constraints')
    initial_utility=utility
    history.append({'round':-1,'phase':'initial','metrics':metrics,'utility':utility,'accepted':True})
    status.update(stage='searching',initial_utility=utility)
    save()
    for round_index in range(args.rounds):
        seed=args.seed+100*round_index
        generated=args.output/f'round_{round_index}_ligands'
        run('flowr','flowr_pair.py',['--protein',str(protein),'--ligand',str(ligand),
            '--output',str(generated),'--samples',str(args.ligand_proposals),
            '--seed',str(seed),'--steps',str(args.steps)],args.output/f'round_{round_index}_flowr.log')
        counts['flowr_calls']+=1
        for row in json.loads((generated/'candidates.json').read_text()):
            counts['ligand_proposals']+=1
            record={'round':round_index,'phase':'ligand','proposal':row,'accepted':False}
            try:
                if not row['valid']:
                    raise ValueError(row.get('error','Invalid ligand'))
                m,u=evaluate(protein,row['path'],f'r{round_index}_ligand_score_{row["index"]}')
                record.update(metrics=m,utility=u)
                if u is not None and u>utility:
                    ligand,utility=Path(row['path']),u
                    record['accepted']=True
            except Exception as exc:
                record['error']=str(exc)
            history.append(record)
            save()
        if not args.ligand_only:
            generated=args.output/f'round_{round_index}_pockets'
            run('pocketgen','pocketgen_pilot.py',['--protein',str(protein),'--ligand',str(ligand),
                '--output',str(generated),'--samples',str(args.pocket_proposals),
                '--seed',str(seed)],args.output/f'round_{round_index}_pocketgen.log')
            counts['pocketgen_calls']+=1
            for i in range(args.pocket_proposals):
                counts['pocket_proposals']+=1
                candidate=generated/f'{i}_whole_relaxed.pdb'
                candidate_ligand=generated/f'{i}.sdf'
                record={'round':round_index,'phase':'pocket','protein':str(candidate),
                        'ligand':str(candidate_ligand),'accepted':False}
                try:
                    m,u=evaluate(candidate,candidate_ligand,f'r{round_index}_pocket_score_{i}')
                    record.update(metrics=m,utility=u)
                    if u is not None and u>utility:
                        protein,ligand,utility=candidate,candidate_ligand,u
                        record['accepted']=True
                except Exception as exc:
                    record['error']=str(exc)
                history.append(record)
                save()
    status.update(stage='complete',final_protein=str(protein),final_ligand=str(ligand),
                  final_utility=utility,utility_improvement=utility-initial_utility,
                  counts=counts,seconds=time.monotonic()-start)
    save()
except Exception as exc:
    status.update(stage='failed',error=str(exc),counts=counts,seconds=time.monotonic()-start)
    save()
    raise
