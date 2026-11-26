"""Build a real-model finite candidate bank for downstream controller pilots."""
import argparse
import json
import time
from pathlib import Path
from rdkit import Chem
from design_utils import ROOT,run_model,proxy_utility,mol_features
from score_pair import score
from make_conformations import make_conformations
from mutation_lattice import build_lattice

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seed',default=42,type=int)
    p.add_argument('--ligands',default=12,type=int)
    p.add_argument('--pockets',default=2,type=int)
    args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    protein=ROOT/'vendor/PocketGen/examples/2p16/2p16.pdb'
    ligand=ROOT/'vendor/PocketGen/examples/2p16/2p16_ligand.sdf'
    status={'stage':'generating','seed':args.seed,'purpose':'single-example finite-pool integration pilot'}
    def save():
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save(); start=time.monotonic()
    try:
        run_model('flowr','flowr_pair.py',['--protein',protein,'--ligand',ligand,'--output',out/'ligands',
                  '--samples',args.ligands,'--seed',args.seed],out/'flowr.log')
        candidates=[{'path':str(ligand),'source':'reference','valid':True}]
        candidates.extend(dict(x,source='FLOWR') for x in json.loads((out/'ligands/candidates.json').read_text()) if x['valid'])
        if len(candidates)<5:
            raise RuntimeError('Need at least four valid generated ligands for train/evaluation pool splitting')
        run_model('pocketgen','pocketgen_pilot.py',['--protein',protein,'--ligand',ligand,'--output',out/'pockets',
                  '--samples',args.pockets,'--seed',args.seed],out/'pocketgen.log')
        pockets=[{'path':str(protein),'source':'reference','sequence_fixed':True}]
        pockets.extend({'path':str(out/'pockets'/f'{i}_whole_relaxed.pdb'),'source':'PocketGen','sequence_fixed':False}
                       for i in range(args.pockets))
        ensemble=make_conformations(protein,out/'ensemble',seed=args.seed,frames=3)
        conformations=[{'path':str(protein),'source':'reference','sequence_fixed':True}]
        conformations.extend({'path':x,'source':'restrained MD local perturbation','sequence_fixed':True} for x in ensemble['files'])
        bank={'stage':'scoring','seed':args.seed,'protein':str(protein),'reference_ligand':str(ligand),
              'ligands':candidates,'pockets':pockets,'conformations':conformations,
              'pocket_scores':[],'conformation_scores':[],
              'interpretation':'cached finite-pool pilot; upfront oracle costs are included separately; no unseen-target claim'}
        reference_atoms=next(iter(Chem.SDMolSupplier(str(ligand),removeHs=True))).GetNumHeavyAtoms()
        bank['reference_atoms']=reference_atoms
        for j,c in enumerate(candidates):
            mol=next(iter(Chem.SDMolSupplier(c['path'],removeHs=True)))
            c['features']=mol_features(mol).tolist()
        for family,structures in [('pocket',pockets),('conformation',conformations)]:
            for i,structure in enumerate(structures):
                for j,candidate in enumerate(candidates):
                    record={'protein_index':i,'ligand_index':j,'feasible':False,'utility':None}
                    try:
                        metrics=score(structure['path'],candidate['path'],out/f'{family}_scores/{i}_{j}',args.seed)
                        value=proxy_utility(metrics,reference_atoms)
                        record.update(metrics=metrics,utility=value,feasible=value is not None)
                    except Exception as exc:
                        record['error']=str(exc)
                    bank[family+'_scores'].append(record)
                    (out/'bank.partial.json').write_text(json.dumps(bank,indent=2)+'\n')
        if not any(r['protein_index']==0 and r['ligand_index']>0 and r['feasible'] for r in bank['pocket_scores']):
            raise RuntimeError('No generated ligand passed native-pocket feasibility; inspect the bank before controller fitting')
        bank['mutation_lattice']=build_lattice(protein,pockets[1]['path'],ligand,out/'mutation_lattice',
                                               reference_atoms,seed=args.seed)
        bank.update(stage='complete',seconds=time.monotonic()-start,
                    scoring_calls=len(bank['pocket_scores'])+len(bank['conformation_scores'])+len(bank['mutation_lattice']['states']))
        (out/'bank.json').write_text(json.dumps(bank,indent=2)+'\n')
        status.update(stage='complete',seconds=time.monotonic()-start,bank=str(out/'bank.json'))
        save()
    except Exception as exc:
        status.update(stage='failed',error=str(exc),seconds=time.monotonic()-start); save(); raise

if __name__=='__main__':
    main()
