"""Shared constraints and frozen-model subprocess interfaces for pilot controllers."""
import json
import os
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run_model(model,script,argv,log):
    env=dict(os.environ)
    env.update(PATH=str(ROOT/'envs'/model/'bin')+os.pathsep+env['PATH'],
               PYTHONPATH=str(ROOT/'vendor'/('flowr' if model=='flowr' else 'PocketGen')),
               TORCH_HOME=str(ROOT/'data'/'torch-cache'),FLOWR_NUM_WORKERS='0',WANDB_MODE='disabled')
    with Path(log).open('w') as out:
        subprocess.run([str(ROOT/'envs'/model/'bin'/'python'),str(ROOT/'scripts'/script)]+list(map(str,argv)),
                       cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)

def proxy_utility(metrics,reference_atoms):
    import math
    strain=metrics.get('mmff94s_strain_kcal_mol')
    vina=metrics.get('vina_score_input_pose')
    if (strain is None or vina is None or not math.isfinite(strain) or not math.isfinite(vina) or
        not metrics['connected'] or not metrics['mmff_minimization_converged'] or
        metrics['clashing_atom_pairs_0_4A']>5 or strain>100 or strain< -0.01 or
        not 0.8*reference_atoms<=metrics['heavy_atoms']<=1.2*reference_atoms):
        return None
    return -vina-0.01*strain

def mol_features(mol):
    import numpy as np
    from rdkit.Chem import QED, Descriptors, rdMolDescriptors
    return np.array([mol.GetNumHeavyAtoms()/50,Descriptors.MolWt(mol)/600,
                     Descriptors.MolLogP(mol)/6,rdMolDescriptors.CalcTPSA(mol)/150,
                     rdMolDescriptors.CalcNumRotatableBonds(mol)/15,
                     rdMolDescriptors.CalcNumHBD(mol)/5,rdMolDescriptors.CalcNumHBA(mol)/10,
                     rdMolDescriptors.CalcNumRings(mol)/6,QED.qed(mol)],dtype=np.float32)

def retained_cores(mol,protected_smiles,max_actions=8):
    """Return chemically valid cores containing the prescribed protected subgraph."""
    from rdkit import Chem
    protected=Chem.MolFromSmiles(protected_smiles)
    if protected is None or not mol.HasSubstructMatch(protected):
        raise ValueError('Protected core absent')
    cores={Chem.MolToSmiles(protected)}
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType()!=Chem.BondType.SINGLE:
            continue
        cut=Chem.FragmentOnBonds(mol,[bond.GetIdx()],addDummies=False)
        for component in Chem.GetMolFrags(cut,asMols=True,sanitizeFrags=True):
            n=component.GetNumHeavyAtoms()
            if protected.GetNumHeavyAtoms()<=n<mol.GetNumHeavyAtoms() and component.HasSubstructMatch(protected):
                cores.add(Chem.MolToSmiles(component))
    return sorted(cores,key=lambda s:(Chem.MolFromSmiles(s).GetNumHeavyAtoms(),s))[:max_actions]
