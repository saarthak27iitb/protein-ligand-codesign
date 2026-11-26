"""Evaluate a protein/ligand pair with Vina and transparent geometry metrics.

Scores are computational proxies, not experimental affinity. Failed preparation
or unavailable force-field parameters are errors, never silently good rewards.
"""
import argparse
import hashlib
import json
import subprocess
import os
from pathlib import Path

if os.environ.get('SLURM_JOB_ID'):
    from capture_provenance import capture
    capture()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def geometry(protein, mol):
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem, QED, Descriptors
    from scipy.spatial.distance import cdist
    Chem.SanitizeMol(mol)
    heavy = Chem.RemoveHs(mol)
    if heavy.GetNumConformers() != 1 or not np.isfinite(heavy.GetConformer().GetPositions()).all():
        raise ValueError('Expected finite 3D coordinates')
    coords, radii = [], []
    table = Chem.GetPeriodicTable()
    for line in Path(protein).read_text().splitlines():
        if not line.startswith('ATOM  '):
            continue
        element = line[76:78].strip().title() or line[12:16].strip()[0].upper()
        if element == 'H':
            continue
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        radii.append(table.GetRvdw(element))
    if not coords:
        raise ValueError('Protein has no ATOM heavy atoms')
    distances = cdist(heavy.GetConformer().GetPositions(), coords)
    overlaps = np.array([table.GetRvdw(a.GetAtomicNum()) for a in heavy.GetAtoms()])[:,None] + np.array(radii)[None,:] - distances
    result = {'smiles': Chem.MolToSmiles(heavy), 'connected': len(Chem.GetMolFrags(heavy)) == 1,
              'heavy_atoms': heavy.GetNumHeavyAtoms(), 'molecular_weight': Descriptors.MolWt(heavy),
              'qed': QED.qed(heavy), 'clashing_atom_pairs_0_4A': int((overlaps > 0.4).sum()),
              'minimum_heavy_atom_distance_A': float(distances.min())}
    mh = Chem.AddHs(heavy, addCoords=True)
    if AllChem.MMFFHasAllMoleculeParams(mh):
        props = AllChem.MMFFGetMoleculeProperties(mh, mmffVariant='MMFF94s')
        ff = AllChem.MMFFGetMoleculeForceField(mh, props)
        energy_before = ff.CalcEnergy()
        convergence = ff.Minimize(maxIts=500)
        result.update(mmff94s_strain_kcal_mol=float(energy_before-ff.CalcEnergy()),
                      mmff_minimization_converged=convergence == 0)
    else:
        result.update(mmff94s_strain_kcal_mol=None, mmff_minimization_converged=False)
    return result

def score(protein, ligand, output, seed=42, dock=False):
    import numpy as np
    from rdkit import Chem
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from vina import Vina
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protein, ligand = Path(protein).resolve(), Path(ligand).resolve()
    mols = list(Chem.SDMolSupplier(str(ligand), removeHs=False))
    if len(mols) != 1 or mols[0] is None:
        raise ValueError('Supply exactly one valid ligand per scoring call')
    mol = mols[0]
    result = geometry(protein, mol)
    if not result['connected']:
        raise ValueError('Disconnected ligand')
    mh = Chem.AddHs(Chem.RemoveHs(mol), addCoords=True)
    setups = MoleculePreparation().prepare(mh)
    if len(setups) != 1:
        raise ValueError('Ambiguous ligand preparation')
    pdbqt, ok, error = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(error)
    ligand_qt = output/'ligand.pdbqt'
    ligand_qt.write_text(pdbqt)
    receptor_qt = output/'receptor.pdbqt'
    clean = output/'protein.pdb'
    lines = [line for line in protein.read_text().splitlines() if line.startswith(('ATOM  ', 'TER'))]
    clean.write_text('\n'.join(lines)+'\nEND\n')
    command = ['obabel', str(clean), '-O', str(receptor_qt), '-xr', '-h', '--partialcharge', 'gasteiger']
    with (output/'preparation.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    if not receptor_qt.exists() or not receptor_qt.stat().st_size:
        raise RuntimeError('Empty receptor preparation')
    v = Vina(sf_name='vina', cpu=1, seed=seed, verbosity=0)
    v.set_receptor(str(receptor_qt))
    v.set_ligand_from_file(str(ligand_qt))
    xyz = Chem.RemoveHs(mol).GetConformer().GetPositions()
    v.compute_vina_maps(center=xyz.mean(axis=0).tolist(), box_size=np.maximum(np.ptp(xyz,axis=0)+12,20).tolist())
    result['vina_score_input_pose'] = float(v.score()[0])
    result['vina_score_local_optimization'] = float(v.optimize()[0])
    if dock:
        v.dock(exhaustiveness=8, n_poses=5)
        result['vina_score_docking'] = float(v.energies(n_poses=1)[0,0])
    v.write_pose(str(output/'optimized.pdbqt'), overwrite=True)
    result.update(protein=str(protein), ligand=str(ligand), protein_sha256=digest(protein), ligand_sha256=digest(ligand),
                  seed=seed, receptor_preparation='OpenBabel Gasteiger charges; rigid receptor; water and HETATM excluded',
                  interpretation='computational proxy; no measured binding affinity')
    (output/'scores.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein', required=True, type=Path)
    parser.add_argument('--ligand', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dock', action='store_true')
    args = parser.parse_args()
    print(json.dumps(score(args.protein,args.ligand,args.output,args.seed,args.dock),indent=2))
