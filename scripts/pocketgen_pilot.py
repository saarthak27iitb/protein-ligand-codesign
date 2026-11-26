"""PocketGen example inference using upstream preprocessing and full-scaffold output.

Adapted from the MIT-licensed upstream generate_new.py. Docking is a separate stage.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--example', default='2p16', choices=['2p16', 'PiB', 'luxsit'])
parser.add_argument('--protein', type=Path)
parser.add_argument('--ligand', type=Path)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--samples', default=2, type=int)
parser.add_argument('--prepare-only', action='store_true')
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
if bool(args.protein) != bool(args.ligand):
    parser.error('--protein and --ligand must be supplied together')
if args.samples < 1:
    parser.error('--samples must be positive')
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
status = {'stage': 'initializing', 'model': 'PocketGen (approved PocketFlow substitute)',
          'purpose': 'upstream example inference pilot, not an unseen-target benchmark',
          'example': args.example, 'seed': args.seed}
status_file = args.output/'status.json'
def save():
    status_file.write_text(json.dumps(status, indent=2)+'\n')
save()
start = time.monotonic()
try:
    import torch
    import esm
    from torch_geometric.transforms import Compose
    from utils.protein_ligand import PDBProtein, parse_sdf_file
    from utils.data import torchify_dict, collate_mols_block
    from utils.transforms import FeaturizeProteinAtom, FeaturizeLigandAtom
    from utils.misc import load_config, seed_all
    from models.PD import Pocket_Design_new
    if not args.prepare_only and not torch.cuda.is_available():
        raise RuntimeError('Run through Slurm on a GPU node')
    seed_all(args.seed)
    for suffix, supplied in [('.pdb', args.protein), ('_ligand.sdf', args.ligand)]:
        source = supplied or ROOT/'vendor'/'PocketGen'/'examples'/args.example/(args.example+suffix)
        shutil.copy2(source, args.output/(args.example+suffix))
    if args.protein:
        status.update(protein=str(args.protein.resolve()), ligand=str(args.ligand.resolve()),
                      purpose='user-specified pair inference; benchmark split not established')
    protein_path = args.output/(args.example+'.pdb')
    ligand_path = args.output/(args.example+'_ligand.sdf')
    pocket_path = args.output/(args.example+'_pocket.pdb')
    protein = PDBProtein(protein_path.read_text())
    seq = ''.join(protein.to_dict_residue()['seq'])
    ligand = parse_sdf_file(str(ligand_path), feat=False)
    r10_idx, r10_residues = protein.query_residues_ligand(ligand, radius=10, selected_residue=None, return_mask=False)
    full_seq_idx, _ = protein.query_residues_ligand(ligand, radius=3.5, selected_residue=r10_residues, return_mask=False)
    pocket_path.write_text(protein.residues_to_pdb_block(r10_residues))
    pocket = PDBProtein(pocket_path.read_text())
    residue = pocket.to_dict_residue()
    _, residue['protein_edit_residue'] = pocket.query_residues_ligand(ligand)
    assert residue['protein_edit_residue'].sum() == len(full_seq_idx) > 0
    assert len(residue['protein_edit_residue']) == len(r10_idx)
    data = {'protein_'+k:v for k,v in torchify_dict(pocket.to_dict_atom()).items()}
    data.update({'ligand_'+k:v for k,v in torchify_dict(ligand).items()})
    data.update(torchify_dict(residue))
    data.update(seq=seq, full_seq_idx=torch.tensor(sorted(full_seq_idx)), r10_idx=torch.tensor(sorted(r10_idx)),
                protein_filename=str(pocket_path), ligand_filename=str(ligand_path), whole_protein_name=str(protein_path))
    pf, lf = FeaturizeProteinAtom(), FeaturizeLigandAtom()
    data = Compose([pf, lf])(data)
    converter = esm.data.Alphabet.from_architecture('ESM-1b').get_batch_converter()
    if args.prepare_only:
        batch = collate_mols_block([data], batch_converter=converter)
        status.update(stage='preprocessing_verified', pocket_residues=len(data['amino_acid']),
                      edited_residues=len(full_seq_idx), sequence_length=len(seq),
                      batch_keys=sorted(batch), seconds=time.monotonic()-start)
        save()
        raise SystemExit(0)
    config = load_config(str(ROOT/'vendor'/'PocketGen'/'configs'/'train_model.yml'))
    model = Pocket_Design_new(config.model, protein_atom_feature_dim=pf.feature_dim,
                             ligand_atom_feature_dim=lf.feature_dim, device='cuda:0').to('cuda:0')
    checkpoint = torch.load(ROOT/'checkpoints'/'pocketgen.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model.eval()
    status.update(stage='sampling', torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    save()
    metrics = []
    import copy
    for i in range(args.samples):
        batch = collate_mols_block([copy.deepcopy(data)], batch_converter=converter)
        batch = {k:(v.to('cuda:0') if torch.is_tensor(v) else v) for k,v in batch.items()}
        with torch.no_grad():
            aar, upstream_rmsd, attention = model.generate(batch, str(args.output))
        # The upstream rmsd expression is not standard coordinate RMSD; label explicitly.
        metrics.append({'sample': i, 'amino_acid_recovery': float(aar),
                        'upstream_nonstandard_rmsd_statistic': float(upstream_rmsd)})
    status.update(stage='complete', seconds=time.monotonic()-start, samples=metrics,
                  interpretation='Inference smoke check; binding and protein stability not evaluated')
    save()
except Exception as exc:
    status.update(stage='failed', error=str(exc), seconds=time.monotonic()-start)
    save()
    raise
