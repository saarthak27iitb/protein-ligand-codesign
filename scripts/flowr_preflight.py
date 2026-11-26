"""Load actual checkpoint and validate both data interfaces without GPU inference."""
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['FLOWR_NUM_WORKERS']='0'
import torch
from flowr.gen import generate_from_smol as smol
from flowr.gen import generate_from_pdb as pdb
from flowr.util.pocket import PocketComplexBatch
from flowr.data.datasets import GeometricDataset
report={}
out=ROOT/'runs'/('flowr-preflight-'+os.environ.get('SLURM_JOB_ID','local'))
out.mkdir(parents=True,exist_ok=False)
def save():
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
try:
    sys.argv=['preflight','--arch','pocket','--pocket_noise','fix',
              '--ckpt_path',str(ROOT/'checkpoints'/'flowr_noHs.ckpt'),
              '--data_path',str(out),'--save_dir',str(out),'--dataset','spindr',
              '--batch_cost','1','--gpus','1','--mp_index','1','--num_workers','0']
    args=smol.get_args()
    model,hparams,vocab,_,_=smol.load_model(args)
    report.update(checkpoint_loaded=True,model_parameters=sum(p.numel() for p in model.parameters()),
                  torch=torch.__version__,use_interactions=bool(hparams['flow_interactions'] or hparams['predict_interactions']))
    save()
    systems=PocketComplexBatch.from_bytes((ROOT/'data'/'spindr'/'val.smol').read_bytes(),remove_hs=hparams['remove_hs'])
    transform,interpolant=smol.load_util(args,hparams,vocab)
    ds=GeometricDataset(PocketComplexBatch([systems[0]]),data_cls=PocketComplexBatch,transform=transform)
    dl=smol.get_dataloader(ds,vocab,interpolant,args,hparams)
    batch=next(iter(dl))
    report.update(validation_targets=len(systems),validation_batch_ready=True)
    save()
    sys.argv=['preflight','--arch','pocket','--pocket_type','holo',
              '--pdb_file',str(ROOT/'vendor'/'PocketGen'/'examples'/'2p16'/'2p16.pdb'),
              '--ligand_file',str(ROOT/'vendor'/'PocketGen'/'examples'/'2p16'/'2p16_ligand.sdf'),
              '--data_path',str(out),'--save_dir',str(out),'--dataset','spindr',
              '--batch_cost','1','--gpus','1','--sample_n_molecules_per_target','1','--cut_pocket']
    pargs=pdb.get_args()
    data,transform,interpolant=pdb.load_data(pargs,hparams,vocab)
    ds=pdb.get_dataset(data,transform,vocab,interpolant,pargs,hparams)
    batch=next(iter(pdb.get_dataloader(ds,vocab,interpolant,pargs,hparams)))
    report.update(pair_batch_ready=True,stage='complete')
    save()
except Exception as exc:
    report.update(stage='failed',error=str(exc))
    save()
    raise
