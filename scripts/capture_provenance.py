"""Capture code and package versions at job start without environment secrets."""
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tarfile
from pathlib import Path
def capture():
    ROOT=Path(os.environ.get('DH307_ROOT',str(Path(__file__).resolve().parents[1])))
    job=os.environ['SLURM_JOB_ID']
    folder=ROOT/'runs'/f'provenance-{job}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'provenance.json').exists():
        return
    paths=list((ROOT/'scripts').glob('*.py'))+list((ROOT/'scripts').glob('*.slurm'))
    paths+=list((ROOT/'configs').glob('*'))
    for repo in ['flowr','PocketGen']:
        paths+=list((ROOT/'vendor'/repo).rglob('*.py'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    with tarfile.open(folder/'source.tar.gz','w:gz') as archive:
        for relative in sorted(hashes):
            archive.add(ROOT/relative,arcname=relative)
    packages={}
    for name in ['torch','numpy','scipy','rdkit','lightning','MDAnalysis','prolif','openmm','meeko','vina']:
        try:
            packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name]=None
    report={'job_id':job,'packages':packages,'code_sha256':hashes}
    if (ROOT/'reports/repository-commits.json').exists():
        report['upstream_commits']=json.loads((ROOT/'reports/repository-commits.json').read_text())
    (folder/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__ == "__main__":
    capture()
