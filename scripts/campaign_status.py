"""Read-only scheduler and artifact summary."""
import json
import os
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
campaign=json.loads((ROOT/'reports/campaign-jobs.json').read_text())
ids=[x['job_id'] for x in campaign['jobs']]
try:
    print(subprocess.check_output(['sacct','-X','-n','-j',','.join(ids),
          '--format=JobID,JobName,State,ExitCode,Elapsed,NodeList'],text=True))
except subprocess.CalledProcessError as exc:
    print('Scheduler unavailable:',exc)
for entry in campaign['jobs']:
    matches=list((ROOT/'runs').glob('*-'+entry['job_id']+'/status.json'))
    for path in matches:
        status=json.loads(path.read_text())
        print(entry['job_id'],entry['label'],status.get('stage'),status.get('error',''))
if campaign.get('deferred'):
    print('Deferred:',', '.join(campaign['deferred']))
