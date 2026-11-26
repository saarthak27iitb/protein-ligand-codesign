"""Submit prepared pilots once, respecting eight total active jobs.

Run again to submit deferred stages. Existing running/pending jobs are preserved.
All submissions require successful prerequisites through Slurm afterok dependencies.
"""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/'reports'/'campaign-jobs.json'
campaign=json.loads(FILE.read_text()) if FILE.exists() else {
    'purpose':'initial integration/controller pilots, not a completed research benchmark',
    'jobs':[{'label':'FLOWR validation inference','job_id':'454948','script':'scripts/flowr_pilot.slurm'},
            {'label':'FLOWR pair interface','job_id':'454949','script':'scripts/flowr_pair.slurm'}],
    'external_prerequisites':{'PocketGen_success':'454103','FLOWR_CPU_preflight_success':'454951',
                              'fixed_sequence_MD_preflight':'454952'}}
def save():
    FILE.write_text(json.dumps(campaign,indent=2)+'\n')
def job_id(label):
    return next(j['job_id'] for j in campaign['jobs'] if j['label']==label)
def active_count():
    uid=str(__import__('os').getuid())
    result=subprocess.check_output(['squeue','-h','-r','-u',uid,'-o','%i'],text=True)
    return len(result.splitlines())
def submit(label,script,dependencies,exports=None):
    if any(j['label']==label for j in campaign['jobs']):
        return True
    if active_count()>=8:
        campaign.setdefault('deferred',[])
        if label not in campaign['deferred']:
            campaign['deferred'].append(label)
        save(); print('DEFERRED:',label,flush=True); return False
    command=['sbatch','--parsable','--dependency=afterok:'+':'.join(dependencies)]
    if exports:
        command.append('--export=ALL,'+','.join(f'{k}={v}' for k,v in exports.items()))
    command.append(script)
    result=subprocess.check_output(command,cwd=ROOT,text=True).strip().split(';')[0]
    if not result.isdigit():
        raise RuntimeError(f'Unrecognized sbatch reply: {result}')
    entry={'label':label,'job_id':result,'script':script,'command':command,
           'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'script_sha256':hashlib.sha256((ROOT/script).read_bytes()).hexdigest()}
    campaign['jobs'].append(entry)
    campaign['deferred']=[x for x in campaign.get('deferred',[]) if x!=label]
    save(); print(json.dumps(entry),flush=True); return True

save()
pair=job_id('FLOWR pair interface')
submit('Frozen cooperative search','scripts/cooperative_search.slurm',[pair])
submit('RL fragment controller','scripts/rl_fragment.slurm',[pair])
if submit('Real-model candidate bank','scripts/build_candidate_bank.slurm',[pair,campaign['external_prerequisites']['fixed_sequence_MD_preflight']]):
    bank_job=job_id('Real-model candidate bank')
    bank=str(ROOT/'runs'/f'candidate-bank-{bank_job}'/'bank.json')
    submit('Finite conformation game and joint compatibility','scripts/finite_pool.slurm',[bank_job],
           {'BANK':bank,'EXPERIMENT_MODE':'game-and-joint'})
    submit('RL evaluation allocation','scripts/finite_pool.slurm',[bank_job],
           {'BANK':bank,'EXPERIMENT_MODE':'allocation'})
    submit('RL mutation budget','scripts/finite_pool.slurm',[bank_job],
           {'BANK':bank,'EXPERIMENT_MODE':'mutation'})
print('Active jobs:',active_count(),flush=True)
