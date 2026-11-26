"""REINFORCE pilot for selecting retained ligand cores with frozen FLOWR proposals.

Single-pair pilot: fitting this controller is not evidence of target generalization.
The protected core, constraints, proposal budget and oracle are shared with random
evaluation. Failed calls consume a proposal action and incur a cost.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from design_utils import ROOT, run_model, proxy_utility, mol_features, retained_cores
from score_pair import score

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(22,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,state,action_features):
        return self.net(torch.cat([state.expand(len(action_features),-1),action_features],dim=-1)).squeeze(-1)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protein',type=Path,required=True)
    p.add_argument('--ligand',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=4)
    p.add_argument('--horizon',type=int,default=2)
    p.add_argument('--samples',type=int,default=4)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if min(args.episodes,args.horizon,args.samples,args.steps)<1:
        p.error('budgets must be positive')
    args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(args.seed); rng=np.random.default_rng(args.seed)
    reference=next(iter(Chem.SDMolSupplier(str(args.ligand),removeHs=True)))
    if reference is None:
        raise ValueError('Invalid starting ligand')
    protected=MurckoScaffold.GetScaffoldForMol(reference)
    if protected.GetNumHeavyAtoms()==0:
        raise ValueError('This pilot requires a ring-containing starting ligand')
    protected_smiles=Chem.MolToSmiles(protected)
    reference_atoms=reference.GetNumHeavyAtoms()
    initial_metrics=score(args.protein,args.ligand,args.output/'initial_score',args.seed)
    initial_u=proxy_utility(initial_metrics,reference_atoms)
    if initial_u is None:
        raise ValueError('Reference fails feasibility constraints')
    actor=Policy(); optimizer=torch.optim.Adam(actor.parameters(),lr=3e-4)
    baseline=0.; trace=[]; start=time.monotonic(); proposal_calls=0; score_calls=1; scored_candidates=0
    status={'stage':'training','algorithm':'REINFORCE retained-core controller',
            'protected_core':protected_smiles,'initial_utility':initial_u,
            'interpretation':'single-pair pipeline pilot, no unseen-target or experimental claim'}
    def save():
        (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        (args.output/'episodes.json').write_text(json.dumps(trace,indent=2)+'\n')
    save()
    try:
        for episode in range(args.episodes+2):
            training=episode<args.episodes
            label=f'train_{episode}' if training else ('eval_policy' if episode==args.episodes else 'eval_random')
            folder=args.output/label; folder.mkdir()
            current=reference; ligand=args.ligand.resolve(); u=initial_u
            logs=[]; log_probs=[]; entropies=[]; rewards=[]
            for step in range(args.horizon):
                cores=retained_cores(current,protected_smiles,max_actions=7)
                # Action zero is stop. Remaining actions specify a retained core.
                feats=[np.r_[np.zeros(9,dtype=np.float32),1.]]
                feats.extend(np.r_[mol_features(Chem.MolFromSmiles(c)),0.] for c in cores)
                af=torch.tensor(np.array(feats),dtype=torch.float32)
                state=torch.tensor(np.r_[mol_features(current),u/15,(args.horizon-step)/args.horizon,1.],dtype=torch.float32).unsqueeze(0)
                logits=actor(state,af)
                # Require one proposal in the integration pilot; stopping is learned thereafter.
                if step==0:
                    logits=torch.cat([logits.new_tensor([-1e9]),logits[1:]])
                dist=torch.distributions.Categorical(logits=logits)
                if training:
                    action=dist.sample()
                    log_probs.append(dist.log_prob(action)); entropies.append(dist.entropy())
                    a=int(action)
                elif label=='eval_random':
                    a=int(rng.integers(1 if step==0 else 0,len(feats)))
                else:
                    a=int(dist.logits.argmax())
                record={'step':step,'action':a,'cores':cores,'utility_before':u}
                if a==0:
                    rewards.append(0.); record.update(stop=True,reward=0.)
                    logs.append(record); break
                reward=-0.02; generated=folder/f'step_{step}'
                proposal_calls+=1
                try:
                    run_model('flowr','flowr_pair.py',[
                        '--protein',args.protein.resolve(),'--ligand',ligand,'--output',generated,
                        '--mode','substructure','--substructure',cores[a-1],
                        '--samples',args.samples,'--steps',args.steps,
                        '--seed',args.seed+1000*episode+step],folder/f'step_{step}.log')
                    candidates=json.loads((generated/'candidates.json').read_text())
                    scores=[]; before=u
                    for candidate in candidates:
                        row={'proposal':candidate}
                        if candidate['valid']:
                            try:
                                score_calls+=1
                                m=score(args.protein,candidate['path'],folder/f'step_{step}_score_{candidate["index"]}',args.seed)
                                scored_candidates+=1
                                value=proxy_utility(m,reference_atoms)
                                row.update(metrics=m,utility=value)
                                if value is not None and value>u:
                                    u=value; ligand=Path(candidate['path'])
                                    current=next(iter(Chem.SDMolSupplier(str(ligand),removeHs=True)))
                            except Exception as exc:
                                row['error']=str(exc)
                        scores.append(row)
                    record['candidates']=scores; reward+=u-before
                except Exception as exc:
                    record['error']=str(exc); reward-=0.1
                rewards.append(reward)
                record.update(reward=reward,utility_after=u); logs.append(record)
            total=float(sum(rewards))
            if training:
                returns=[]; running=0.
                for r in reversed(rewards):
                    running+=r; returns.insert(0,running)
                loss=sum(-lp*(ret-baseline)-0.01*ent for lp,ret,ent in zip(log_probs,returns,entropies))
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(actor.parameters(),1.); optimizer.step()
                baseline=0.9*baseline+0.1*total
            trace.append({'episode':label,'return':total,'final_utility':u,'steps':logs,
                          'final_ligand':str(ligand),'training':training})
            torch.save({'policy':actor.state_dict(),'optimizer':optimizer.state_dict(),
                        'episode':episode,'protected_core':protected_smiles},args.output/'controller.pt')
            status.update(completed_episodes=episode+1,proposal_calls=proposal_calls,score_calls=score_calls,
                          successfully_scored_candidates=scored_candidates,
                          seconds=time.monotonic()-start)
            save()
        if scored_candidates==0:
            raise RuntimeError('All generated candidate evaluations failed; controller pipeline not validated')
        status.update(stage='complete'); save()
    except Exception as exc:
        status.update(stage='failed',error=str(exc)); save(); raise

if __name__=='__main__':
    main()
