"""Controller/mathematical integration pilots over a bank of real model proposals.

These runs do not fine-tune FLOWR/PocketGen. They isolate controller behavior
before expensive online training. Cached oracle construction is reported, not free.
"""
import argparse
import json
import time
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn

def payoff(bank,key,n_rows):
    a=np.full((n_rows,len(bank['ligands'])),-20.,dtype=np.float32)
    for row in bank[key]:
        if row['utility'] is not None:
            a[row['protein_index'],row['ligand_index']]=row['utility']
    return a

def adversarial(bank,output,seed):
    """Finite zero-sum policy optimization; hold out the final MD conformation."""
    matrix=payoff(bank,'conformation_scores',len(bank['conformations']))
    if len(matrix)<3:
        raise ValueError('Need training and held-out conformations')
    train=torch.tensor(matrix[:-1])
    x=nn.Parameter(torch.zeros(train.shape[1])); y=nn.Parameter(torch.zeros(train.shape[0]))
    ax=torch.optim.Adam([x],lr=0.02); ay=torch.optim.Adam([y],lr=0.02)
    avgp=torch.zeros_like(x); avgq=torch.zeros_like(y); history=[]
    for step in range(2000):
        p=x.softmax(0); q=y.softmax(0)
        value=q@train@p
        # Full-information expected policy gradients; no generator weight updates.
        gx,gy=torch.autograd.grad(value,[x,y])
        ax.zero_grad(); ay.zero_grad(); x.grad=-gx; y.grad=gy; ax.step(); ay.step()
        if step>=1000:
            avgp+=p.detach()/1000; avgq+=q.detach()/1000
        if step%100==0:
            history.append({'step':step,'value':float(value.detach())})
    p=avgp.numpy(); q=avgq.numpy(); a=matrix[:-1]
    lower=float((a@p).min()); upper=float((q@a).max())
    result={'algorithm':'finite zero-sum full-information policy optimization',
            'ligand_mixture':p.tolist(),'conformation_mixture':q.tolist(),
            'train_lower_value':lower,'train_upper_value':upper,'exploitability_gap':upper-lower,
            'heldout_conformation_value':float(matrix[-1]@p),
            'uniform_heldout_value':float(matrix[-1].mean()),'reference_heldout_value':float(matrix[-1,0]),
            'feasible_ligand_counts_per_conformation':(matrix>-20).sum(axis=1).tolist(),
            'history':history,'generator_training':False,
            'limitation':'One target and short restrained MD states; not proof of biological robustness'}
    torch.save({'ligand_logits':x.detach(),'conformation_logits':y.detach()},output/'agents.pt')
    return result

AA=['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
def pocket_features(path):
    residues={}
    for line in Path(path).read_text().splitlines():
        if line.startswith('ATOM  '):
            residues[(line[21],line[22:27])]=line[17:20]
    counts=np.array([sum(v==aa for v in residues.values()) for aa in AA],dtype=np.float32)
    return counts/max(counts.sum(),1)

def joint(bank,output,seed):
    """Learn a finite-pool energy and explicitly normalize a common joint law."""
    matrix=payoff(bank,'pocket_scores',len(bank['pockets']))
    rng=np.random.default_rng(seed)
    pf=[pocket_features(x['path']) for x in bank['pockets']]
    features=torch.tensor(np.array([np.r_[pf[i],bank['ligands'][j]['features']] for i in range(len(pf)) for j in range(matrix.shape[1])]),dtype=torch.float32)
    targets=torch.tensor(matrix.reshape(-1)/20)
    order=rng.permutation(len(targets)); n=max(1,int(0.75*len(order)))
    train=torch.tensor(order[:n]); test=torch.tensor(order[n:])
    if len(test)==0:
        raise ValueError('No held-out candidate pairs')
    model=nn.Sequential(nn.Linear(29,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    for _ in range(1000):
        prediction=model(features[train]).squeeze(-1)
        loss=(prediction-targets[train]).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        energies=model(features).squeeze(-1).reshape(matrix.shape)*20
        probabilities=energies.flatten().softmax(0).reshape(matrix.shape)
        pocket_marginal=probabilities.sum(1); ligand_marginal=probabilities.sum(0)
        ligand_given_pocket=probabilities/pocket_marginal[:,None]
        pocket_given_ligand=probabilities/ligand_marginal[None,:]
        residual=(ligand_given_pocket*pocket_marginal[:,None]-pocket_given_ligand*ligand_marginal[None,:]).abs().max()
        pred=model(features[test]).squeeze(-1)*20
        rmse=float((pred-targets[test]*20).square().mean().sqrt())
    torch.save(model.state_dict(),output/'compatibility.pt')
    return {'algorithm':'finite-pool learned compatibility energy with normalized joint distribution',
            'joint':probabilities.tolist(),'ligand_given_pocket':ligand_given_pocket.tolist(),
            'pocket_given_ligand':pocket_given_ligand.tolist(),'bayes_consistency_max_residual':float(residual),
            'heldout_pair_proxy_rmse':rmse,'train_pair_indices':train.tolist(),'test_pair_indices':test.tolist(),
            'generator_training':False,
            'limitation':'Normalization consistency follows by construction; this does not establish consistency or improvement of the original generative models'}

def allocation(bank,output,seed):
    """REINFORCE acquisition controller; disjoint generated candidates for evaluation."""
    values=payoff(bank,'pocket_scores',len(bank['pockets']))[0,1:]
    features=np.array([x['features'] for x in bank['ligands'][1:]],dtype=np.float32)
    split=len(values)//2
    if split<2 or len(values)-split<2:
        raise ValueError('Need at least two generated candidates in each split')
    model=nn.Sequential(nn.Linear(11,32),nn.Tanh(),nn.Linear(32,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=0.003)
    rng=np.random.default_rng(seed); baseline=0.; history=[]
    budget=min(3,split,len(values)-split)
    def rollout(indices,training=False,random=False):
        remaining=list(indices); best=-20.; logs=[]; selections=[]
        for step in range(budget):
            x=torch.tensor(np.column_stack([features[remaining],np.full(len(remaining),best/20),
                                            np.full(len(remaining),(budget-step)/budget)]),dtype=torch.float32)
            distribution=torch.distributions.Categorical(logits=model(x).squeeze(-1))
            if training:
                action=distribution.sample(); logs.append(distribution.log_prob(action)); a=int(action)
            elif random:
                a=int(rng.integers(len(remaining)))
            else:
                a=int(distribution.logits.argmax())
            index=remaining.pop(a); selections.append(index+1); best=max(best,float(values[index]))
        return best,logs,selections
    for episode in range(300):
        reward,logs,selection=rollout(range(split),training=True)
        loss=-torch.stack(logs).sum()*(reward-baseline)
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
        baseline=0.9*baseline+0.1*reward
        if episode%25==0:
            history.append({'episode':episode,'best_utility':reward})
    policy,_,selected=rollout(range(split,len(values)))
    random_rewards=[rollout(range(split,len(values)),random=True)[0] for _ in range(100)]
    # Linear prediction is a supervised acquisition baseline on the same fully cached training pool.
    x=np.column_stack([np.ones(split),features[:split]])
    beta=np.linalg.solve(x.T@x+0.1*np.eye(x.shape[1]),x.T@values[:split])
    estimates=np.column_stack([np.ones(len(values)-split),features[split:]])@beta
    ridge_indices=np.argsort(estimates)[-budget:]+split
    torch.save(model.state_dict(),output/'allocation_policy.pt')
    return {'algorithm':'cached-oracle REINFORCE acquisition pilot','budget':budget,
            'policy_best_utility':policy,'policy_selected_ligand_indices':selected,
            'random_mean_best_utility':float(np.mean(random_rewards)),
            'random_std_best_utility':float(np.std(random_rewards)),
            'ridge_best_utility':float(values[ridge_indices].max()),
            'oracle_best_utility':float(values[split:].max()),
            'training_generated_ligand_indices':list(range(1,split+1)),
            'evaluation_generated_ligand_indices':list(range(split+1,len(values)+1)),
            'history':history,'online_oracle_calls':0,
            'limitation':'Offline simulator over one precomputed bank; does not yet learn allocation between evaluation fidelities or demonstrate online savings'}

def mutation(bank,output,seed):
    """Sequential policy over a real, exhaustively scored small mutation lattice."""
    lattice=bank['mutation_lattice']; n=lattice['max_mutations']
    values={x['mask']:(x['utility'] if x['utility'] is not None else -20.) for x in lattice['states']}
    if values[0]<=-20:
        raise ValueError('Reference mutation state infeasible')
    model=nn.Sequential(nn.Linear(n+1,32),nn.Tanh(),nn.Linear(32,n+1))
    optimizer=torch.optim.Adam(model.parameters(),lr=0.003)
    baseline=0.; rng=np.random.default_rng(seed); history=[]
    def rollout(training=False,random=False):
        mask=0; logs=[]; total=0.; path=[0]
        for step in range(n+1):
            state=torch.tensor([float(bool(mask&(1<<i))) for i in range(n)]+[values[mask]/20])
            logits=model(state)
            valid=torch.tensor([True]+[not bool(mask&(1<<i)) for i in range(n)])
            dist=torch.distributions.Categorical(logits=logits.masked_fill(~valid,-1e9))
            if training:
                action=dist.sample(); logs.append(dist.log_prob(action)); a=int(action)
            elif random:
                a=int(rng.choice(np.flatnonzero(valid.numpy())))
            else:
                a=int(dist.logits.argmax())
            if a==0:
                break
            new=mask|(1<<(a-1)); total+=values[new]-values[mask]-0.02
            mask=new; path.append(mask)
        return total,logs,path
    for episode in range(400):
        reward,logs,path=rollout(training=True)
        loss=-torch.stack(logs).sum()*(reward-baseline)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); baseline=0.9*baseline+0.1*reward
        if episode%50==0:
            history.append({'episode':episode,'return':reward,'path':path})
    reward,_,path=rollout()
    random_rewards=[rollout(random=True)[0] for _ in range(100)]
    optimum=max(values[m]-values[0]-0.02*int(m).bit_count() for m in values)
    torch.save(model.state_dict(),output/'mutation_policy.pt')
    return {'algorithm':'REINFORCE mutation-budget policy on a cached finite lattice','policy_return':reward,
            'policy_path':path,'final_mutation_count':int(path[-1]).bit_count(),'mutation_budget':n,
            'random_mean_return':float(np.mean(random_rewards)),'exact_lattice_optimal_return':optimum,
            'history':history,'online_oracle_calls':0,
            'limitation':'Policy trained and checked on the same tiny MDP; no held-out-target performance or foldability claim'}

def main():
    if os.environ.get('SLURM_JOB_ID'):
        from capture_provenance import capture
        capture()
    parser=argparse.ArgumentParser()
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--mode',choices=['adversarial','joint','allocation','mutation'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    start=time.monotonic(); status={'stage':'running','mode':args.mode,'seed':args.seed}
    file=args.output/'status.json'
    file.write_text(json.dumps(status,indent=2)+'\n')
    try:
        bank=json.loads(args.bank.read_text())
        if bank['stage']!='complete':
            raise ValueError('Incomplete bank')
        result=globals()[args.mode](bank,args.output,args.seed)
        result.update(seed=args.seed,bank=str(args.bank.resolve()),bank_construction_scoring_calls=bank['scoring_calls'])
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        status.update(stage='complete',seconds=time.monotonic()-start)
        file.write_text(json.dumps(status,indent=2)+'\n')
    except Exception as exc:
        status.update(stage='failed',error=str(exc),seconds=time.monotonic()-start)
        file.write_text(json.dumps(status,indent=2)+'\n'); raise

if __name__=='__main__':
    main()
