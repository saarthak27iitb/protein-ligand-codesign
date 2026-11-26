"""Read-only interim analysis of the two completed cooperative seeds.

Does not modify submitted code, checkpoints, source snapshots or job dependencies.
"""
import json,statistics,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
manifest=json.loads((ROOT/'data/full_v1/manifest.json').read_text())
targets={r['id']:r for r in manifest['splits']['test']}
from rdkit import Chem
reference_smiles={}
for key,target in targets.items():
 mol=Chem.SDMolSupplier(target['ligand'],removeHs=True)[0]
 reference_smiles[key]=Chem.MolToSmiles(Chem.RemoveHs(mol))
methods=['learned','random_joint','greedy_alternating','ligand_only']
rows=[];seed_summary={};datasets={};matched={}
for seed in [42,123]:
 rr=json.loads((ROOT/f'runs/full_v1/cooperative/seed_{seed}/evaluation/test/results.json').read_text());datasets[seed]={}
 limits={r['target']:min(x['optimization_queries'] for x in rr if x['target']==r['target']) for r in rr}
 seed_summary[seed]={}
 for r in rr:
  datasets[seed].setdefault(r['method'],{})[r['target']]=r
  values=[r['initial_utility']]+[x['utility'] for x in r['observations'][:limits[r['target']]]]
  r['matched_utility']=max([float(v) for v in values if v is not None],default=-20.)
 for method in methods:
  r=list(datasets[seed][method].values());metrics=[x['final_docking']['metrics'] for x in r if x['final_docking']['ok']]
  chemical=sum(x['final_docking'].get('metrics',{}).get('smiles')!=reference_smiles[x['target']] for x in r)
  mutations=sum(x['final_docking'].get('metrics',{}).get('mutations',0)>0 for x in r)
  both=sum(x['final_docking'].get('metrics',{}).get('mutations',0)>0 and x['final_docking'].get('metrics',{}).get('smiles')!=reference_smiles[x['target']] for x in r)
  row={'targets':len(r),'evaluation_successes':sum(x['ok'] for x in r),'mean_utility':statistics.mean(x['best_utility'] for x in r),'matched_mean_utility':statistics.mean(x['matched_utility'] for x in r),'improved_over_reference_gt_0_01':sum(x['best_utility']>(x['initial_utility'] if x['initial_utility'] is not None else -20.)+.01 for x in r),'final_feasible':sum(x['final_docking']['feasible'] for x in r),'mean_vina_dock':statistics.mean(x['vina_score_docking'] for x in metrics),'mean_queries':statistics.mean(x['optimization_queries'] for x in r),'zero_query_targets':sum(x['optimization_queries']==0 for x in r),'changed_ligand_chemistry':chemical,'changed_protein_sequence':mutations,'changed_both_chemistries':both,'observed_feasible_proposals':sum(o['feasible'] for x in r for o in x['observations']),'observed_proposals':sum(len(x['observations']) for x in r)}
  seed_summary[seed][method]=row
  rows.append(dict(seed=seed,method=method,**row))
combined={m:{k:statistics.mean(seed_summary[s][m][k] for s in [42,123]) for k in seed_summary[42][m]} for m in methods}

def ci(values):
 rng=np.random.default_rng(307);x=np.array(values,float);boots=[rng.choice(x,len(x),replace=True).mean() for _ in range(5000)]
 return {'mean_difference':float(x.mean()),'ci95':[float(v) for v in np.quantile(boots,[.025,.975])],'protein_clusters':len(x)}
comparisons={}
for baseline in methods[1:]:
 by=collections.defaultdict(list);primary=collections.defaultdict(list);pairwise=collections.defaultdict(list)
 for tid,target in targets.items():
  delta=[];mdelta=[];pdelta=[]
  for seed in [42,123]:
   a=datasets[seed]['learned'][tid];b=datasets[seed][baseline][tid]
   delta.append(a['best_utility']-b['best_utility']);mdelta.append(a['matched_utility']-b['matched_utility'])
   budget=min(a['optimization_queries'],b['optimization_queries'])
   def best(row):
    values=[row['initial_utility']]+[o['utility'] for o in row['observations'][:budget]]
    return max([v for v in values if v is not None],default=-20.)
   pdelta.append(best(a)-best(b))
  by[target['sequence_sha256']].append(statistics.mean(delta));primary[target['sequence_sha256']].append(statistics.mean(mdelta));pairwise[target['sequence_sha256']].append(statistics.mean(pdelta))
 comparisons[baseline]={'full_cap_exploratory':ci([statistics.mean(v) for v in by.values()]),'prespecified_all_methods_equal_queries':ci([statistics.mean(v) for v in primary.values()]),'pairwise_equal_queries_exploratory':ci([statistics.mean(v) for v in pairwise.values()])}
result={'scope':'interim, cooperative seeds 42 and 123 only; third seed and competitive evaluation excluded','per_seed':seed_summary,'combined_means':combined,'learned_minus_baseline':comparisons,'limitations':['Two of three policy seeds; third seed incomplete at initial status check.','Utility is a computational optimization proxy, not measured affinity.','Ligand-only has zero queries for 48/64 targets, making the common all-method query budget zero on those targets.','The same 64 targets recur across policy seeds; bootstrap resamples protein-sequence clusters after averaging seed results.','Generation validity is not feasibility or improved binding.','Docking uses the same Vina scoring family; protein foldability is not established.']}
out=ROOT/'reports/full_v1/cooperative_completed_seeds_42_123.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'combined_means':combined,'comparisons':comparisons},indent=2))
