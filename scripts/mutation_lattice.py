"""Small mutation-budget MDP from actual PocketGen-proposed substitutions.

Align proposed side chains to the native residue backbone. Keep the scaffold and
unselected residues fixed. This does not establish protein foldability.
"""
import itertools
import json
from pathlib import Path
import numpy as np
from score_pair import score
from design_utils import proxy_utility

def residues(path):
    result={}
    for line in Path(path).read_text().splitlines():
        if not line.startswith('ATOM  '):
            continue
        element=line[76:78].strip().upper() or line[12:16].strip()[0]
        if element=='H':
            continue
        key=(line[21],line[22:26],line[26])
        result.setdefault(key,[]).append(line)
    return result

def xyz(line):
    return np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])])

def substitute(native,proposed,selected,path):
    output=[]; backbone={'N','CA','C','O','OXT'}
    for key,original in native.items():
        if key not in selected:
            output.extend(original); continue
        new=proposed[key]; name=new[0][17:20]
        old_atoms={l[12:16].strip():l for l in original}
        new_atoms={l[12:16].strip():l for l in new}
        if not all(a in old_atoms and a in new_atoms for a in ['N','CA','C']):
            raise ValueError('Missing alignment backbone atoms')
        p=np.array([xyz(new_atoms[a]) for a in ['N','CA','C']])
        q=np.array([xyz(old_atoms[a]) for a in ['N','CA','C']])
        u,_,vt=np.linalg.svd((p-p.mean(0)).T@(q-q.mean(0)))
        if np.linalg.det(u@vt)<0:
            u[:,-1]*=-1
        rotation=u@vt
        for line in original:
            if line[12:16].strip() in backbone:
                output.append(line[:17]+name+line[20:])
        for line in new:
            if line[12:16].strip() in backbone:
                continue
            position=(xyz(line)-p.mean(0))@rotation+q.mean(0)
            output.append(line[:21]+key[0]+key[1]+key[2]+line[27:30]+
                          ''.join(f'{v:8.3f}' for v in position)+line[54:])
    Path(path).write_text('\n'.join(line[:6]+f'{i+1:5d}'+line[11:] for i,line in enumerate(output))+'\nEND\n')

def build_lattice(protein,proposed,ligand,output,reference_atoms,max_edits=3,seed=42):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    native,proposal=residues(protein),residues(proposed)
    if set(native)!=set(proposal):
        raise ValueError('PocketGen scaffold residue identifiers changed')
    keys=[k for k in native if native[k][0][17:20]!=proposal[k][0][17:20]][:max_edits]
    if not keys:
        raise ValueError('No proposed substitutions')
    states=[]
    for mask in range(2**len(keys)):
        selected={k for i,k in enumerate(keys) if mask&(1<<i)}
        path=output/f'state_{mask}.pdb'
        substitute(native,proposal,selected,path)
        resulting=residues(path)
        changed={k for k in native if native[k][0][17:20]!=resulting[k][0][17:20]}
        if changed!=selected:
            raise RuntimeError('Mutation budget violated')
        for k in native:
            if k not in selected and resulting[k]!=native[k]:
                # Only serial numbers are globally renumbered.
                if [x[:6]+x[11:] for x in resulting[k]] != [x[:6]+x[11:] for x in native[k]]:
                    raise RuntimeError('Unselected scaffold changed')
        row={'mask':mask,'mutations':len(selected),'path':str(path),'utility':None}
        try:
            metrics=score(path,ligand,output/f'score_{mask}',seed)
            row.update(metrics=metrics,utility=proxy_utility(metrics,reference_atoms))
        except Exception as exc:
            row['error']=str(exc)
        states.append(row)
    manifest={'states':states,'residue_keys':[list(k) for k in keys],
              'substitutions':[{'key':list(k),'from':native[k][0][17:20],'to':proposal[k][0][17:20]} for k in keys],
              'max_mutations':len(keys),'scaffold_fixed':True,
              'interpretation':'finite mutation-budget controller pilot; no protein stability/folding assessment'}
    (output/'lattice.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
