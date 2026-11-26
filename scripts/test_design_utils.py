import math
import unittest
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from design_utils import ROOT,retained_cores,proxy_utility,mol_features

class DesignChecks(unittest.TestCase):
    def test_retained_cores_preserve_real_reference_scaffold(self):
        mol=next(iter(Chem.SDMolSupplier(str(ROOT/'vendor/PocketGen/examples/2p16/2p16_ligand.sdf'),removeHs=True)))
        core=MurckoScaffold.GetScaffoldForMol(mol)
        options=retained_cores(mol,Chem.MolToSmiles(core))
        self.assertGreater(len(options),0)
        for option in options:
            fragment=Chem.MolFromSmiles(option)
            self.assertTrue(fragment.HasSubstructMatch(core))
            self.assertTrue(mol.HasSubstructMatch(fragment))
            self.assertLess(fragment.GetNumHeavyAtoms(),mol.GetNumHeavyAtoms())

    def test_unrelated_core_rejected(self):
        with self.assertRaises(ValueError):
            retained_cores(Chem.MolFromSmiles('CCO'),'c1ccccc1')

    def test_unavailable_or_nonfinite_reward_cannot_win(self):
        valid={'mmff94s_strain_kcal_mol':10.,'vina_score_input_pose':-8.,'connected':True,
               'mmff_minimization_converged':True,'clashing_atom_pairs_0_4A':0,'heavy_atoms':30}
        self.assertAlmostEqual(proxy_utility(valid,30),7.9)
        for field,value in [('mmff94s_strain_kcal_mol',None),('vina_score_input_pose',float('nan')),
                            ('heavy_atoms',100),('clashing_atom_pairs_0_4A',100),
                            ('mmff_minimization_converged',False)]:
            trial=dict(valid); trial[field]=value
            self.assertIsNone(proxy_utility(trial,30))

    def test_controller_can_learn_with_variable_action_counts(self):
        import torch
        from rl_fragment import Policy
        actor=Policy()
        optimizer=torch.optim.Adam(actor.parameters(),lr=0.01)
        state=torch.zeros(1,12)
        before=[p.detach().clone() for p in actor.parameters()]
        for n in [2,7]:
            actions=torch.randn(n,10)
            distribution=torch.distributions.Categorical(logits=actor(state,actions))
            loss=-distribution.log_prob(torch.tensor(n-1))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,actor.parameters())))

if __name__=='__main__':
    unittest.main()
