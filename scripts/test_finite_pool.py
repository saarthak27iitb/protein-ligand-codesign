"""Synthetic fixtures check algorithms only; they are not experiment results."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from design_utils import ROOT
from finite_pool_experiments import adversarial,joint,allocation,mutation
from mutation_lattice import residues,substitute

class FinitePoolChecks(unittest.TestCase):
    def test_matching_pennies_and_heldout_isolation(self):
        def bank(heldout):
            matrix=[[1.,-1.],[-1.,1.],heldout]
            return {'ligands':[{},{}],'conformations':[{}, {}, {}],
                    'conformation_scores':[{'protein_index':i,'ligand_index':j,'utility':v} for i,row in enumerate(matrix) for j,v in enumerate(row)]}
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(42)
            a=adversarial(bank([9.,-9.]),Path(tmp),42)
            torch.manual_seed(42)
            b=adversarial(bank([-100.,100.]),Path(tmp),42)
        np.testing.assert_allclose(a['ligand_mixture'],b['ligand_mixture'])
        self.assertLess(a['exploitability_gap'],0.05)
        self.assertAlmostEqual(sum(a['ligand_mixture']),1.,places=4)

    def test_joint_has_consistent_conditionals(self):
        native=ROOT/'vendor/PocketGen/examples/2p16/2p16.pdb'
        bank={'pockets':[{'path':str(native)}]*2,
              'ligands':[{'features':[float(j)/10]*9} for j in range(4)],
              'pocket_scores':[{'protein_index':i,'ligand_index':j,'utility':float(i+j)} for i in range(2) for j in range(4)]}
        with tempfile.TemporaryDirectory() as tmp:
            result=joint(bank,Path(tmp),42)
        self.assertLess(result['bayes_consistency_max_residual'],1e-6)
        self.assertAlmostEqual(np.array(result['joint']).sum(),1.,places=5)
        np.testing.assert_allclose(np.array(result['ligand_given_pocket']).sum(1),1.,atol=1e-6)
        np.testing.assert_allclose(np.array(result['pocket_given_ligand']).sum(0),1.,atol=1e-6)

    def test_mutations_preserve_unselected_real_scaffold(self):
        native=residues(ROOT/'vendor/PocketGen/examples/2p16/2p16.pdb')
        proposed=residues(ROOT/'runs/pocketgen-pilot-454103/0_whole_relaxed.pdb')
        self.assertEqual(set(native),set(proposed))
        edited=[k for k in native if native[k][0][17:20]!=proposed[k][0][17:20]]
        self.assertGreater(len(edited),0)
        selected={edited[0]}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'edited.pdb'; substitute(native,proposed,selected,path); result=residues(path)
        changed={k for k in native if native[k][0][17:20]!=result[k][0][17:20]}
        self.assertEqual(changed,selected)
        for k in native:
            if k not in selected:
                self.assertEqual([x[11:] for x in native[k]],[x[11:] for x in result[k]])

    def test_mutation_policy_respects_budget(self):
        bank={'mutation_lattice':{'max_mutations':2,'states':[{'mask':0,'utility':1.},
              {'mask':1,'utility':1.2},{'mask':2,'utility':1.4},{'mask':3,'utility':1.8}]}}
        with tempfile.TemporaryDirectory() as tmp:
            result=mutation(bank,Path(tmp),42)
        self.assertLessEqual(result['final_mutation_count'],2)
        self.assertLessEqual(result['policy_return'],result['exact_lattice_optimal_return']+1e-6)

    def test_allocation_evaluation_candidates_are_disjoint_and_budgeted(self):
        bank={'pockets':[{}],'ligands':[{'features':[float(j)/10]*9} for j in range(9)],
              'pocket_scores':[{'protein_index':0,'ligand_index':j,'utility':float(j)} for j in range(9)]}
        with tempfile.TemporaryDirectory() as tmp:
            result=allocation(bank,Path(tmp),42)
        train=set(result['training_generated_ligand_indices'])
        evaluation=set(result['evaluation_generated_ligand_indices'])
        selected=result['policy_selected_ligand_indices']
        self.assertFalse(train&evaluation)
        self.assertTrue(set(selected)<=evaluation)
        self.assertEqual(len(set(selected)),result['budget'])
        self.assertLessEqual(result['policy_best_utility'],result['oracle_best_utility'])

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
