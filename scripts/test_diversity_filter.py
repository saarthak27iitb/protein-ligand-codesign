"""Regression checks for the invalid proposal that crashed job 454949."""
import unittest
import warnings
from rdkit import Chem
from flowr.gen.generate_from_pdb import filter_diverse_ligands_bulk

class DiversityChecks(unittest.TestCase):
    def test_invalids_are_removed_and_duplicates_filtered(self):
        valid=Chem.MolFromSmiles('CCO')
        bad=Chem.MolFromSmiles('[CH5]',sanitize=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result,pockets=filter_diverse_ligands_bulk(
                [None,valid,bad,Chem.Mol(valid),Chem.MolFromSmiles('c1ccccc1')],
                ['missing','ethanol','bad','duplicate','benzene'])
        self.assertEqual(pockets,['ethanol','benzene'])
        self.assertEqual([Chem.MolToSmiles(m) for m in result],['CCO','c1ccccc1'])

    def test_empty_and_all_invalid_preserve_return_contract(self):
        self.assertEqual(filter_diverse_ligands_bulk([]),[])
        self.assertEqual(filter_diverse_ligands_bulk([],[]),([],[]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertEqual(filter_diverse_ligands_bulk([None,Chem.Mol()]),[])
            self.assertEqual(filter_diverse_ligands_bulk([None],['pocket']),([],[]))

    def test_pocket_alignment_errors_are_not_silently_hidden(self):
        with self.assertRaises(ValueError):
            filter_diverse_ligands_bulk([Chem.MolFromSmiles('CCO')],[])

    def test_valid_input_is_not_mutated(self):
        mol=Chem.MolFromSmiles('c1ccccc1',sanitize=False)
        before=mol.ToBinary()
        result=filter_diverse_ligands_bulk([mol])
        self.assertEqual(before,mol.ToBinary())
        self.assertEqual(len(result),1)

if __name__=='__main__':
    unittest.main()
