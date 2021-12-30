import unittest
import numpy as np
from pyrfume.odorants import all_smiles, smiles_to_mol
from pyrfume.features import tanimoto_sim, get_maximum_tanimoto_similarity


class TanimotoTestCase(unittest.TestCase):
    def setUp(self):
        self.n = 100
        smiles = all_smiles()[:self.n]  # First 100 smiles
        mols = smiles_to_mol(smiles)
        self.mols = list(mols.values())
        
    def test_tanimoto_sim(self):
        sim = tanimoto_sim(self.mols[0], self.mols[1])
        self.assertIsInstance(sim, float)
        self.assertGreaterEqual(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_bulk_tanimoto(self):
        # First 50 vs last 50
        m = int(self.n/2)
        sims = get_maximum_tanimoto_similarity(self.mols[:m], self.mols[m:])
        self.assertIsInstance(sims, np.ndarray)
        self.assertEqual(len(sims), m)
        self.assertGreaterEqual(sims.min(), 0.0)
        self.assertLess(sims.max(), 1.0)  # None should be maximally self-similar


if __name__ == '__main__':
    unittest.main()
