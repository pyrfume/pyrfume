import unittest
from datetime import datetime
from pyrfume import Mixture, Component
from pyrfume.odorants import Compound, ChemicalOrder, Vendor, Molecule
class MixtureTestCase(unittest.TestCase):
    def test_constructor(self):
        mixture1 = Mixture(0)
        mixture2 = Mixture(9999)
        self.assertEqual(mixture1.N, 0)
        self.assertEqual(mixture1.r(mixture2), 0)
        self.assertEqual(mixture1.overlap(mixture2), 0)
        self.assertEqual(mixture1.hamming(mixture2), 0)

        cid = 21946271
        cas = "12002-61-8"
        vendor = Vendor("Test_Vendor", "https://en.wikipedia.org/wiki/Actinium(III)_oxide")
        molecule = Molecule(cid, "test_mol", False)
        molecule = Molecule(cid, "test_mol", True)
        chemical_order = ChemicalOrder(molecule, vendor, "", 0.9, None)
        compound = Compound(chemical_order, "TEST", datetime.now, datetime.now, False)
        component = Component(cid, "test_name", cas, 0.5, compound)
        mixture = Mixture(100, [component])

if __name__ == '__main__':
    unittest.main()
