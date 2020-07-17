import unittest
import quantities as pq
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

    def test_example(self):
        cid = 21946271
        cas = "12002-61-8"
        vendor = Vendor("Test_Vendor", "https://en.wikipedia.org/wiki/Actinium(III)_oxide")
        molecule = Molecule(cid, "test_mol", False)
        molecule = Molecule(cid, "test_mol", True)
        self.assertIsNone(molecule.molarity)
        molecule.molecular_weight = pq.Quantity(502.1, pq.g / pq.mol)
        molecule.density = pq.Quantity(501.9982, pq.g / pq.cm ** 3)
        self.assertAlmostEqual(molecule.molarity.item(), 999.797251, 3)

        chemical_order = ChemicalOrder(molecule, vendor, "", 0.9, None)
        compound = Compound(chemical_order, "TEST", datetime.now, datetime.now, False)

        self.assertIsInstance(getattr(compound, "vendor"), Vendor)
        self.assertIsInstance(getattr(compound, "cid"), int)

        component = Component(cid, "test_name", cas, 0.5, compound)
        mixture = Mixture(100, [component])

if __name__ == '__main__':
    unittest.main()
