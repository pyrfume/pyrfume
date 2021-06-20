import unittest
import quantities as pq
from datetime import datetime
from pyrfume.objects import Mixture, Component
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

        component = Component(cid, "name1", cas, 0.5, compound)
        component1 = Component(cid, "name2", cas, 0.8, compound)
        mixture = Mixture(1, [component])

        components_vector = mixture.components_vector(mixture.components, 1)
        self.assertEqual(components_vector.size, 1)

        mixture1 = Mixture(0)
        self.assertIsNone(mixture1.r(mixture))
        self.assertEqual(mixture.overlap(mixture, True), 100.0)
        
        mixture.add_component(component1)
        mixture.remove_component(component1)
        self.assertEqual(len(mixture.components), 1)
        mixture.remove_component(component)
        self.assertEqual(len(mixture.components), 0)

        cas_descriptor = {
            cas : ["cas descriptor"],
            "dravnieks" : ["dravnieks descriptor"],
            "sigma_ff" : ["sigma_ff descriptor"]
        }
        cas_descriptor1 = {
            cas : {"cas descriptor 1" : 1},
            "dravnieks" : {"dravnieks cas description 1" : 2},
            "sigma_ff" : {"sigma_ff descriptor 1"}
        }
        component.set_descriptors("unittest source", cas_descriptor)
        component1.set_descriptors("unittest source", cas_descriptor1)
        mixture.add_component(component)
        mixture.add_component(component1)

        descriptors_list = mixture.descriptor_list("unittest source")
        self.assertTrue('cas descriptor 1' in descriptors_list)
        self.assertTrue('cas descriptor' in descriptors_list)

        all_descriptors = {
            'unittest source' : 'cas descriptor 1', 
            'unittest source' : 'cas descriptor'
        }
        self.assertEqual(mixture.descriptor_vector("unittest source", all_descriptors).size, 14)
        
        all_descriptors = {
            'dravnieks' : "dravnieks cas description 1", 
            'sigma_ff' : "sigma_ff descriptor 1"
        }

        self.assertEqual(mixture.descriptor_vector2(all_descriptors).size, 48)

        described_components = mixture.described_components("unittest source")
        self.assertTrue(component in described_components)
        self.assertTrue(component1 in described_components)

        n = mixture.n_described_components("unittest source")
        self.assertEqual(n, 2)

        self.assertEqual(mixture.fraction_components_described("unittest source"), 1)

        test_feature = {cid : {"a" : 1}}
        self.assertEqual(mixture.matrix(test_feature).shape, (2, 1))
        self.assertEqual(mixture.vector(test_feature)[0], 2.0)
        self.assertIsInstance(str(mixture), str)
        self.assertIsInstance(str(mixture1), str)

        self.assertEqual(component.cid, 21946271)
        self.assertIsInstance(str(component), str)
        self.assertIsInstance(str(component1), str)

if __name__ == '__main__':
    unittest.main()
