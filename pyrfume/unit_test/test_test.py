import unittest

from quantities.umath import sin
from pyrfume.odorants import Compound, ChemicalOrder, Vendor, Molecule
from pyrfume import TriangleTest, Component, Mixture
from datetime import datetime



class TriangleTestTestCase(unittest.TestCase):

    def test_test(self):

        vendor = Vendor("Test_Vendor", "")

        cid_CaCl2 = 24854
        cas_CaCl2 = "10043-52-4"
        molecule_CaCl2 = Molecule(cid_CaCl2, "Calcium chloride", True)
        chemical_order_molecule_CaCl2 = ChemicalOrder(molecule_CaCl2, vendor, "part 0", 0.5, None)
        compound_CaCl2 = Compound(chemical_order_molecule_CaCl2, "TEST", datetime.now, datetime.now, False)
        component_CaCl2 = Component(cid_CaCl2, "CaCl2", cas_CaCl2, 0.5, compound_CaCl2)
        mixture_CaCl2 = Mixture(2, [component_CaCl2])
        descriptors = {
            cas_CaCl2 : ["CaCl2 unique descriptor", "common descriptor"],
            "dravnieks" : ["CaCl2 dravnieks descriptor", "common dravnieks descriptor"],
            "sigma_ff" : ["CaCl2 sigma_ff descriptor", "common sigma_ff descriptor"]
        }
        component_CaCl2.set_descriptors('unittest source', descriptors)

        cid_NaCl = 5234
        cas_NaCl = "7647-14-5"
        molecule_NaCl = Molecule(cid_NaCl, "Sodium chloride", True)
        chemical_order_NaCl = ChemicalOrder(molecule_NaCl, vendor, "part 1", 0.5, None)
        compound_NaCl = Compound(chemical_order_molecule_CaCl2, "TEST", datetime.now, datetime.now, False)
        component_NaCl = Component(cid_NaCl, "NaCl", cas_NaCl, 0.5, compound_NaCl)
        mixture_NaCl = Mixture(2, [component_NaCl])
        descriptors = {
            cas_NaCl : ["NaCl unique descriptor", "common descriptor"],
            "dravnieks" : ["NaCl dravnieks descriptor", "common dravnieks descriptor"],
            "sigma_ff" : ["NaCl sigma_ff descriptor", "common sigma_ff descriptor"]
        }
        component_NaCl.set_descriptors('unittest source', descriptors)


        cid_HCl = 313
        cas_HCl = "7647-01-0"
        molecule_HCl = Molecule(cid_HCl, "Hydrochloric acid", True)
        chemical_order_HCl = ChemicalOrder(molecule_HCl, vendor, "part 1", 0.5, None)
        compound_HCl = Compound(chemical_order_molecule_CaCl2, "TEST", datetime.now, datetime.now, False)
        component_HCl = Component(cid_HCl, "HCl", cas_HCl, 0.5, compound_HCl)
        mixture_HCl = Mixture(2, [component_HCl])

        descriptors = {
            cas_HCl : ["HCl unique descriptor", "common descriptor"],
            "dravnieks" : ["HCl dravnieks descriptor", "common dravnieks descriptor"],
            "sigma_ff" : ["HCl sigma_ff descriptor", "common sigma_ff descriptor"]
        }
        component_HCl.set_descriptors('unittest source', descriptors)
        odorants = [mixture_CaCl2]

        test = TriangleTest(0, odorants, 1.0, True)
        test.add_odorant(mixture_NaCl)
        test.add_odorant(mixture_NaCl)

        self.assertEqual(mixture_NaCl, test.double)
        self.assertEqual(mixture_CaCl2, test.single)
        self.assertEqual(mixture_NaCl, test.pair[0])
        self.assertEqual(mixture_CaCl2, test.pair[1])
        

        odorants = [mixture_CaCl2]

        test = TriangleTest(0, odorants, 1.0, True)
        test.add_odorants([mixture_NaCl, mixture_NaCl])
        
        self.assertEqual(mixture_NaCl, test.double)
        self.assertEqual(mixture_CaCl2, test.single)
        self.assertEqual(mixture_NaCl, test.pair[0])
        self.assertEqual(mixture_CaCl2, test.pair[1])

        self.assertEqual(1, test.N)
        self.assertEqual(1, test.r)
        self.assertEqual(0, test.overlap(False))
        self.assertEqual(0, test.overlap(True))        

        self.assertTrue("common descriptor" in test.common_descriptors('unittest source'))
        self.assertTrue("NaCl unique descriptor" in test.unique_descriptors('unittest source'))
        self.assertTrue("CaCl2 unique descriptor" in test.unique_descriptors('unittest source'))
        all_descriptors = {
            'unittest source' : [
                'NaCl unique descriptor', 
                'CaCl2 unique descriptor',
                "common descriptor"
            ],
            "dravnieks" : [
                "NaCl dravnieks descriptor",
                "HCl dravnieks descriptor",
                "CaCl2 dravnieks descriptor", 
                "common dravnieks descriptor"
            ],
            "sigma_ff" : [
                "NaCl sigma_ff descriptor",
                "HCl sigma_ff descriptor",
                "CaCl2 sigma_ff descriptor", 
                "common sigma_ff descriptor"
            ]
        }
        self.assertAlmostEqual(
            test.descriptors_correlation('unittest source', all_descriptors), -0.5, 2
        )

        self.assertAlmostEqual(
            test.descriptors_correlation('unittest source', all_descriptors), -0.5, 2
        )

        test.descriptors_correlation2(all_descriptors)
        diff_list = list(test.descriptors_difference('unittest source', all_descriptors))
        self.assertEqual(diff_list.count(1.0), 2)
        self.assertEqual(diff_list.count(0), 1)

        self.assertEqual(len(test.common_components), 0)
        mixture_NaCl_HCl = mixture_NaCl
        mixture_NaCl_HCl.add_component(component_HCl)
        mixture_CaCl2_HCl = mixture_CaCl2
        mixture_CaCl2_HCl.add_component(component_HCl)
        self.assertTrue(component_HCl in test.common_components)
        self.assertTrue(component_NaCl in test.unique_components)
        self.assertTrue(component_CaCl2 in test.unique_components)


if __name__ == '__main__':
    unittest.main()
