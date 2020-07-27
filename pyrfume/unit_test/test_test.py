import unittest

from pyrfume.odorants import Compound, ChemicalOrder, Vendor, Molecule
from pyrfume import TriangleTest, Component, Mixture
from datetime import datetime

from unittest_utils import get_substances

class TriangleTestTestCase(unittest.TestCase):

    def test_test(self):

        mixture_CaCl2, mixture_HCl, mixture_NaCl = get_substances("mixtures")
        component_CaCl2, component_HCl, component_NaCl = get_substances("components")

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
        unique_components = test.unique_components
        unique_components = [str(x) for x in unique_components]
        self.assertTrue(component_HCl in test.common_components)
        self.assertTrue(str(component_NaCl) in unique_components)
        self.assertTrue(str(component_CaCl2) in unique_components)
        self.assertEqual(test.n_undescribed('unittest source'), (0, 0))

        from pyrfume import Result

        result1 = Result(test, 0, True)
        results = [result1]
        self.assertEqual(test.fraction_correct(results), 1)

        test2 = TriangleTest(0, odorants, 1.0, True)
        result2 = Result(test2, 0, False)
        results.append(result2)
        self.assertEqual(test.fraction_correct(results), 0.5)

if __name__ == '__main__':
    unittest.main()
