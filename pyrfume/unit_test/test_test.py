import unittest

from pyrfume.odorants import Compound, ChemicalOrder, Vendor, Molecule
from pyrfume.experiments import TriangleTest, Result
from pyrfume.objects import Component, Mixture
from datetime import datetime

from .unittest_utils import get_substances

class TriangleTestTestCase(unittest.TestCase):

    def test_test(self):

        mixture_C4H8O2, mixture_C4H8S, mixture_C2H6O = get_substances("mixtures")
        component_C4H8O2, component_C4H8S, component_C2H6O = get_substances("components")

        odorants = [mixture_C4H8O2]

        test = TriangleTest(0, odorants, 1.0, True)
        test.add_odorant(mixture_C2H6O)
        test.add_odorant(mixture_C2H6O)

        self.assertEqual(mixture_C2H6O, test.double)
        self.assertEqual(mixture_C4H8O2, test.single)
        self.assertEqual(mixture_C2H6O, test.pair[0])
        self.assertEqual(mixture_C4H8O2, test.pair[1])
        

        odorants = [mixture_C4H8O2]

        test = TriangleTest(0, odorants, 1.0, True)
        test.add_odorants([mixture_C2H6O, mixture_C2H6O])
        
        self.assertEqual(mixture_C2H6O, test.double)
        self.assertEqual(mixture_C4H8O2, test.single)
        self.assertEqual(mixture_C2H6O, test.pair[0])
        self.assertEqual(mixture_C4H8O2, test.pair[1])

        self.assertEqual(1, test.N)
        self.assertEqual(1, test.r)
        self.assertEqual(0, test.overlap(False))
        self.assertEqual(0, test.overlap(True))        

        self.assertTrue("common descriptor" in test.common_descriptors('unittest source'))
        self.assertTrue("C2H6O unique descriptor" in test.unique_descriptors('unittest source'))
        self.assertTrue("C4H8O2 unique descriptor" in test.unique_descriptors('unittest source'))
        all_descriptors = {
            'unittest source' : [
                'C2H6O unique descriptor', 
                'C4H8O2 unique descriptor',
                "common descriptor"
            ],
            "dravnieks" : [
                "C2H6O dravnieks descriptor",
                "C4H8S dravnieks descriptor",
                "C4H8O2 dravnieks descriptor", 
                "common dravnieks descriptor"
            ],
            "sigma_ff" : [
                "C2H6O sigma_ff descriptor",
                "C4H8S sigma_ff descriptor",
                "C4H8O2 sigma_ff descriptor", 
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
        mixture_C2H6O_C4H8S = mixture_C2H6O
        mixture_C2H6O_C4H8S.add_component(component_C4H8S)
        mixture_C4H8O2_C4H8S = mixture_C4H8O2
        mixture_C4H8O2_C4H8S.add_component(component_C4H8S)
        unique_components = test.unique_components
        unique_components = [str(x) for x in unique_components]
        self.assertTrue(component_C4H8S in test.common_components)
        self.assertTrue(str(component_C2H6O) in unique_components)
        self.assertTrue(str(component_C4H8O2) in unique_components)
        self.assertEqual(test.n_undescribed('unittest source'), (0, 0))

        result1 = Result(test, 0, True)
        results = [result1]
        self.assertEqual(test.fraction_correct(results), 1)

        test2 = TriangleTest(0, odorants, 1.0, True)
        result2 = Result(test2, 0, False)
        results.append(result2)
        self.assertEqual(test.fraction_correct(results), 0.5)

if __name__ == '__main__':
    unittest.main()
