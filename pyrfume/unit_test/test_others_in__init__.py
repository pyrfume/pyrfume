import unittest
import numpy as np
from pyrfume import Distance, Result, odorant_distances, ROC, correct_matrix, TriangleTest
from .test_test import get_mixtures_odorants

odorants = get_mixtures_odorants()
mixture_CaCl2, mixture_HCl, mixture_NaCl = odorants[0]
component_CaCl2, component_HCl, component_NaCl = odorants[1]

class OthersTestCase(unittest.TestCase):

    def test_Distance(self):
        distance = Distance(mixture_CaCl2, mixture_NaCl, 1.0)

    def test_odorant_distances_ROC_correct_matrix(self):
        test1 = TriangleTest(0, list(odorants[0]), 0.5, True)
        new_odorants = [mixture_CaCl2, mixture_NaCl, mixture_NaCl]
        test1.add_odorants(new_odorants)
        
        test2 = TriangleTest(1, list(odorants[0]), 0.5, False)
        new_odorants = [mixture_CaCl2, mixture_HCl, mixture_HCl]
        test2.add_odorants(new_odorants)

        result1 = Result(test1, 0, True)
        result2 = Result(test2, 1, False)
        results = [result1, result2]

        distances: dict = odorant_distances(results, 0)
        self.assertTrue(1 in distances.values())
        self.assertTrue(0 in distances.values())
        
        roc = ROC(results, 1)
        self.assertEqual(roc[0][0], 1)
        self.assertEqual(roc[1][0], 1)
        matrix = correct_matrix(results, 1, None)

if __name__ == '__main__':
    unittest.main()
