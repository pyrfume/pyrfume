import unittest

from pyrfume.experiments import (
    ROC,
    Distance,
    Result,
    TriangleTest,
    correct_matrix,
    odorant_distances,
)

from .unittest_utils import get_substances


class OthersTestCase(unittest.TestCase):
    def setUp(self):
        self.mixtures = get_substances("mixtures")
        self.mixture_C4H8O2, self.mixture_C4H8S, self.mixture_C2H6O = self.mixtures
        self.components = get_substances("components")
        self.component_C4H8O2, self.component_C4H8S, self.component_C2H6O = self.components

    def test_Distance(self):
        Distance(self.mixture_C4H8O2, self.mixture_C2H6O, 1.0)

    def test_odorant_distances_ROC_correct_matrix(self):
        test1 = TriangleTest(0, list(self.mixtures), 0.5, True)
        new_odorants = [self.mixture_C4H8O2, self.mixture_C2H6O, self.mixture_C2H6O]
        test1.add_odorants(new_odorants)

        test2 = TriangleTest(1, list(self.mixtures), 0.5, False)
        new_odorants = [self.mixture_C4H8O2, self.mixture_C4H8S, self.mixture_C4H8S]
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
        correct_matrix(results, 1, None)


if __name__ == "__main__":
    unittest.main()
