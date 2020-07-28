import unittest
import quantities as pq
from pyrfume.odorants import Solution
from unittest_utils import get_substances

class OdotantsTestCase(unittest.TestCase):
    def setUp(self):
        compound_C4H8O2, compound_HCl, compound_C2H6O = get_substances("compounds")
        self.components = {
            compound_C4H8O2 : 100 * pq.mL,
            compound_HCl : 100 * pq.mL,
            compound_C2H6O : 100 * pq.mL
        }

    def test_solution(self):
        solution = Solution(self.components, None)
        for key, value in self.components.items():
            self.assertTrue(key in solution.compounds.keys())
            self.assertTrue(value in solution.compounds.values())


if __name__ == '__main__':
    unittest.main()
