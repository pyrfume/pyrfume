import unittest
import quantities as pq
from pyrfume.odorants import Solution
from unittest_utils import get_substances

class OdotantsTestCase(unittest.TestCase):
    def setUp(self):
        compound_CaCl2, compound_HCl, compound_NaCl = get_substances("compounds")
        self.components = {
            compound_CaCl2 : 100 * pq.mL,
            compound_HCl : 100 * pq.mL,
            compound_NaCl : 100 * pq.mL
        }

    def test_solution(self):
        solution = Solution(self.components, None)
        for key, value in self.components.items():
            self.assertTrue(key in solution.compounds.keys())
            self.assertTrue(value in solution.compounds.values())


if __name__ == '__main__':
    unittest.main()
