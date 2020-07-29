import unittest
import quantities as pq
from quantities.units.velocity import speed_of_light
from pyrfume.odorants import Compound, Molecule, Solution
from .unittest_utils import get_substances, get_water

class OdotantsTestCase(unittest.TestCase):
    def setUp(self):
        self.compound_C4H8O2, self.compound_C4H8S, self.compound_C2H6O = get_substances("compounds")
        
        self.compounds = {
            self.compound_C4H8O2 : 100 * pq.mL,
            self.compound_C4H8S : 100 * pq.mL,
            self.compound_C2H6O : 100 * pq.mL
        }

    def test_solution(self):
        components = self.compounds
        solution = Solution(components, None)
        for key, value in self.compounds.items():
            self.assertTrue(key in solution.compounds.keys())
            self.assertTrue(value in solution.compounds.values())

        for compound in self.compounds.keys():
            self.assertTrue(compound in solution.compounds)

        components = {solution : 100 * pq.mL}
        solution = Solution(components, None)
        for compound in self.compounds.keys():
            self.assertTrue(compound in solution.compounds)
        
        molecule_C4H8O2 = self.compound_C4H8O2.chemical_order.molecule
        molecule_C4H8S = self.compound_C4H8S.chemical_order.molecule
        molecule_C2H6O = self.compound_C2H6O.chemical_order.molecule
        
        self.assertAlmostEqual(solution.molecules[molecule_C4H8O2].item(), 1.02376683, 3)
        self.assertAlmostEqual(solution.molecules[molecule_C4H8S].item(), 1.1307701, 3)
        self.assertAlmostEqual(solution.molecules[molecule_C2H6O].item(), 1.71329962, 3)

        self.assertAlmostEqual(solution.molarities[molecule_C4H8O2].item(), 0.01023767, 3)
        self.assertAlmostEqual(solution.molarities[molecule_C4H8S].item(), 0.0113077, 3)
        self.assertAlmostEqual(solution.molarities[molecule_C2H6O].item(), 0.017133, 3)

        self.assertAlmostEqual(solution.mole_fraction(molecule_C4H8O2), 0.2646872, 3)
        self.assertAlmostEqual(solution.mole_fraction(molecule_C4H8S), 0.2923520, 3)
        self.assertAlmostEqual(solution.mole_fraction(molecule_C2H6O), 0.442960, 3)

        self.assertAlmostEqual(solution.partial_pressure(molecule_C4H8O2).item(), 3.28873846, 3)
        self.assertAlmostEqual(solution.partial_pressure(molecule_C4H8S).item(), 0.70164502, 3)
        self.assertAlmostEqual(solution.partial_pressure(molecule_C2H6O).item(), 3.50204736, 3)

        self.assertAlmostEqual(solution.total_pressure.item(), 7492.4308428, 4)
if __name__ == '__main__':
    unittest.main()
