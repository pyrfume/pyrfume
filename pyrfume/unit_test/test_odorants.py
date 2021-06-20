import json
import unittest
import quantities as pq
from quantities.units.velocity import speed_of_light
from pyrfume import save_data, set_data_path, get_data_path
from pyrfume.odorants import url_to_json, Solution, get_cid, \
                            get_cids, from_cids, cids_to_cas, \
                            cids_to_smiles, cactus, cactus_image, \
                            get_compound_summary, get_compound_odor, \
                            _parse_other_info, smiles_to_image, crop_image, \
                            all_odorants, all_sources, all_cids, all_smiles
                            

from .unittest_utils import get_substances

class OdorantsTestCase(unittest.TestCase):
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

        self.assertAlmostEqual(solution.vapor_concentration(molecule_C4H8O2), 0.032457325, 3)
        self.assertAlmostEqual(solution.vapor_concentration(molecule_C4H8S), 0.006924697, 3)
        self.assertAlmostEqual(solution.vapor_concentration(molecule_C2H6O), 0.034562520, 3)

        self.assertAlmostEqual(solution.molar_evaporation_rate(molecule_C4H8O2).item(), 0.00116778, 3)
        self.assertAlmostEqual(solution.molar_evaporation_rate(molecule_C4H8S).item(), 0.00023938, 3)
        self.assertAlmostEqual(solution.molar_evaporation_rate(molecule_C2H6O).item(), 0.00122993, 3)

    def test_molecular(self):
        molecule_C4H8O2_1, molecule_C4H8S_1, molecule_C2H6O_1 = get_substances("molecules")
        molecule_C4H8O2_2, molecule_C4H8S_2, molecule_C2H6O_2 = get_substances("molecules")
        self.assertTrue(molecule_C4H8O2_1 == molecule_C4H8O2_2)
        self.assertTrue(molecule_C4H8S_1 == molecule_C4H8S_2)
        self.assertTrue(molecule_C2H6O_1 == molecule_C2H6O_2)

        self.assertTrue(molecule_C4H8O2_1 > molecule_C4H8S_1)
        self.assertTrue(molecule_C4H8S_1 > molecule_C2H6O_1)
        self.assertTrue(molecule_C2H6O_1 < molecule_C4H8O2_1)

        self.assertIsInstance(repr(molecule_C4H8O2_1), str)
        self.assertIsInstance(repr(molecule_C4H8S_1), str)
        self.assertIsInstance(repr(molecule_C2H6O_1), str)

    def test_url_to_json(self):
        json_dict = url_to_json("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/962/synonyms/JSON")
        self.assertEqual(json_dict["InformationList"]["Information"][0]["Synonym"][0], "water")

    def test_get_cid(self):
        self.assertEqual(get_cid("64-17-5"), 702)
        self.assertEqual(get_cid("141-78-6"), 8857)
        self.assertEqual(get_cid("110-01-0"), 1127)

        cids = get_cids(["64-17-5",  "141-78-6", "110-01-0"])
        self.assertEqual(cids["64-17-5"], 702)
        self.assertEqual(cids["141-78-6"], 8857)
        self.assertEqual(cids["110-01-0"], 1127)

    def test_from_cids(self):
        results = from_cids([702, 8857, 1127])
        self.assertEqual(results[0]['IUPACName'], 'ethanol')
        self.assertEqual(results[1]['IUPACName'], 'ethyl acetate')
        self.assertEqual(results[2]['IUPACName'], 'thiolane')

    def test_cids_to_smiles(self):
        results = cids_to_smiles([702, 8857, 1127])
        self.assertEqual(results[702], 'CCO')
        self.assertEqual(results[8857], 'CCOC(=O)C')
        self.assertEqual(results[1127], 'C1CCSC1')

    def test_cids_to_cas(self):
        results = cids_to_cas([702, 8857, 1127])
        self.assertEqual(results[702][0], '64-17-5')
        self.assertEqual(results[8857][0], '141-78-6')
        self.assertEqual(results[1127][0], '110-01-0')

    def test_cactus(self):
        results = cactus("Ethyl acetate", "cas").split("\n")
        self.assertTrue("141-78-6" in results)

        results = cactus("Ethanol", "cas").split('\n')
        self.assertTrue("64-17-5" in results)

        results = cactus("Tetrahydrothiophene", "cas").split('\n')
        self.assertTrue("110-01-0" in results)

    def test_get_compound_summary(self):
        self.assertIsInstance(get_compound_summary(702, 'Physical Description'), dict)
        self.assertIsInstance(get_compound_summary(8857, 'Physical Description'), dict)
        self.assertIsInstance(get_compound_summary(1127, 'Physical Description'), dict)

    def test_get_compound_odor(self):
        self.assertIsInstance(get_compound_odor(702, False), list)
        self.assertIsInstance(get_compound_odor(8857, False), list)
        self.assertIsInstance(get_compound_odor(1127, False), list)
        self.assertIsInstance(get_compound_odor(702, True), list)
        self.assertIsInstance(get_compound_odor(8857, True), list)
        self.assertIsInstance(get_compound_odor(1127, True), list)

    def test__parse_other_info(self):
        self.assertTrue(len(_parse_other_info(None)) == 0)

        test_info_dict = {
            "String" : "value of String",
            "Value" : "Number 0",
            "Dict" : {"String" : "value of String in the sub-dict"}
        }
        result = _parse_other_info(test_info_dict)
        self.assertTrue('value of String' in result)
        self.assertTrue('Number 0' in result)
        self.assertTrue('value of String in the sub-dict' in result)

        test_info_list = [test_info_dict]
        result = _parse_other_info(test_info_list)
        self.assertTrue('value of String' in result)
        self.assertTrue('Number 0' in result)
        self.assertTrue('value of String in the sub-dict' in result)

    @unittest.skip("Failure expected. A fix is needed.")
    def test_cactus_image(self):
        cactus_image("CCC")

    @unittest.skip("Failure expected. The data is needed for this test case.")
    def test_all_items(self):
        from pandas import DataFrame
        #set_data_path('C:/icon/pyrfume')
        self.assertIsInstance(all_odorants(), DataFrame)
        self.assertIsInstance(all_sources(), DataFrame)
        self.assertIsInstance(all_cids(), list)
        self.assertIsInstance(all_smiles(), list)

if __name__ == '__main__':
    unittest.main()
