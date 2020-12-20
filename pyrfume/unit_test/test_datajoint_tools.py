import unittest
import datajoint as dj
import pandas as pd



class DataJointTestCase(unittest.TestCase):

    def setUp(self):
        dj.config['database.host'] = '127.0.0.1'
        dj.config['database.user'] = 'root'
        dj.config['database.password'] = 'simple'

    def test_schematizing_odorants(self):
        """Test case for schematizations of classes in the odorants module 
            and inserting sample data into datajoint table.
        """

        from pyrfume.odorants import Molecule, Vendor, ChemicalOrder, Compound, Solution
        from pyrfume import datajoint_tools as djt

        schematized_classes = {}
        schema = dj.schema("test_schema")
        schema.drop(force=True)
        schema = dj.schema("test_schema", schematized_classes)

        classes_tobe_schematized = ['Molecule', 'Vendor', 'ChemicalOrder', 'Compound', 'Solution']
        for cls_name in classes_tobe_schematized:
            schematized_classes[cls_name] = djt.schematize(locals()[cls_name], schema)

        # test if definition defined.
        for cls_name in classes_tobe_schematized:
            self.assertIsInstance(schematized_classes[cls_name].definition, str)


        Molecule = schematized_classes['Molecule']
        Molecule.insert1({'cid': 6334, 'cas': '', 'name': 'propane', 'iupac': 'propane', 'synonyms': ''})
        molecules = pd.DataFrame(Molecule.fetch())
        self.assertEqual(molecules.shape[0], 1)

        Vendor = schematized_classes['Vendor']
        Vendor.insert1({"name": "vendor1", "url": "www.vendor1.com"})
        vendors = pd.DataFrame(Vendor.fetch())
        self.assertEqual(vendors.shape[0], 1)

        ChemicalOrder = schematized_classes['ChemicalOrder']
        ChemicalOrder.insert1({"molecule_id": 1, "vendor_id": 1, 'part_id': 1, 'purity': 0.5})
        chemicalorders = pd.DataFrame(ChemicalOrder.fetch())
        self.assertEqual(chemicalorders.shape[0], 1)

        Compound = schematized_classes['Compound']
        Compound.insert1({'chemicalorder_id': 1, 'stock': 'enough', 'is_solvent': 1, 'date_arrived': '2020-09-27 02:18:24', 'date_opened': '2020-09-27 02:18:24'})
        compounds = pd.DataFrame(Compound.fetch())
        self.assertEqual(compounds.shape[0], 1)

        Solution = schematized_classes['Solution']
        Solution.insert1({"date_created": '2020-09-27 02:18:24'})
        solutions = pd.DataFrame(Solution.fetch())
        self.assertEqual(solutions.shape[0], 1)

        Solution.Component.insert1({"solution_id": 1, "compound_id": 1, "value": 1.0})
        solution_components = pd.DataFrame(Solution.Component.fetch())
        self.assertEqual(solution_components.shape[0], 1)


if __name__ == '__main__':
    unittest.main()
