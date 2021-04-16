import unittest
import pyrfume

class DataLoadRemoteTestCase(unittest.TestCase):    
    def test_load_manoel_2021(self):
        data = pyrfume.load_data('manoel_2021/behavior.csv')
    
    def test_load_snitz_2013(self):
        data = pyrfume.load_data('snitz_2013/behavior.csv')
        molecules = pyrfume.load_data('snitz_2013/molecules.csv')
        
    def test_load_ravia_2020(self):
        data = pyrfume.load_data('ravia_2020/behavior1.csv')
        manifest = pyrfume.load_manifest('ravia_2020')


if __name__ == '__main__':
    unittest.main()
