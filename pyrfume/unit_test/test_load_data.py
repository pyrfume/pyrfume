import platform
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
        
    @unittest.skipIf('spike' not in platform.node(), "Only tested on Spike")
    def test_load_morgan_skipped(self):
        my_cids = [129, 239]
        morgan_sim = pyrfume.load_data('morgan/features_sim.csv', cids=my_cids)
        self.assertEqual(morgan_sim.shape[0], len(my_cids))
        
if __name__ == '__main__':
    unittest.main()
