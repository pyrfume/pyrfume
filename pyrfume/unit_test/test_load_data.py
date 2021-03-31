import unittest

class DataLoadRemoteTestCase(unittest.TestCase):
    def setUp(self):
        import pyrfume
    
    def test_load_manoel_2021(self):
        data = pyrfume.load_data('manoel_2021/behavior-main.csv', remote=True)
    
    def test_load_snitz_2013(self):
        data = pyrfume.load_data('snitz_2013/behavior-main.csv', remote=True)
        molecules = pyrfume.load_data('snitz_2013/molecules-info.csv', remote=True)


if __name__ == '__main__':
    unittest.main()
