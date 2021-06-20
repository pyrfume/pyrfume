from pathlib import Path
import unittest

class DataAndConfigTestCase(unittest.TestCase):

    def test_init_reset_config(self):
        from pyrfume import init_config, reset_config
        init_config(False)
        reset_config()
        init_config(True)
        
    def test_read_write_config(self):
        from pyrfume import read_config, write_config
        write_config("PATHS", "a", "b")
        self.assertEqual(read_config("PATHS", "a"), "b")

    def test_data_path(self):
        from pyrfume import set_data_path, get_data_path
        from pyrfume.base import PACKAGE_DIR, DEFAULT_DATA_PATH
        import os

        curr_data_path = get_data_path()
        
        path_not_exists = PACKAGE_DIR / "THIS_IS_AN_INVALID_PATH"
        self.assertRaises(Exception, set_data_path, path_not_exists, create=False)
        self.assertRaises(Exception, get_data_path, path_not_exists, create=False)

        path1 = PACKAGE_DIR / "unit_test"
        set_data_path(path1)
        path2 = get_data_path()
        self.assertEqual(path1, path2)

        Path(DEFAULT_DATA_PATH).mkdir(exist_ok=True)
        set_data_path(DEFAULT_DATA_PATH)
        path3 = get_data_path()
        self.assertEqual(path3, DEFAULT_DATA_PATH)

        set_data_path(curr_data_path)

    def test_load_data(self):
        import pickle, os
        from pyrfume.base import DEFAULT_DATA_PATH
        from pyrfume import load_data, save_data
        import pandas as pd

        data = {'col1': [1, 2], 'col2': [3, 4]}
        file_path = DEFAULT_DATA_PATH / "data.pkl"
        path_not_exists = DEFAULT_DATA_PATH / "THIS_IS_AN_INVALID_PATH"
        
        self.assertRaises(Exception, save_data, data, path_not_exists)
        save_data(data, file_path)

        data_gain = load_data(file_path)
        self.assertEqual(data_gain, data)
        os.remove(file_path)

        file_path = DEFAULT_DATA_PATH / "data.csv"

        df = pd.DataFrame(data)
        save_data(df, file_path)
        #with open(file_path, "w") as f:
        #    f.write("0,1,2,3\n0,1,2,3")

        data_gain = load_data(file_path)

        for index1 in range(len(data_gain.values)):
            for index2 in range(len(data_gain.values[index1])):
                self.assertEqual(data_gain.values[index1][index2], df.values[index1][index2])
                
        os.remove(file_path)

    def test_save_data(self):
        from pyrfume.base import DEFAULT_DATA_PATH


if __name__ == '__main__':
    unittest.main()
