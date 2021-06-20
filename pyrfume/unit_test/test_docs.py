from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
from pathlib import Path
import platform
import unittest


class DocsTest(unittest.TestCase):
    def setUp(self):
        import pyrfume
        pyrfume_dir = Path(pyrfume.__path__[0]).parent
        self.docs_path = pyrfume_dir / 'docs'
        #pyrfume.set_data_path(pyrfume_dir.parent / 'pyrfume-data')
    
    def execute(self, path):
        with open(path) as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb)
        
    def execute_doc(self, name):
        self.execute(self.docs_path / (name+'.ipynb'))

    def test_1_your_data(self):
        self.execute_doc('your-data')
        
    @unittest.skipIf('spike' not in platform.node(), "Only tested on Spike")
    def test_2_featurization(self):
        self.execute_doc('featurization')
        
    def test_3_visualization(self):
        self.execute_doc('visualization')
        
    def test_4_published_data(self):
        self.execute_doc('published-data')


if __name__ == '__main__':
    unittest.main()
