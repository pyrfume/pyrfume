import unittest
from pyrfume import Mixture, Component

class MixtureTestCase(unittest.TestCase):
    def test_constructor(self):
        mixture1 = Mixture(0)
        mixture2 = Mixture(9999)
        self.assertEqual(mixture1.N, 0)
        self.assertEqual(mixture1.r(mixture2), 0)
        self.assertEqual(mixture1.overlap(mixture2), 0)
        self.assertEqual(mixture1.hamming(mixture2), 0)
if __name__ == '__main__':
    unittest.main()
