import unittest
from pyrfume import Mixture, Component

class MixtureTestCase(unittest.TestCase):
    def test_constructor(self):
        mixture = Mixture(0)
        mixture = Mixture(9999)

if __name__ == '__main__':
    unittest.main()
