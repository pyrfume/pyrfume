import unittest
from pyrfume.physics import *
import quantities as pq

class PhysicsTestCase(unittest.TestCase):
    def test_mackay(self):
        q = 1 * pq.Pa
        self.assertAlmostEqual(2.8238e-07, mackay(q).item(), 3)
        q = -1 * pq.Pa
        self.assertEqual(0, mackay(q).item())
    
    def test_bernoulli(self):
        bernoulli()
        bernoulli(1 * pq.m/pq.s)
        bernoulli(1 * pq.m/pq.s, 1 * pq.Pa)
        bernoulli(1 * pq.m/pq.s, 1 * pq.Pa, 1 * pq.g / pq.cm ** 3)
        bernoulli(1 * pq.m/pq.s, 1 * pq.Pa, 1 * pq.g / pq.cm ** 3, 3.72076 * pq.m / (pq.s) ** 2)
        bernoulli(1 * pq.m/pq.s, 1 * pq.Pa, 1 * pq.g / pq.cm ** 3, 3.72076 * pq.m / (pq.s) ** 2, 1)
        self.assertEqual(
            bernoulli(
                1 * pq.m/pq.s, 
                1 * pq.Pa, 
                1 * pq.g / pq.cm ** 3, 
                3.72076 * pq.m / (pq.s) ** 2, 
                1, 
                1 * (pq.m / pq.s) ** 2
            ),
            []
        )

    def test_venturi(self):
        venturi(1 * pq.g / pq.cm ** 3)
        venturi(1 * pq.g / pq.cm ** 3, 1 * pq.Pa)
        venturi(1 * pq.g / pq.cm ** 3, 1 * pq.Pa, 2 * pq.Pa, )

        v2 = Symbol("v2", real=True, positive=True)
        result = venturi(1 * pq.g / pq.cm ** 3, 1 * pq.Pa, 2 * pq.Pa, 1 * pq.m / pq.s)[0][v2]
        self.assertAlmostEqual(result, 0.998999, 4)

        result = venturi(1 * pq.g / pq.cm ** 3, 1 * pq.Pa, 2 * pq.Pa, 1 * pq.m / pq.s, 2 * pq.m / pq.s)
        self.assertEqual(result, [])



if __name__ == '__main__':
    unittest.main()
