import unittest

#functions
from efficient_frontier import *

def someFunction(x):
    return x

class Test(unittest.TestCase):

    def test1(self):
        result = someFunction(0)
        self.assertEqual(result, 0)

    def test2(self):
        result = calculate_efficient_frontier_esg()
        self.assertEqual(result, None)


if __name__ == '__main__':
    unittest.main()