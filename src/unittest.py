import unittest

def someFunction(x):
    return x

class Test(unittest.TestCase):

    def test1(self):
        result = someFunction(0)
        self.assertEqual(result, 0)


if __name__ == '__main__':
    unittest.main()