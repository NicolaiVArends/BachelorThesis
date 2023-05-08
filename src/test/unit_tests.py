from src import data
import unittest
import pandas as pd
import numpy as np

class TestESG_SCORE_WEIGHT(unittest.TestCase):
    def test_basic(self):
        testcase = pd.read_excel(r"C:\Users\Tor Osted\OneDrive\Dokumenter\GitHub\BachelorThesis\All_Code\test_data\test_esg_sp500_data.xlsx")
        expected = pd.read_excel(r"C:\Users\Tor Osted\OneDrive\Dokumenter\GitHub\BachelorThesis\All_Code\test_data\test_esg_sp500_after_test.xlsx").sort_index().sort_index(axis=1).reset_index(drop=True)
        self.assertEqual(data.esg_score_weight(testcase,np.array([1/3,1/3,1/3]),1150).sort_index().sort_index(axis=1).reset_index(drop=True),expected)

unittest.main()

class monthly_returns_test(unittest.TestCase):
    def test_monthly_returns(self):
        testcase = pd.DataFrame(np.array([[104.70999908,  18.85750008,  36.91999817],
       [110.26999664,  19.60000038,  36.66999817],
       [109.34999847,  18.13999939,  34.88000107],
       [117.43000031,  20.65999985,  36.63000107],
       [113.58000183,  21.02499962,  33.33000183],
       [119.41000366,  22.60000038,  33.18999863],
       [125.84999847,  25.82500076,  36.54999924],
       [133.50999451,  27.07500076,  38.18999863],
       [140.25      ,  26.96999931,  38.33000183],
       [128.19000244,  23.61000061,  36.65999985],
       [134.72999573,  24.85000038,  39.77999878]]))
        expected = pd.DataFrame(np.array([np.array([0.026750,0.031154,0.09160])]))
        self.assertAlmostEqual()
    