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