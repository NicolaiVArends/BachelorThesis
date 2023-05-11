
import portfolio
import data
import efficient_frontier
import unittest

import pandas as pd
import numpy as np
import numpy.testing as npt
from scipy.optimize import Bounds

#We calculate the different parameters in excel, and then test if they are the same as the ones we calculate 
#Using our functions
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
        expected = np.array([0.026750,0.031154,0.009160])
        npt.assert_almost_equal(expected,data.pct_returns_from_prices(testcase).mean().to_numpy(),decimal = 5)
    def test_of_weights_no_short_selling(self):
        testcase = pd.read_csv(r'C:\Users\toros\OneDrive\Dokumenter\GitHub\BachelorThesis\data\test\test_prices.csv',index_col=['Date'])
        
        expected = np.array([[1., 0., 0.]])
        npt.assert_almost_equal(expected,efficient_frontier.weights_of_portfolio(testcase,portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1)))
    def test_of_expected_returns_no_short_selling(self):
        expected = 0.0268
        testcase = pd.read_csv(r'C:\Users\toros\OneDrive\Dokumenter\GitHub\BachelorThesis\data\test\test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1)[0][1], places=4, msg=None, delta=None)

    def test_of_expected_risk_no_short_selling(self):
        expected = 0.0517
        testcase = pd.read_csv(r'C:\Users\toros\OneDrive\Dokumenter\GitHub\BachelorThesis\data\test\test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1)[0][0], places=4, msg=None, delta=None)

    def test_of_sharpe_ratio_no_short_selling(self):
        expected =  0.517347 
        testcase = pd.read_csv(r'C:\Users\toros\OneDrive\Dokumenter\GitHub\BachelorThesis\data\test\test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1)[0][1]/portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1)[0][0], places=5, msg=None, delta=None)
    
    def test_of_expected_returns_with_short_selling(self):
        expected = 0.041845

        testcase = pd.read_csv(r'C:\Users\toros\OneDrive\Dokumenter\GitHub\BachelorThesis\data\test\test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase),Bounds(-1,2), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1)[0][1], places=4, msg=None, delta=None)

    def test_of_expected_risk_with_short_selling(self):
        expected = 0.068691

        testcase = pd.read_csv(r'C:\Users\toros\OneDrive\Dokumenter\GitHub\BachelorThesis\data\test\test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase),Bounds(-1,2), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1)[0][0], places=4, msg=None, delta=None)

    def test_of_sharpe_ratio_with_short_selling(self):
        expected =   0.6091821 

        testcase = pd.read_csv(r'C:\Users\toros\OneDrive\Dokumenter\GitHub\BachelorThesis\data\test\test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase),Bounds(-1,2), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1)[0][1]/portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase),Bounds(-1,2), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1)[0][0], places=5, msg=None, delta=None)

unittest.main()