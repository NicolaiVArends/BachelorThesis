import sys
sys.path.append("../")

from src import portfolio
from src import data
from src import efficient_frontier
from src import capm
from src import backtesting

import unittest
import pandas as pd
import numpy as np
import numpy.testing as npt
from scipy.optimize import Bounds

#We calculate the different parameters in excel, and then test if they are the same as the ones we calculate 
#Using our functions
class monthly_returns_test(unittest.TestCase):
    def test_of_beta(self):
        expected = 1.8088
        testcase = data.pct_returns_from_prices(pd.read_csv('../data/test/test_prices.csv',index_col=['Date']))
        self.assertAlmostEqual(expected,capm.calculate_portfolio_beta(testcase,testcase[['MMM','AOS','ABT']],pd.DataFrame(np.array([[0.33,0.33,0.33]])),'SP500'), places=5, msg=None, delta=None)

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
        testcase = pd.read_csv('../data/test/test_prices.csv',index_col=['Date'])
        print(testcase.keys(),'fisk')
        expected = np.array([[1., 0., 0.]])
        npt.assert_almost_equal(expected,efficient_frontier.weights_of_portfolio(testcase[['MMM','AOS','ABT']],portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)))
    
    def test_of_expected_returns_no_short_selling(self):
        expected = 0.0268
        testcase = pd.read_csv('../data/test/test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][1], places=4, msg=None, delta=None)

    def test_of_expected_risk_no_short_selling(self):
        expected = 0.0517
        testcase = pd.read_csv('../data/test/test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][0], places=4, msg=None, delta=None)

    def test_of_sharpe_ratio_no_short_selling(self):
        expected =  0.517347 
        testcase = pd.read_csv('../data/test/test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][1]/portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][0], places=5, msg=None, delta=None)
    
    def test_of_expected_returns_with_short_selling(self):
        expected = 0.041845
        testcase = pd.read_csv('../data/test/test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(-1,2), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][1], places=4, msg=None, delta=None)

    def test_of_expected_risk_with_short_selling(self):
        expected = 0.068691
        testcase = pd.read_csv('../data/test/test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(-1,2), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][0], places=4, msg=None, delta=None)

    def test_of_sharpe_ratio_with_short_selling(self):
        expected =   0.6091821 
        testcase = pd.read_csv('../data/test/test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(-1,2), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][1]/portfolio.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(-1,2), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][0], places=4, msg=None, delta=None)
   
    def test_of_beta(self):
        expected = 1.8088
        testcase = data.pct_returns_from_prices(pd.read_csv('../data/test/test_prices.csv',index_col=['Date']))
        self.assertAlmostEqual(expected,capm.calculate_portfolio_beta(testcase,testcase[['MMM','AOS','ABT']],pd.DataFrame(np.array([[0.33,0.33,0.33]])),'SP500'), places=3, msg=None, delta=None)

        #expected = 0.0204076
    def test_of_capm(self):
        expected = 0.0204076
        testcase = data.pct_returns_from_prices(pd.read_csv('../data/test/test_prices.csv',index_col=['Date']))
        self.assertAlmostEqual(expected,capm.capm_calc(testcase['SP500'].mean(),1.8087630,0.01), places=4, msg=None, delta=None)

    def test_of_jensens_alpha(self):
        expected = 0.2009064
        testcase = data.pct_returns_from_prices(pd.read_csv('../data/test/test_prices2.csv',index_col=['Date']))
        self.assertAlmostEqual(expected, capm.jensens_alpha(capm.capm_calc(testcase['SP500'].mean(),1.8087630,0.01), np.sum(portfolio.portfolio_return(testcase[['MMM','AOS','ABT']], pd.DataFrame(np.array([[0.33,0.33,0.33]]))))), places=4, msg=None, delta=None)

    #def test_of_backtesting_portfolio_actual_returns_cml(self):
    #    expected = []
    #    testcase = pd.read_excel('../data/test/test_esg_sp500_data.xlsx')
    #    self.assertAlmostEqual(expected, backtesting.backtesting({'df': testcase, 'weights': np.array([1/3,1/3,1/3]), 'min_esg_score': 1200, 'max_esg_score': 2000, 'Bounds1': Bounds(0,1), 'sharpe_type': "No_extra_constraint", 'Wanted_return': 0.20, 'maximum_risk': 0.10, 'rebalancing_freq': 'monthly', 'risk_free_rate': 0.01, 'start_year' : pd.Timestamp(np.datetime64('2000-01-01')), 'end_year' : pd.Timestamp(np.datetime64('2022-01-01'))} , 'yearly', 1, '2015-01-01','2022-01-01',10,'SPY')[7])

    #def test_of_backtesting_pct_returns_sp500(self):
    #    expected = []
    #    testcase = pd.read_excel('../data/test/test_esg_sp500_data.xlsx')
    #    self.assertAlmostEqual(expected, backtesting.backtesting({'df': testcase, 'weights': np.array([1/3,1/3,1/3]), 'min_esg_score': 1200, 'max_esg_score': 2000, 'Bounds1': Bounds(0,1), 'sharpe_type': "No_extra_constraint", 'Wanted_return': 0.20, 'maximum_risk': 0.10, 'rebalancing_freq': 'monthly', 'risk_free_rate': 0.01, 'start_year' : pd.Timestamp(np.datetime64('2000-01-01')), 'end_year' : pd.Timestamp(np.datetime64('2022-01-01'))} , 'yearly', 1, '2015-01-01','2022-01-01',10,'SPY')[6])

unittest.main()

