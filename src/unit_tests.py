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
import warnings

#We calculate the different parameters in excel, and then test if they are the same as the ones we calculate 
#Using our functions
class monthly_returns_test(unittest.TestCase):

    def test_zero_input_esg(self):
        expected = KeyError
        testcase = pd.DataFrame()
        self.assertRaises(expected, data.esg_score_weight, testcase, np.array([[1/3,1/3,1/3]]))

    def test_zero_input_prices(self):
        expected = KeyError
        testcase = pd.DataFrame()
        self.assertRaises(expected, data.stock_monthly_close, testcase)

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
        expected = np.array([[1., 0., 0.]])
        npt.assert_almost_equal(expected,efficient_frontier.weights_of_portfolio(testcase[['MMM','AOS','ABT']],efficient_frontier.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)))
    
    def test_of_expected_returns_no_short_selling(self):
        expected = 0.0268
        testcase = pd.read_csv('../data/test/test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,efficient_frontier.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][1], places=4, msg=None, delta=None)

    def test_of_expected_risk_no_short_selling(self):
        expected = 0.0517
        testcase = pd.read_csv('../data/test/test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,efficient_frontier.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][0], places=4, msg=None, delta=None)

    def test_of_sharpe_ratio_no_short_selling(self):
        expected =  0.517347 
        testcase = pd.read_csv('../data/test/test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,efficient_frontier.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][1]/efficient_frontier.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(0,1), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][0], places=5, msg=None, delta=None)
    
    def test_of_expected_returns_with_short_selling(self):
        expected = 0.041845
        testcase = pd.read_csv('../data/test/test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,efficient_frontier.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(-1,2), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][1], places=4, msg=None, delta=None)

    def test_of_expected_risk_with_short_selling(self):
        expected = 0.068691
        testcase = pd.read_csv('../data/test/test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,efficient_frontier.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(-1,2), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][0], places=4, msg=None, delta=None)

    def test_of_sharpe_ratio_with_short_selling(self):
        expected =   0.6091821 
        testcase = pd.read_csv('../data/test/test_prices.csv',index_col=['Date'])
        self.assertAlmostEqual(expected,efficient_frontier.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(-1,2), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][1]/efficient_frontier.efficient_frontier_solo(data.pct_returns_from_prices(testcase[['MMM','AOS','ABT']]),Bounds(-1,2), "No_extra_constraint",'2013-04-01','2014-03-01', 0.2, 0.1,'monthly', False)[0][0], places=4, msg=None, delta=None)
   
    def test_of_beta(self):
        expected = 1.8088
        testcase = data.pct_returns_from_prices(pd.read_csv('../data/test/test_prices.csv',index_col=['Date']))
        self.assertAlmostEqual(expected,capm.calculate_portfolio_beta(testcase,testcase[['MMM','AOS','ABT']],pd.DataFrame(np.array([[0.33,0.33,0.33]])),'SP500'), places=3, msg=None, delta=None)

    def test_of_capm(self):
        expected = 0.0204076
        testcase = data.pct_returns_from_prices(pd.read_csv('../data/test/test_prices.csv',index_col=['Date']))
        self.assertAlmostEqual(expected,capm.capm_calc(testcase['SP500'].mean(),1.8087630,0.01), places=4, msg=None, delta=None)

    def test_of_jensens_alpha(self):
        expected = 0.051888
        testcase = data.pct_returns_from_prices(pd.read_csv('../data/test/test_prices.csv',index_col=['Date']))
        self.assertAlmostEqual(expected, capm.jensens_alpha(0.0179752,0.069864), places=4, msg=None, delta=None)
    def setUp(self):
        testcase = pd.read_excel('../data/ESG_DATA_S&P500.xlsx')
        testcase = testcase[(testcase["stock_symbol"] == 'MMM') |(testcase['stock_symbol'] == 'AOS') | (testcase['stock_symbol']=='ABT')]
        self.correct_strategy = {'df': testcase, 'weigh**-ts': np.array([1/3,1/3,1/3]), 
                                                        'min_esg_score': 0,
                                                        'max_esg_score': 2000, 
                                                        'bounds': Bounds(-1,2), 
                                                        'sharpe_type': "No_extra_constraint", 
                                                        'wanted_return': 0.20, 
                                                        'maximum_risk': 0.10, 
                                                        'rebalancing_freq': 'monthly', 
                                                        'risk_free_rate': 0.01, 
                                                        }


    def test_of_backtesting_portfolio_actual_returns_cml(self):
        expected = 0.097149
        self.assertAlmostEqual(expected, 
                               backtesting.backtesting(self.correct_strategy.copy(), 
                                                        'monthly', 
                                                        6, 
                                                        '2014-02-01',
                                                        '2014-07-01',
                                                        0,
                                                        10,
                                                        '^GSPC',False,'Close')['portfolio_actual_returns_cmle'][0][0], places=4, msg=None, delta=None)
    
    def test_of_backtesting_pct_returns_portfolio(self):
        expected = 0.069864
        self.assertAlmostEqual(expected, 
                               backtesting.backtesting(self.correct_strategy.copy(), 
                                                        'monthly', 
                                                        6, 
                                                        '2014-02-01',
                                                        '2014-07-01',
                                                        0,
                                                        10,
                                                        '^GSPC',False,'Close')['portfolio_actual_returns'][0][0], places=4, msg=None, delta=None)
    
    def test_of_backtesting_pct_returns_sp500(self):
        expected = 0.07740
        self.assertAlmostEqual(expected, 
                               backtesting.backtesting(self.correct_strategy.copy(), 
                                                        'monthly', 
                                                        6, 
                                                        '2014-02-01',
                                                        '2014-07-01',
                                                        0,
                                                        10,
                                                        '^GSPC',False,'Close')['pct_returns_sp500'][0], places=4, msg=None, delta=None)
    
    def test_of_backtesting_capm(self):
        expected = 0.017986
        self.assertAlmostEqual(expected, 
                               backtesting.backtesting(self.correct_strategy.copy(), 
                                                        'monthly', 
                                                        6, 
                                                        '2014-02-01',
                                                        '2014-07-01',
                                                        0,
                                                        10,
                                                        '^GSPC',False,'Close')['capm_for_portfolio'][0], places=4, msg=None, delta=None)        
    
    def test_of_backtesting_beta(self):
        expected = 1.388915771
        self.assertAlmostEqual(expected, 
                               backtesting.backtesting(self.correct_strategy.copy(), 
                                                        'monthly', 
                                                        6, 
                                                        '2014-02-01',
                                                        '2014-07-01',
                                                        0,
                                                        10,
                                                        '^GSPC',False,'Close')['betas_of_portfolios'][0], places=3, msg=None, delta=None)
    def test_of_backtesting_beta(self):
        expected = 1.388915771
        self.assertAlmostEqual(expected, 
                               backtesting.backtesting(self.correct_strategy.copy(), 
                                                        'monthly', 
                                                        6, 
                                                        '2014-02-01',
                                                        '2014-07-01',
                                                        0,
                                                        10,
                                                        '^GSPC',False,'Close')['betas_of_portfolios'][0], places=3, msg=None, delta=None)
    def test_strategy_is_not_dict(self):
        with self.assertRaises(TypeError):
            backtesting.backtesting('not a dict', 
                                    'monthly', 
                                    6, 
                                    '2014-02-01',
                                    '2014-07-01',
                                    0,
                                    10,
                                    '^GSPC',False,'Close')
    def test_missing_key_in_strategy(self):
        wrong_strat = self.correct_strategy.copy()
        del wrong_strat['df']
        with self.assertRaises(ValueError):
            backtesting.backtesting(wrong_strat, 
                                    'monthly', 
                                    6, 
                                    '2014-02-01',
                                    '2014-07-01',
                                    0,
                                    10,
                                    '^GSPC',False,'Close') 
    def test_wrong_esg(self):    
        wrong_strat = self.correct_strategy.copy()
        wrong_strat['min_esg_score'] = 2000
        wrong_strat['max_esg_score'] = 0
        with self.assertRaises(ValueError):
            backtesting.backtesting(wrong_strat, 
                                    'monthly', 
                                    6, 
                                    '2014-02-01',
                                    '2014-07-01',
                                    0,
                                    10,
                                    '^GSPC',False,'Close') 
    def test_wrong_rebalancing(self):
        with self.assertRaises(ValueError):
            backtesting.backtesting(self.correct_strategy.copy(), 
                                    2, 
                                    6, 
                                    '2014-02-01',
                                    '2014-07-01',
                                    0,
                                    10,
                                    '^GSPC',False,'Close') 
    def test_wrong_rebalancing_int(self):
        with self.assertRaises(TypeError):
            backtesting.backtesting(self.correct_strategy.copy(), 
                                    'monthly', 
                                    '6', 
                                    '2014-02-01',
                                    '2014-07-01',
                                    0,
                                    10,
                                    '^GSPC',False,'Close') 
    def test_low_rebalancing_int(self):
        with self.assertRaises(ValueError):
            backtesting.backtesting(self.correct_strategy.copy(), 
                                    'monthly', 
                                    0, 
                                    '2014-02-01',
                                    '2014-07-01',
                                    0,
                                    10,
                                    '^GSPC',False,'Close') 
    def test_wrong_date_string1(self):
        with self.assertRaises(TypeError):
            backtesting.backtesting(self.correct_strategy.copy(), 
                                    'monthly', 
                                    '6', 
                                    2,
                                    '2014-07-01',
                                    0,
                                    10,
                                    '^GSPC',False,'Close')
    def test_wrong_date_string2(self):
        with self.assertRaises(TypeError):
            backtesting.backtesting(self.correct_strategy.copy(), 
                                    'monthly', 
                                    '6', 
                                    '2014-07-01',
                                    1,
                                    0,
                                    10,
                                    '^GSPC',False,'Close')
    def test_wrong_date_order_warning(self):
        with self.assertRaises(ValueError):
            backtesting.backtesting(self.correct_strategy.copy(), 
                                    'monthly', 
                                    0, 
                                    '2014-07-01',
                                    '2014-02-01',
                                    0,
                                    10,
                                    '^GSPC',False,'Close') 
    def test_wrong_covariance_window(self):
        with self.assertRaises(TypeError):
            backtesting.backtesting(self.correct_strategy.copy(), 
                                    'monthly', 
                                    2, 
                                    '2014-02-01',
                                    '2014-07-01',
                                    '2',
                                    '2',
                                    '^GSPC',False,'Close')
    def test_no_covariance_window(self):
        with self.assertRaises(ValueError):
            backtesting.backtesting(self.correct_strategy.copy(), 
                                    'monthly', 
                                    1, 
                                    '2014-02-01',
                                    '2014-07-01',
                                    0,
                                    0,
                                    '^GSPC',False,'Close') 
    
    def test_market_name(self):
        with self.assertRaises(TypeError):
            backtesting.backtesting(self.correct_strategy.copy(), 
                                    'monthly', 
                                    1, 
                                    '2014-02-01',
                                    '2014-07-01',
                                    1,
                                    2,
                                    1,False,'Close') 
    def test_closing_name(self):
        with self.assertRaises(ValueError):
                        backtesting.backtesting(self.correct_strategy.copy(), 
                                    'monthly', 
                                    1, 
                                    '2014-02-01',
                                    '2014-07-01',
                                    1,
                                    2,
                                    '^GSPC',False,'closing') 
    def test_no_future_data(self):
        with self.assertRaises(ValueError):
                        backtesting.backtesting(self.correct_strategy.copy(), 
                                    'monthly', 
                                    6, 
                                    '2014-02-01',
                                    '2023-06-01',
                                    1,
                                    2,
                                    '^GSPC',False,'Close') 

                        
            
            

            
            


        

unittest.main()

