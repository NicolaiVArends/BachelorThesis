from src import filter_assets
import data
import portfolio
import pandas as pd

def rolling_window_backtesting(prices: pd.DataFrame, 
                               weigths_of_each_portfolio: pd.DataFrame, 
                               window_size = 10, 
                               backtest_start_year=2013):
    """ This function makes a backtest/performancetest using the optimal weight allocation in-sample and the returns data for the next year out-sample.
      
    In this function we takes the returns price data for the portfolio assets and the weight allocation for each rolling window of the portfolio.

    :param prices: Returns or price data for the portfolio assets
    :param weigths_of_each_portfolio: Optimal weight allocations as a dataframe for each calculated rolling window
    :param window_size: Size of the window in year, default is 10
    :param backtest_start_year: The starting year for when the first out-sample data is calculated for, default is 2013
    :returns: A tuple which contains two list of the calculated backtest: One for portfolio return backtest out-sample and one for portfolio risk backtest out-sample
    """
    backtest_portfolio_return = []
    backtest_portfolio_risk = []
    for i in range(0, window_size):
        prices_out_sample = prices.loc['{}-01-01'.format(str(backtest_start_year+i)):'{}-12-01'.format(str(backtest_start_year+i))]
        pct_returns_out_sample = data.pct_returns_from_prices(prices_out_sample)
        return_out_sample = portfolio.mean_return_annual(pct_returns_out_sample)
        port_ret_out_sample = portfolio.portfolio_return(return_out_sample, weigths_of_each_portfolio[i:i+1].T)
        variance_out_sample = portfolio.covariance_matrix_annual(pct_returns_out_sample)
        port_var_out_sample = portfolio.portfolio_std(variance_out_sample, weigths_of_each_portfolio[i:i+1].values.tolist()[0])
        print(f'The average portfolio return in the out-sample data in period from {backtest_start_year+i} to end of {backtest_start_year+i} is {round(port_ret_out_sample[0]*100,3)} % \n The risk in this period for the out-of-sample portfolio is {round(port_var_out_sample,3)}')
        backtest_portfolio_return.append(port_ret_out_sample)
        backtest_portfolio_risk.append(port_var_out_sample)

    return backtest_portfolio_return, backtest_portfolio_risk
