from src import filter_assets
import data
import portfolio

def rolling_window_backtesting(prices, weigths_of_each_portfolio, window_size = 10, backtest_start_year=2013):
    """
    
    :param:
    :param: 
    :returns: 
    """
    for i in range(0, window_size):
        prices_out_sample = prices.loc['{}-01-01'.format(str(backtest_start_year+i)):'{}-12-01'.format(str(backtest_start_year+i))]
        pct_returns_out_sample = data.pct_returns_from_prices(prices_out_sample)
        return_out_sample = portfolio.mean_return_annual(pct_returns_out_sample)
        port_ret_out_sample = portfolio.portfolio_return(return_out_sample, weigths_of_each_portfolio[i:i+1].T)
        variance_out_sample = portfolio.covariance_matrix_annual(pct_returns_out_sample)
        port_var_out_sample = portfolio.portfolio_std(variance_out_sample, weigths_of_each_portfolio[i:i+1].values.tolist()[0])
        print(f'The average portfolio return in the out-sample data in period from {backtest_start_year+i} to end of {backtest_start_year+i} is {round(port_ret_out_sample[0]*100,3)} % \n The risk in this period for the out-of-sample portfolio is {round(port_var_out_sample,3)}')

    return port_ret_out_sample, port_var_out_sample
