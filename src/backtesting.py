from src import efficient_frontier
from src import capm
from src import data
from src import portfolio
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
import warnings
# Suppress the FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning) #We get a weird warning, a bug with pandas according to their github

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


def backtesting(strategy, monthly_or_yearly_rebalancing,rebalancing_freq,start_date,end_date,covariance_window,market_name):
    """ This function makes a backtest ...
      
    In this function

    :param
    :param 
    :param 
    :param 
    :returns:
    """
    covariance_window_time_delta = relativedelta(years=covariance_window) #The time delta for the covariance window
    esg_data = data.esg_score_weight(strategy['df'],strategy['weights'],strategy['min_esg_score'],strategy['max_esg_score'])
    full_data = data.stock_monthly_close(esg_data,[pd.Timestamp(start_date-covariance_window_time_delta),pd.Timestamp(end_date+relativedelta(years=rebalancing_freq,months=1))]) #Making sure we download data from the first covariance date, until the last date we sell our stocks
    prices,esgdata = data.seperate_full_data(full_data)
    pct_returns = data.pct_returns_from_prices(prices)
    stock_data_download = yf.download(market_name, start=strategy['start_year']-covariance_window_time_delta, end=end_date+relativedelta(years=rebalancing_freq+1), interval='1mo', progress=False) #Downloads the market stock close for same window.
    stock_data_download = stock_data_download[['Close']].rename(columns={'Close': market_name})
    stock_data = pd.concat([prices,stock_data_download], axis = 1)
    pct = data.pct_returns_from_prices(stock_data)[start_date-covariance_window_time_delta:end_date+relativedelta(years=rebalancing_freq+1)]
    pct.index =pct.index.tz_localize(None)

    listparameters = []
    list_of_port_weights =[]
    list_of_port_esg_scores = []
    list_of_port_allocations = []
    list_of_cmle_returns = []
    list_of_portfolio_actual_returns = []
    sp500_list = []
    list_of_pct_returns_sp500 = []
    betas_of_portfolios = []
    capm_for_portfolio = []
    list_of_cml_allocations = []
    list_of_return_dates = []
    i = 0

    if monthly_or_yearly_rebalancing == 'yearly':
        delta = relativedelta(years=rebalancing_freq)
        while (start_date <= end_date):
            listparameters.append(portfolio.efficient_frontier_solo(pct_returns,
                            strategy['Bounds1'],
                            strategy['sharpe_type'],
                            start_date-covariance_window_time_delta,#The start date for portfolio optimization will be the start date minus the covariance window
                            start_date,#The end date for portfolio optimization will be the start date minus the covariance window
                            strategy['Wanted_return'],
                            strategy['maximum_risk'],
                            strategy['rebalancing_freq'])) 
            list_of_port_weights.append(efficient_frontier.weights_of_portfolio(prices,listparameters[i]))
            list_of_port_esg_scores.append(portfolio.esg_score_of_portfolio(list_of_port_weights[i],esgdata.head(1)))
            list_of_port_allocations.append(portfolio.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[1])
            list_of_cmle_returns.append(portfolio.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[0])
            list_of_cml_allocations.append(portfolio.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[1])
            betas_of_portfolios.append(capm.calculate_portfolio_beta(pct,prices[start_date-covariance_window_time_delta:start_date],list_of_port_weights[i],market_name))            
            capm_for_portfolio.append(capm.capm_calc(pct[market_name][start_date-covariance_window_time_delta:start_date].tz_localize(None).mean(),strategy['risk_free_rate'],betas_of_portfolios[i]))
            list_of_portfolio_actual_returns.append(((portfolio.portfolio_return(prices.loc[start_date+delta],list_of_port_weights[i].T)[0])
                                                     *(1-list_of_cml_allocations[i][0])-(portfolio.portfolio_return(prices.loc[start_date],list_of_port_weights[i].T)[0])
                                                     *(1-list_of_cml_allocations[i][0]))/(portfolio.portfolio_return(prices.loc[start_date],list_of_port_weights[i].T)[0])
                                                     *(1-list_of_cml_allocations[i][0])+list_of_cml_allocations[i][0]*strategy['risk_free_rate'])#list of how large a percentage you would have made using your investment strategy
            #List of portfolios actual returns skal laves om så man tager højde cml allokeringen
            list_of_pct_returns_sp500.append((stock_data_download.loc[start_date+delta][0]-stock_data_download.loc[start_date][0])/stock_data_download.loc[start_date][0])#List of how large a percentage you would have made investing everything in the market
            list_of_return_dates.append((start_date+delta).strftime('%Y/%m/%d'))
            start_date += delta
            i += 1
        return(list_of_port_weights,list_of_port_esg_scores,list_of_port_allocations,betas_of_portfolios,list_of_cmle_returns,list_of_portfolio_actual_returns,list_of_pct_returns_sp500,list_of_return_dates)
    
    elif monthly_or_yearly_rebalancing == 'monthly':
        delta = relativedelta(months=rebalancing_freq)
        while (start_date <= end_date):
            listparameters.append(portfolio.efficient_frontier_solo(pct_returns,
                            strategy['Bounds1'],
                            strategy['sharpe_type'],
                            start_date-covariance_window_time_delta,#The start date for portfolio optimization will be the start date minus the covariance window
                            start_date,#The end date for portfolio optimization will be the start date minus the covariance window
                            strategy['Wanted_return'],
                            strategy['maximum_risk'],
                            strategy['rebalancing_freq'])) 
            list_of_port_weights.append(efficient_frontier.weights_of_portfolio(prices,listparameters[i]))
            list_of_port_esg_scores.append(portfolio.esg_score_of_portfolio(list_of_port_weights[i],esgdata.head(1)))
            list_of_port_allocations.append(portfolio.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[1])
            list_of_cmle_returns.append(portfolio.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[0])
            list_of_cml_allocations.append(portfolio.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[1])
            betas_of_portfolios.append(capm.calculate_portfolio_beta(pct,prices[start_date-covariance_window_time_delta:start_date],list_of_port_weights[i],market_name))            
            capm_for_portfolio.append(capm.capm_calc(pct[market_name][start_date-covariance_window_time_delta:start_date].tz_localize(None).mean(),strategy['risk_free_rate'],betas_of_portfolios[i]))

            list_of_portfolio_actual_returns.append(((portfolio.portfolio_return(prices.loc[start_date+delta],list_of_port_weights[i].T)[0])
                                                     *(1-list_of_cml_allocations[i][0])-(portfolio.portfolio_return(prices.loc[start_date],list_of_port_weights[i].T)[0])
                                                     *(1-list_of_cml_allocations[i][0]))/(portfolio.portfolio_return(prices.loc[start_date],list_of_port_weights[i].T)[0])
                                                     *(1-list_of_cml_allocations[i][0])+list_of_cml_allocations[i][0]*strategy['risk_free_rate'])#list of how large a percentage you would have made using your investment strategy
            #List of portfolios actual returns skal laves om så man tager højde cml allokeringen
            list_of_pct_returns_sp500.append((stock_data_download.loc[start_date+delta][0]-stock_data_download.loc[start_date][0])/stock_data_download.loc[start_date][0])#List of how large a percentage you would have made investing everything in the market
            list_of_return_dates.append((start_date+delta).strftime('%Y/%m/%d'))
            print(list_of_portfolio_actual_returns)
            print(list_of_pct_returns_sp500)
            start_date += delta
            i += 1

        return(list_of_port_weights,list_of_port_esg_scores,list_of_port_allocations,betas_of_portfolios,list_of_return_dates,list_of_cmle_returns,list_of_pct_returns_sp500)      
    
    else:
        print('You can only call this function with monthly or yearly')
        return(False)
    
