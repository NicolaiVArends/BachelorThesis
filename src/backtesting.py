from src import efficient_frontier
from src import capm
from src import data
from src import portfolio
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='pandas')
# Suppress the FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning) #We get a weird warning, a bug with pandas according to their github


def backtesting(strategy, monthly_or_yearly_rebalancing,
                rebalancing_freq,start_date,end_date,
                covariance_window_yearly,covariance_window_monthly,
                market_name,benoit_wolfe = True): #,covariance_window_monthly
    """ This function makes a backtest given a strategy, rebalacing informations, information of data period, window size and benchmark market name.
      
    In this function

    :param
    :param 
    :param 
    :param 
    :returns:
    """
    
    assert isinstance(strategy, dict), "strategy should be a dictionary"
    
    required_keys = ['df', 'weights', 'min_esg_score', 'max_esg_score', 'bounds', 'sharpe_type', 'wanted_return', 'maximum_risk', 'rebalancing_freq', 'risk_free_rate']
    
    missing_keys = [k for k in required_keys if k not in strategy.keys()]
    if monthly_or_yearly_rebalancing == 'yearly':
        assert pd.to_datetime(end_date) + pd.DateOffset(years=rebalancing_freq, months = 1) <= pd.Timestamp(datetime.now()), "The end_date and rebalancing dates should not be in the future"
    elif monthly_or_yearly_rebalancing == 'monthly':
        assert pd.to_datetime(end_date) + pd.DateOffset(months=rebalancing_freq + 1) <= pd.Timestamp(datetime.now()), "The end_date and rebalancing dates should not be in the future"

    
    
    assert not missing_keys, "The strategy dictionary is missing the following keys: {}".format(', '.join(missing_keys))
    assert strategy['min_esg_score']<strategy['max_esg_score'], "The minimum ESG score cannot be greater or equal to the maximum esg score" 
    assert monthly_or_yearly_rebalancing in ["monthly", "yearly"], "monthly_or_yearly_rebalancing should be 'monthly' or 'yearly'"
    assert isinstance(rebalancing_freq, int), "rebalancing_freq should be an integer"
    assert rebalancing_freq > 0, "rebalancing_freq should be greater than zero"
    assert isinstance(start_date, (str, pd.Timestamp)), "start_date should be a string or a pandas Timestamp object"
    assert isinstance(end_date, (str, pd.Timestamp)), "end_date should be a string or a pandas Timestamp object"
    assert isinstance(covariance_window_yearly, int), "covariance_window_yearly should be an integer"
    assert isinstance(covariance_window_monthly, int), "covariance_window_monthly should be an integer"
    assert covariance_window_yearly + covariance_window_monthly > 0, "covariance_window should be greater than zero"
    assert isinstance(market_name, str), "market_name should be a string"

    if isinstance(start_date, str):
        start_date = parse(start_date)

    if isinstance(end_date, str):
        end_date = parse(end_date)

    assert start_date <= end_date, "start_date should be less than or equal to end_date"
    covariance_window_time_delta = relativedelta(years=covariance_window_yearly,months=covariance_window_monthly) #The time delta for the covariance window
    
    esg_data = data.esg_score_weight(strategy['df'],strategy['weights'],strategy['min_esg_score'],strategy['max_esg_score'])
    #print(esg_data)
    #print([pd.Timestamp(start_date-covariance_window_time_delta),pd.Timestamp(end_date+relativedelta(years=rebalancing_freq,months=1))])
    if monthly_or_yearly_rebalancing == 'yearly':
        full_data = data.stock_monthly_close(esg_data,[pd.Timestamp(start_date-covariance_window_time_delta),pd.Timestamp(end_date+relativedelta(years=rebalancing_freq,months=1))]) 
    if monthly_or_yearly_rebalancing == 'monthly':
        full_data = data.stock_monthly_close(esg_data,[pd.Timestamp(start_date-covariance_window_time_delta),pd.Timestamp(end_date+relativedelta(months=rebalancing_freq+1))]) 

#Making sure we download data from the first covariance date, until the last date we sell our stocks
    prices,esgdata = data.seperate_full_data(full_data)
    pct_returns = data.pct_returns_from_prices(prices)
    if monthly_or_yearly_rebalancing == 'yearly':
        stock_data_download = yf.download(market_name, start=start_date-covariance_window_time_delta, end=end_date+relativedelta(years=rebalancing_freq+1,months=1), interval='1mo', progress=False) #Downloads the market stock close for same window.
    elif monthly_or_yearly_rebalancing == 'monthly':
        stock_data_download = yf.download(market_name, start=start_date-covariance_window_time_delta, end=end_date+relativedelta(months=rebalancing_freq+1), interval='1mo', progress=False)

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
    list_of_portfolio_actual_returns_cmle = []
    list_of_pct_returns_sp500 = []
    betas_of_portfolios = []
    capm_for_portfolio = []
    list_of_cml_allocations = []
    list_of_return_dates = []
    cumulative_market_return = 1
    cumulative_cml_return = 1
    cumulative_portfolio_return = 1
    cumulative_market_return_list = []
    cumulative_cml_return_list = []
    cumulative_portfolio_return_list = []

    i = 0

    if monthly_or_yearly_rebalancing == 'yearly':
        delta = relativedelta(years=rebalancing_freq)
        while (start_date <= end_date):
            listparameters.append(portfolio.efficient_frontier_solo(pct_returns,
                            strategy['bounds'],
                            strategy['sharpe_type'],
                            start_date-covariance_window_time_delta,#The start date for portfolio optimization will be the start date minus the covariance window
                            start_date,#The end date for portfolio optimization will be the start date minus the covariance window
                            strategy['wanted_return'],
                            strategy['maximum_risk'],
                            strategy['rebalancing_freq'],
                            benoit_wolfe)) 
            
            
            list_of_port_weights.append(efficient_frontier.weights_of_portfolio(prices,listparameters[i]))

            list_of_port_esg_scores.append(portfolio.esg_score_of_portfolio(list_of_port_weights[i],esgdata.head(1)))

            list_of_port_allocations.append(portfolio.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[1])

            list_of_cmle_returns.append(portfolio.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[0])

            list_of_cml_allocations.append(portfolio.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[1])

            betas_of_portfolios.append(capm.calculate_portfolio_beta(pct[start_date-covariance_window_time_delta:start_date],prices[start_date-covariance_window_time_delta:start_date],list_of_port_weights[i],market_name))     

            capm_for_portfolio.append(capm.capm_calc(pct[market_name][start_date-covariance_window_time_delta:start_date].tz_localize(None).mean(),betas_of_portfolios[i],strategy['risk_free_rate']))
            
            list_of_portfolio_actual_returns.append((portfolio.portfolio_return(prices.loc[start_date+delta],list_of_port_weights[i])
                                                    -portfolio.portfolio_return(prices.loc[start_date],list_of_port_weights[i]))/portfolio.portfolio_return(prices.loc[start_date],list_of_port_weights[i]))
            
            list_of_portfolio_actual_returns_cmle.append(list_of_cml_allocations[i][0]*list_of_portfolio_actual_returns[i]+((1-list_of_cml_allocations[i][0])*strategy['risk_free_rate']))

            list_of_pct_returns_sp500.append((stock_data_download.loc[start_date+delta][0]-stock_data_download.loc[start_date][0])/stock_data_download.loc[start_date][0])
            #List of how large a percentage you would have made investing everything in the market
            list_of_return_dates.append((start_date+delta).strftime('%Y/%m/%d'))

            cumulative_market_return *=(1+list_of_pct_returns_sp500[i])

            cumulative_cml_return *= (1+list_of_portfolio_actual_returns_cmle[i])

            cumulative_portfolio_return *= (1+list_of_portfolio_actual_returns[i])

            cumulative_cml_return_list.append(cumulative_cml_return-1)

            cumulative_market_return_list.append(cumulative_market_return-1)

            cumulative_portfolio_return_list.append(cumulative_portfolio_return-1)
            





            start_date += delta
            i += 1
        results = {
            'portfolio_weights': list_of_port_weights,
            'portfolio_esg_scores': list_of_port_esg_scores,
            'portfolio_allocations': list_of_port_allocations,
            'betas_of_portfolios': betas_of_portfolios,
            'capm_for_portfolio': capm_for_portfolio,
            'cmle_returns': list_of_cmle_returns,
            'portfolio_actual_returns': list_of_portfolio_actual_returns,
            'pct_returns_sp500': list_of_pct_returns_sp500,
            'portfolio_actual_returns_cmle': list_of_portfolio_actual_returns_cmle,
            'return_dates': list_of_return_dates, 
            'cumulative_cml_return_list': cumulative_cml_return_list,
            'cumulative_market_return_list': cumulative_market_return_list,
            'cumulative_portfolio_return_list': cumulative_portfolio_return_list
        }
        return(results)
    
    elif monthly_or_yearly_rebalancing == 'monthly':
        delta = relativedelta(months=rebalancing_freq)
        while (start_date <= end_date):
            listparameters.append(portfolio.efficient_frontier_solo(pct_returns,
                            strategy['bounds'],
                            strategy['sharpe_type'],
                            start_date-covariance_window_time_delta,#The start date for portfolio optimization will be the start date minus the covariance window
                            start_date,#The end date for portfolio optimization will be the start date minus the covariance window
                            strategy['wanted_return'],
                            strategy['maximum_risk'],
                            strategy['rebalancing_freq'],
                            benoit_wolfe)) 
            
            
            list_of_port_weights.append(efficient_frontier.weights_of_portfolio(prices,listparameters[i]))    

            list_of_port_esg_scores.append(portfolio.esg_score_of_portfolio(list_of_port_weights[i],esgdata.head(1)))

            list_of_port_allocations.append(portfolio.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[1])

            list_of_cmle_returns.append(portfolio.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[0])

            list_of_cml_allocations.append(portfolio.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[1])

            betas_of_portfolios.append(capm.calculate_portfolio_beta(pct[start_date-covariance_window_time_delta:start_date],prices[start_date-covariance_window_time_delta:start_date],list_of_port_weights[i],market_name))    

            capm_for_portfolio.append(capm.capm_calc(pct[market_name][start_date-covariance_window_time_delta:start_date].tz_localize(None).mean(),betas_of_portfolios[i],strategy['risk_free_rate']))

            list_of_portfolio_actual_returns.append((portfolio.portfolio_return(prices.loc[start_date+delta],list_of_port_weights[i])
                                                    -portfolio.portfolio_return(prices.loc[start_date],list_of_port_weights[i]))/portfolio.portfolio_return(prices.loc[start_date],list_of_port_weights[i]))
            
            list_of_portfolio_actual_returns_cmle.append(list_of_cml_allocations[i][0]*list_of_portfolio_actual_returns[i]+((1-list_of_cml_allocations[i][0])*strategy['risk_free_rate']))

            list_of_pct_returns_sp500.append((stock_data_download.loc[start_date+delta][0]-stock_data_download.loc[start_date][0])/stock_data_download.loc[start_date][0])#List of how large a percentage you would have made investing everything in the market

            list_of_return_dates.append((start_date+delta).strftime('%Y/%m/%d'))

            cumulative_market_return *=(1+list_of_pct_returns_sp500[i])

            cumulative_cml_return *= (1+list_of_portfolio_actual_returns_cmle[i])

            cumulative_portfolio_return *= (1+list_of_portfolio_actual_returns[i])

            cumulative_cml_return_list.append(cumulative_cml_return-1)

            cumulative_market_return_list.append(cumulative_market_return-1)

            cumulative_portfolio_return_list.append(cumulative_portfolio_return-1)
            





            start_date += delta
            i += 1
        results = {
            'portfolio_weights': list_of_port_weights,
            'portfolio_esg_scores': list_of_port_esg_scores,
            'portfolio_allocations': list_of_port_allocations,
            'betas_of_portfolios': betas_of_portfolios,
            'capm_for_portfolio': capm_for_portfolio,
            'cmle_returns': list_of_cmle_returns,
            'portfolio_actual_returns': list_of_portfolio_actual_returns,
            'pct_returns_sp500': list_of_pct_returns_sp500,
            'portfolio_actual_returns_cmle': list_of_portfolio_actual_returns_cmle,
            'return_dates': list_of_return_dates, 
            'cumulative_cml_return_list': cumulative_cml_return_list,
            'cumulative_market_return_list': cumulative_market_return_list,
            'cumulative_portfolio_return_list': cumulative_portfolio_return_list
        }
        return(results)
    else:
        raise Exception('You can only call this function with monthly or yearly')
    
