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

def backtesting(strategy, 
                monthly_or_yearly_rebalancing,
                rebalancing_freq,
                start_date,
                end_date,
                covariance_window_yearly,
                covariance_window_monthly,
                market_name,
                ledoit_wolfe = True,
                close_type = 'Adj Close'):
    """ This function makes a backtest given a strategy, rebalacing informations, information of data period, window size and benchmark market name.
      


    :param strategy: A dict containing the investment strategy
    should contain the following keys.
    A dictionary specifying the investment strategy. The dictionary should contain the following keys: 
        - 'df': DataFrame of the market data
        - 'weights': Array of weights for the ESG scores
        - 'min_esg_score': Minimum Environmental, Social, and Governance (ESG) score for the assets
        - 'max_esg_score': Maximum ESG score for the assets
        - 'bounds': Bounds for the weights of the assets
        - 'sharpe_type': Type of Sharpe Ratio to use for the optimization 
        - 'wanted_return': Desired return for the portfolio (Only used for return sharpe optimization)
        - 'maximum_risk': Maximum risk acceptable for the portfolio (Used for CMLE returns and minimize risk portfolio optimization)
        - 'rebalancing_freq': Frequency of portfolio rebalancing
        - 'risk_free_rate': The risk-free rate to use in the portfolio optimization

    :param monthly_or_yearly_rebalancing : str. Frequency of portfolio rebalancing, either 'monthly' or 'yearly'.
    :param rebalancing_freq : int. The number of months or years between each rebalance.

    :param start_date : str or pd.Timestamp. The start date of the backtest period. Can be a string formatted as 'yyyy-mm-dd' or a pandas Timestamp object.
    :param end_date : str or pd.timestamp. The end date of the backtest period. Can be a string formatted as 'yyyy-mm-dd' or a pandas Timestamp object.
    :param covariance_window_yearly : int. The number of years of data to use for calculating the covariance matrix.
    :param  covariance_window_monthly : int. The number of months of data to use for calculating the covariance matrix.

    :param market_name : str. The ticker of the benchmark market to compare the investment strategy to.
    :param ledoit_wolfe : bool, optional. Whether to use the Benoit Wolfe method for portfolio optimization. Default is True.

    :param close_type : str, optional. The type of close prices to use in the calculations. Default is 'Adj Close'.

    :returns: Results a Dict.
    The dict contains the following.
    portfolio_parameters: List of the parameters of each portfolio, for each rebalancing date.
    portfolio_weights: List of the weights of the assets at each rebalancing date
    portfolio_esg_scores: List of the ESG scores of the portfolio at each rebalance date.
    betas_of_portfolios: List of the betas of each portfolio for each rebalancing date
    capm_for_portfolio: List of the expected returns based on the Capital Asset Pricing model returns of the portfolio of each rebalaning date
    cmle_returns: List of the expected returns based on the Capital Market Line
    portfolio_actual_returns: List of the actual returns of the portfolio at the rebalancing date
    pct_returns_sp500: List of the returns of the benchmark market.
    portfolio_actual_returns _cmle: List of the actual returns based on cml allocation
    return_dates: List of the dates of each rebalance
    cumulative_cml_return_list: List of the cumulative cml returns
    cumulative_market_return_list': List of the cumulative market returns
    cumulative_portfolio_return_list: List of the cumulative returns of the portfolio

    """
    
    if not isinstance(strategy, dict):
        raise TypeError("strategy should be a dictionary")
    
    required_keys = ['df', 'weights', 'min_esg_score', 'max_esg_score', 'bounds', 'sharpe_type', 'wanted_return', 'maximum_risk', 'rebalancing_freq', 'risk_free_rate']
    
    missing_keys = [k for k in required_keys if k not in strategy.keys()]
    
    if missing_keys:
        raise ValueError("The strategy dictionary is missing the following keys: {}".format(', '.join(missing_keys)))
    
    if strategy['min_esg_score'] >= strategy['max_esg_score']:
        raise ValueError("The minimum ESG score cannot be greater or equal to the maximum ESG score")
    
    if not isinstance(market_name, str):
        raise TypeError("market_name should be a string")

    if close_type not in ['Adj Close', 'Close']:
        raise ValueError("close_type should be either 'Adj Close' or 'Close'")

    if monthly_or_yearly_rebalancing not in ["monthly", "yearly"]:
        raise ValueError("monthly_or_yearly_rebalancing should be 'monthly' or 'yearly'")

    if not isinstance(rebalancing_freq, int):
        raise TypeError("rebalancing_freq should be an integer")

    if rebalancing_freq <= 0:
        raise ValueError("rebalancing_freq should be greater than zero")

    if not isinstance(start_date, (str, pd.Timestamp)):
        raise TypeError("start_date should be a string or a pandas Timestamp object")

    if not isinstance(end_date, (str, pd.Timestamp)):
        raise TypeError("end_date should be a string or a pandas Timestamp object")


    if isinstance(start_date, str):
        start_date = parse(start_date)

    if isinstance(end_date, str):
        end_date = parse(end_date)

    if start_date > end_date:
        raise ValueError("start_date should be less than or equal to end_date")

    if not isinstance(covariance_window_yearly, int):
        raise TypeError("covariance_window_yearly should be an integer")

    if not isinstance(covariance_window_monthly, int):
        raise TypeError("covariance_window_monthly should be an integer")

    if covariance_window_yearly + covariance_window_monthly <= 0:
        raise ValueError("covariance_window should be greater than zero")
    
    if pd.to_datetime(end_date) + pd.DateOffset(months=rebalancing_freq + 1) > pd.Timestamp(datetime.now()):
            raise ValueError("The end_date and rebalancing dates should not be in the future")

    covariance_window_time_delta = relativedelta(years=covariance_window_yearly,months=covariance_window_monthly) #The time delta for the covariance window
    
    esg_data = data.esg_score_weight(strategy['df'],strategy['weights'],strategy['min_esg_score'],strategy['max_esg_score'])
    #print(esg_data)
    #print([pd.Timestamp(start_date-covariance_window_time_delta),pd.Timestamp(end_date+relativedelta(years=rebalancing_freq,months=1))])
    if monthly_or_yearly_rebalancing == 'yearly':
        full_data = data.stock_monthly_close(esg_data,[pd.Timestamp(start_date-covariance_window_time_delta),pd.Timestamp(end_date+relativedelta(years=rebalancing_freq,months=1))],close_type) 
    if monthly_or_yearly_rebalancing == 'monthly':
        full_data = data.stock_monthly_close(esg_data,[pd.Timestamp(start_date-covariance_window_time_delta),pd.Timestamp(end_date+relativedelta(months=rebalancing_freq+1))],close_type) 

#Making sure we download data from the first covariance date, until the last date we sell our stocks
    prices,esgdata = data.seperate_full_data(full_data)
    pct_returns = data.pct_returns_from_prices(prices)
    if monthly_or_yearly_rebalancing == 'yearly':
        stock_data_download = yf.download(market_name, start=start_date-covariance_window_time_delta, end=end_date+relativedelta(years=rebalancing_freq+1,months=1), interval='1mo', progress=False) #Downloads the market stock close for same window.
    elif monthly_or_yearly_rebalancing == 'monthly':
        stock_data_download = yf.download(market_name, start=start_date-covariance_window_time_delta, end=end_date+relativedelta(months=rebalancing_freq+1), interval='1mo', progress=False)

    stock_data_download = stock_data_download[[close_type]].rename(columns={close_type: market_name})
    stock_data = pd.concat([prices,stock_data_download], axis = 1)
    
    pct = data.pct_returns_from_prices(stock_data)[start_date-covariance_window_time_delta:end_date+relativedelta(years=rebalancing_freq+1)]

    pct.index =pct.index.tz_localize(None)

    listparameters = []
    list_of_port_weights =[]
    list_of_port_esg_scores = []

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
    start_date = pd.to_datetime(start_date).tz_localize(None)
    end_date = pd.to_datetime(end_date).tz_localize(None)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    if monthly_or_yearly_rebalancing == 'yearly':
        delta = relativedelta(years=rebalancing_freq)
        while (start_date <= end_date):
            listparameters.append(efficient_frontier.efficient_frontier_solo(pct_returns,
                            strategy['bounds'],
                            strategy['sharpe_type'],
                            start_date-covariance_window_time_delta,#The start date for portfolio optimization will be the start date minus the covariance window
                            start_date,#The end date for portfolio optimization will be the start date minus the covariance window
                            strategy['wanted_return'],
                            strategy['maximum_risk'],
                            strategy['rebalancing_freq'],
                            ledoit_wolfe)) 
            
            
            list_of_port_weights.append(efficient_frontier.weights_of_portfolio(prices,listparameters[i]))

            list_of_port_esg_scores.append(portfolio.esg_score_of_portfolio(list_of_port_weights[i],esgdata.head(1)))


            list_of_cmle_returns.append(efficient_frontier.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[0])

            list_of_cml_allocations.append(efficient_frontier.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[1])

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
            print(f"Has calculated portfolio {i}")
        results = {
            'portfolio_parameters': listparameters,
            'portfolio_weights': list_of_port_weights,
            'portfolio_esg_scores': list_of_port_esg_scores,
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
            listparameters.append(efficient_frontier.efficient_frontier_solo(pct_returns,
                            strategy['bounds'],
                            strategy['sharpe_type'],
                            start_date-covariance_window_time_delta,#The start date for portfolio optimization will be the start date minus the covariance window
                            start_date,#The end date for portfolio optimization will be the start date minus the covariance window
                            strategy['wanted_return'],
                            strategy['maximum_risk'],
                            strategy['rebalancing_freq'],
                            ledoit_wolfe)) 
            
            
            list_of_port_weights.append(efficient_frontier.weights_of_portfolio(prices,listparameters[i]))    

            list_of_port_esg_scores.append(portfolio.esg_score_of_portfolio(list_of_port_weights[i],esgdata.head(1)))


            list_of_cmle_returns.append(efficient_frontier.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[0])

            list_of_cml_allocations.append(efficient_frontier.capital_mark_line_returns(listparameters[i],strategy['risk_free_rate'],strategy['maximum_risk'])[1])

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
            print(f"Has calculated portfolio {i}")
        results = {
            'portfolio_parameters': listparameters,
            'portfolio_weights': list_of_port_weights,
            'portfolio_esg_scores': list_of_port_esg_scores,
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
    
