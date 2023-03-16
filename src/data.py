import pandas as pd
import numpy as np
import itertools as it
import random
from forex_python.converter import CurrencyRates

def currency_rates(prices):
    """
    
    :param start_date: The start date for the stock
    :param end_date:
    :param stocks: 
    :returns: A dataframe containing financial returns stock data
    """

    # convert currency in dataframe to USD from forex currency converter
    cr = CurrencyRates()
    exchange_rate_dict = {}

    for date in range(0, len(prices.index)):
        exchange_rate_dict.update({prices.index[date] : cr.get_rates("USD", prices.index[date])})

    for asset in prices:
        if asset.endswith('.ST'):
            dates = list(prices.index)
            for month in range(len(prices[asset])):
                prices[asset][month] = prices[asset][month]/(exchange_rate_dict[dates[month]]["SEK"])
        if asset.endswith('.OL'):
            dates = list(prices.index)
            for month in range(len(prices[asset])):
                prices[asset][month] = prices[asset][month]/(exchange_rate_dict[dates[month]]["NOK"])
        if asset.endswith('.CO'):
            dates = list(prices.index)
            for month in range(len(prices[asset])):
                prices[asset][month] = prices[asset][month]/(exchange_rate_dict[dates[month]]["DKK"])
        
    return prices

def get_financial_data(start_date, end_date, stocks):
    """
    
    :param start_date: The start date for the stock
    :param end_date:
    :param stocks: 
    :returns: A dataframe containing financial returns stock data
    """

    return 

def get_financial_benchmark_data(start_date, end_date, stocks):
    """
    
    :param start_date: The start date for the stock
    :param end_date:
    :param stocks: 
    :returns: A dataframe containing financial returns stock data
    """

    return

def filter_universe_esg(esg_score, esg_data: pd.DataFrame):
    """
    Takes financial asset data and filter it on ESG-score
    :param esg_score: Given level of min ESG-score
    :param esg_data: A Dataframe containing ESG-score for different assets
    :returns: Asset data filtered on given ESG-score
    """
    filtered_df = esg_data[esg_data['esg'] >= esg_score] 

    return filtered_df

def simulate_random_data_esg(n_rows = 10000):
    """

    :param n: 
    :returns: 
    """
    data = np.random.randint(0, 100, size=n_rows)
    result = {'ESG' : data}
    df = pd.DataFrame(result)
    return df
