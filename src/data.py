import pandas as pd
import numpy as np
import itertools as it
import random
import yfinance as yf
from forex_python.converter import CurrencyRates

def esg_score_weight(data: pd.DataFrame, weights: np.array):
    """
    
    :param: 
    :param:
    :param: 
    :returns:
    """
    if np.sum(weights) != 1:
        return("Weights must sum to 1")
    else:
        data["weighted_score"] = (data["environment_score"] * (3*weights[0])) + \
                           (data["governance_score"] * (3*weights[1])) + \
                           (data["social_score"] * (3*weights[2]))
        for index, row in data.iterrows():
            if row['CurrencyCode'] == 'SEK':
                data.at[index, 'stock_symbol'] += '.ST'
            elif row['CurrencyCode'] == 'NOK':
                data.at[index, 'stock_symbol'] += '.OL'
            elif row['CurrencyCode'] == 'DKK':
                data.at[index, 'stock_symbol'] += '.CO'
        return data
    
def stock_monthly_close(esg_data: pd.DataFrame, dates: tuple):
    """
    
    :param: 
    :param:
    :param: 
    :returns:
    """
    symbols = esg_data['stock_symbol'].unique()

    # create a new dataframe to store the monthly closing data
    monthly_close_df = pd.DataFrame()
    for symbol in symbols:
    # retrieve data from yfinance
        stock_data = yf.download(symbol, start=dates[0], end=dates[1], interval='1mo', progress=False)
        
        # extract the 'Close' column and rename it with the stock symbol
        stock_data = stock_data[['Close']].rename(columns={'Close': symbol})
        
        # add the weighted score for the stock
        weighted_score = esg_data.loc[esg_data['stock_symbol']==symbol, 'weighted_score'].iloc[0]
        stock_data[symbol + '_weighted'] = weighted_score
        
        # append the stock data to the monthly_close_df
        monthly_close_df = pd.concat([monthly_close_df, stock_data], axis=1)
    return(monthly_close_df.dropna(axis=1,how='all'))    

def currency_rates(prices):
    """
    
    :param: 
    :param:
    :param: 
    :returns:
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
        
def financial_benchmark_data(start_date, end_date, stocks):
    """
    
    :param:
    :param:
    :param: 
    :returns:
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
