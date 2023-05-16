import pandas as pd
import numpy as np
import itertools as it
import random
import yfinance as yf
import statsmodels.api as smf
from forex_python.converter import CurrencyRates

def esg_score_weight(data: pd.DataFrame, 
                     weights: np.array,
                     min_esg_score: float, 
                     max_esg_score = 2000):
    """ This function makes 

    :param data:
    :param weights: 
    :param min_esg_score: 
    :param max_esg_score: 
    :returns:
    """
    if np.sum(weights) != 1:
        return("Weights must sum to 1")
    else:
        data["weighted_score"] = (((data["environment_score"] * (weights[0])) + \
                           (data["governance_score"] * (weights[1])) + \
                           (data["social_score"] * (weights[2])))*3)
        for index, row in data.iterrows():
            if row['CurrencyCode'] == 'SEK':
                data.at[index, 'stock_symbol'] += '.ST'
            elif row['CurrencyCode'] == 'NOK':
                data.at[index, 'stock_symbol'] += '.OL'
            elif row['CurrencyCode'] == 'DKK':
                data.at[index, 'stock_symbol'] += '.CO'
            else:
                pass
        result = data[(data["weighted_score"] >= min_esg_score) & (data["weighted_score"] <= max_esg_score)]
        
        #data["weighted_score"] <= max_esg_score
        return result
    
def stock_monthly_close(esg_data: pd.DataFrame, 
                        dates: tuple):
    """ This function uses esg data in the portfolio and the period, downloads monthly close price data and returns a full dataframe with esg scores and price data.

    In this function, we take the esg score and period of the wanted historical price period. With yahoo finance api, we download the monthly close data of each stock from yahoo finance.
    The function then concatenate the dataframes of the esg score and historical monthly close prices of stocks to one dataframe that is returned.

    :param esg_data: Esg score in portfolio
    :param dates: Period for the historical monthly close price data
    :returns: Dataframe with both monthly close price data for the stocks and esg score
    """
    symbols = esg_data['stock_symbol'].unique()

    # create a new dataframe to store the monthly closing data
    full_data = pd.DataFrame()
    for symbol in symbols:
    # retrieve data from yfinance
        stock_data = yf.download(symbol, start=dates[0], end=dates[1], interval='1mo', progress=False)
        
        # extract the 'Close' column and rename it with the stock symbol
        stock_data = stock_data[['Close']].rename(columns={'Close': symbol})
        
        # add the weighted score for the stock
        weighted_score = esg_data.loc[esg_data['stock_symbol']==symbol, 'weighted_score'].iloc[0]
        stock_data[symbol + '_weighted'] = weighted_score
        
        # append the stock data to the full_data
        full_data = pd.concat([full_data, stock_data], axis=1)
        full_data = full_data.dropna(axis=1,how='any')
    return full_data

def seperate_full_data(full_data: pd.DataFrame):
    """ This function takes and seperates a dataframe with both esg weigthed score and 
    
    :param full_data: 
    :returns:
    """
    # select columns ending with '_weighted'
    weighted_cols = full_data.columns[full_data.columns.str.endswith('_weighted')]

    # create a new DataFrame with only the weighted columns
    esg = full_data[weighted_cols]

    # create a new DataFrame with the remaining columns
    prices = full_data.drop(weighted_cols, axis=1)
    prices.index = pd.to_datetime(prices.index)
    return prices, esg

def data_for_beta(symbols,dates):
    """
    
    :param: 
    :param:
    :param: 
    :returns:
    """

    # create a new dataframe to store the monthly closing data
    stock_data1 = pd.DataFrame()
    for i in range(len(symbols)):
        stock_data = yf.download(symbols[i], start=dates[0], end=dates[1], interval='1mo', progress=False)
        #print(stock_data)
        stock_data = stock_data[['Close']].rename(columns={'Close': symbols[i]})
       # print(stock_data)
    # retrieve data from yfinance
        stock_data1 = pd.concat([stock_data1,stock_data], axis = 1)
    return stock_data1

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

def pct_returns_from_prices(prices):
    """
    
    :param:
    :param:
    :returns: 
    """
    returns_pct_change = prices.pct_change().dropna()
    return returns_pct_change
