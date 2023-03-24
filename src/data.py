import pandas as pd
import numpy as np
import itertools as it
import random
import yfinance as yf
from forex_python.converter import CurrencyRates

def esg_score_weight(data: pd.DataFrame, weights: np.array, min_esg_score: float):
    """
    
    :param: 
    :param:
    :param: 
    :returns:
    """
    if np.sum(weights) != 1:
        return("Weights must sum to 1")
    else:


        data["weighted_score"] = (data["environment_score"] * (weights[0]/1)) + \
                           (data["governance_score"] * (weights[1]/1)) + \
                           (data["social_score"] * (weights[2]/1))
        for index, row in data.iterrows():
            if row['CurrencyCode'] == 'SEK':
                data.at[index, 'stock_symbol'] += '.ST'
            elif row['CurrencyCode'] == 'NOK':
                data.at[index, 'stock_symbol'] += '.OL'
            elif row['CurrencyCode'] == 'DKK':
                data.at[index, 'stock_symbol'] += '.CO'
        result = data.loc[data["weighted_score"] >= min_esg_score]
        return result
    
def stock_monthly_close(esg_data: pd.DataFrame, dates: tuple):
    """
    
    :param: 
    :param:
    :param: 
    :returns:
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

def seperate_full_data(full_data):
    """
    
    :param: 
    :param:
    :param: 
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

def filter_prices(prices, from_asset, to_asset):
    """
    
    :param:
    :param:
    :returns: 
    """
    prices_filtered = prices.iloc[:,from_asset:to_asset]
    return prices_filtered

def pct_returns_from_prices(prices):
    """
    
    :param:
    :param:
    :returns: 
    """
    returns_pct_change = prices.pct_change().dropna()
    return returns_pct_change

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
