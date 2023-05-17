import pandas as pd
import numpy as np
import itertools as it
import random
import yfinance as yf
import statsmodels.api as smf
from forex_python.converter import CurrencyRates

def esg_score_weight(data: pd.DataFrame,
                     weights: np.array,
                     min_esg_score: int = 500, 
                     max_esg_score: int = 2000):
    """ This function takes ESG score data and the wanted weight allocation for environment, sustaniability and governance seperately, calculates and returns ESG data with including the new weighted score.

    In this function, we takes raw ESG score data over the assets from a data provider, a wanted weight allocation for E, S and G seperately, a minimum and maximum acceptabel limit of total wanted weigthed ESG score.
    The function takes the weigthed fractions of E, S and/or G scores from the raw data and make a new column with the total weigthed score wanted. The weight allocation must sum up to 100% = 1.00. Further, the function have a limit setting a minimum or/and maximum level of total ESG score.

    :param data: Raw ESG score for the assets
    :param weights: Weight allocation in percent for environment, sustaniability and governance seperately in an array formated np.array([E, S, G]), must sum to 1.00
    :param min_esg_score: Limit for the minimum acceptable total esg score, default is 500
    :param max_esg_score: Limit for the maximum acceptable total esg score, default is 2000
    :returns: Raw ESG score including a new weighted score column according to the wanted levels of environment, sustaniability and governance seperately
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
    """ This function uses ESG data for the assets and the period, downloads monthly close price data and returns a full dataframe with ESG scores and price data.

    In this function, we take the ESG score and period of the wanted historical price period. With yahoo finance api, we download the monthly close data of each stock from yahoo finance.
    The function then concatenate the dataframes of the ESG score and historical monthly close prices of stocks to one dataframe that is returned.

    :param esg_data: ESG score for the assets
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
    """ This function takes and seperates a full dataframe with both esg weigthed score and returns a tuple with dataframe for prices and ESG scores seperately.
    
    In the function we seperate the esg weighted score columns with the prices/returns columns for the assets data.

    :param full_data: Both prices and ESG weigthed scores in one dataframe
    :returns: Prices and ESG scores as seperated dataframes
    """
    # select columns ending with '_weighted'
    weighted_cols = full_data.columns[full_data.columns.str.endswith('_weighted')]

    # create a new DataFrame with only the weighted columns
    esg = full_data[weighted_cols]

    # create a new DataFrame with the remaining columns
    prices = full_data.drop(weighted_cols, axis=1)
    prices.index = pd.to_datetime(prices.index)
    return prices, esg


def data_for_beta(symbols,
                  dates: tuple):
    """
    
    :param symbols: 
    :param dates: Period for the historical data
    :returns:
    """

    # create a new dataframe to store the monthly closing data
    stock_data = pd.DataFrame()
    for i in range(len(symbols)):
        stock_data_download = yf.download(symbols[i], start=dates[0], end=dates[1], interval='1mo', progress=False)
        stock_data_download = stock_data_download[['Close']].rename(columns={'Close': symbols[i]})
        # retrieve data from yfinance
        stock_data = pd.concat([stock_data,stock_data_download], axis = 1)
    return stock_data


def currency_rates(prices: pd.DataFrame):
    """ This function takes the exchange rates from forex-python api for foreign 
    
    :param prices: 
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


def pct_returns_from_prices(prices: pd.DataFrame):
    """
    
    :param:
    :returns: 
    """
    returns_pct_change = prices.pct_change().dropna()
    return returns_pct_change


def simulate_random_data_esg(n_rows: int = 10000):
    """

    :param n_rows: 
    :returns: 
    """
    data = np.random.randint(0, 100, size=n_rows)
    result = {'ESG' : data}
    df = pd.DataFrame(result)
    return df
