import pandas as pd
import numpy as np

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


