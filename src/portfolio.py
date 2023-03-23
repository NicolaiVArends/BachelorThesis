import pandas as pd
import numpy as np
from datetime import datetime as dt

def estimate_rolling_window(prices, start_year = 2003, end_year = 2023, window_size = 10):
    """
    Function that uses return data to plot portfolio returns performance
    :param: 
    :param: 
    :param: 
    :returns: 
    """

    # make different lists to append data in every window
    expected_return = []
    expected_year = []

    # setup af loop to iterate through window and make calculations
    for i in range(0, window_size + 1):

        # define the rolling window
        rolling_window = prices[i*12:i*12+(12*window_size)]

        # calculate the expected return as a dataframe
        expected_annual_returns = expected_return_annual(rolling_window)

        # append the results of expected return and the years to list
        expected_return.append(expected_annual_returns)

    # make list of expected return into a dataframe
    for x in range(start_year+10, end_year+1):
        expected_year.append(dt(x,1,1))
  
    expected_return = pd.DataFrame(expected_return, index=expected_year)

    return expected_return

def expected_return_annual(rolling_window):
    """

    :param: 
    :param: 
    :param: 
    :returns: 
    """

    expected_annual_returns = rolling_window.mean() * 12
    return expected_annual_returns

def historical_return(returns, frequency=12):
    """
   
    :param: 
    :param: 
    :returns: 
    """
    returns_pct_change = returns.pct_change()
    return (1 + returns_pct_change).prod() ** (frequency / returns_pct_change.count()) - 1

def _is_positive_semidefinite(matrix):
    """
   
    :param: 
    :param: 
    :returns: 
    """
    # Significantly more efficient than checking eigenvalues (stackoverflow.com/questions/16266720)
    try:
        # Significantly more efficient than checking eigenvalues (stackoverflow.com/questions/16266720)
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False

def sample_cov(prices, frequency=12, **kwargs):
    """
   
    :param: 
    :param: 
    :returns: 
    """
    returns = returns.pct_change()
    matrix = returns.cov() * frequency
    if _is_positive_semidefinite(matrix):
        return matrix
    else:
        raise Exception("AssertionError the matrix is not positive semidefinite")

def portfolio_return(returns: pd.DataFrame, weights: pd.DataFrame):
    """
    Function that uses portfolio returns and weights to compute the portfolio return by doing dot product between returns and weights
    :param: A dataframe of asset returns in portfolio
    :param: A dataframe or numpy array with the portfolio weights
    :returns: A float of the computed portfolio risk
    """
    
    return np.dot(returns, weights)

def portfolio_mean(returns: pd.DataFrame):
    """
    Function that uses portfolio
    :param:
    :returns: 
    """

    return returns.mean()

def portfolio_covariance(returns: pd.DataFrame):
    """
    Function that takes the returns of the different assets in a portfolio and computes the covariance matrix of it
    :param: A dataframe containing the returns of assets in portfolio
    :returns: A dataframe of the portfolio covariance matrix
    """
    
    return returns.cov()

def portfolio_std(port_cov, weights: pd.DataFrame):
    """
    Function that takes portfolio weigths and covariance matrix and computes the portfolio standard deviation (risk)
    :param: A dataframe or numpy array with the portfolio weights
    :param: A dataframe of the portfolio covariance matrix
    :returns: A float of the computed portfolio standard deviation (risk)
    """
    
    return np.sqrt(np.dot(weights, np.dot(weights, port_cov)))

