import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds

def plot_cummulative_portfolio_returns(returns: pd.DataFrame,
                           mpl_style='default',
                           title='Portfolio cummulative returns'):
    """
    Function that uses return data to plot portfolio returns performance
    :param: 
    :param: 
    :param: 
    :returns: 
    """

    returns_pct_cumm = returns.pct_change().dropna().cumsum()
    returns_pct_cumm['PortfolioMean'] = returns_pct_cumm.mean(numeric_only=True, axis=1)

    mpl.style.use(mpl_style)
    for asset in returns_pct_cumm:
        plt.plot(returns_pct_cumm[asset], alpha=0.4)
    plt.plot(returns_pct_cumm['PortfolioMean'], color='black')
    plt.title(title)
    plt.ylabel("Returns")
    plt.xlabel("Time")
    plt.legend(returns_pct_cumm)
    plt.show()

    return None

def rate_of_return(beginning_price, end_price):
    """
    
    :param:
    :returns: 
    """
    return (end_price-beginning_price)/beginning_price

def annual_return(beginning_price, end_price, years_held):
    """
   
    :param: 
    :param: 
    :returns: 
    """
    rate = end_price/beginning_price
    return (((rate+1)**(1/years_held))-1)

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

