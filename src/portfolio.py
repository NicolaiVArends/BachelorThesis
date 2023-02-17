import pandas as pd
import numpy as np
from scipy.optimize import minimize

def portfolio_return(weights: pd.DataFrame, returns: pd.DataFrame):
    """
    Function that uses portfolio returns and weights to compute the portfolio return by doing dot product between returns and weights
    :param: A dataframe of asset returns in portfolio
    :param: A dataframe or numpy array with the portfolio weights
    :returns: A float of the computed portfolio risk
    """
    return np.dot(returns, weights)

def portfolio_covariance(returns: pd.DataFrame):
    """
    Function that takes the returns of the different assets in a portfolio and computes the covariance matrix of it
    :param: A dataframe containing the returns of assets in portfolio
    :returns: A dataframe of the portfolio covariance matrix
    """
    return returns.cov()

def portfolio_risk(weights: pd.DataFrame, portfolio_covariance):
    """
    Function that takes portfolio weigths and covariance matrix and computes the portfolio risk
    :param: A dataframe or numpy array with the portfolio weights
    :param: A dataframe of the portfolio covariance matrix
    :returns: A float of the computed portfolio risk
    """
    return np.sqrt(np.dot(weights, np.dot(weights, portfolio_covariance)))

def portfolio_minimize_risk(portfolio_covariance, esg_score, x0, linear_constraint, bounds, options):
    """
    Function that will take different inputs and compute the minimum risk of different portfolios u
    :param:
    :param:
    :param:
    :param:
    :param:
    :returns: 
    """

    return

def portfolio_max_sharp_ratio(portfolio_return, portfolio_covariance, x0, linear_constraint, bounds, options):
    """
    
    :param:
    :param:
    :param:
    :param: 
    :param:
    :param: 
    :returns: 
    """
 

    return

