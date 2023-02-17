import pandas as pd
import numpy as np

def portfolio_return(weights: pd.DataFrame, returns: pd.DataFrame):
    """

    :param:
    :param: 
    :returns: 
    """
    return np.dot(returns, weights)

def portfolio_covariance(returns: pd.DataFrame):
    """

    :param:
    :param: 
    :returns: 
    """
    return returns.cov()

def portfolio_risk(weights: pd.DataFrame, portfolio_covariance):
    """
    
    :param:
    :param: 
    :returns: 
    """
    return np.sqrt(np.dot(weights, np.dot(weights, portfolio_covariance)))

def portfolio_minimize_risk(portfolio_covariance, x0, linear_constraint, bounds, options):
    """
    Function that will take different inputs and compute the minimum risk of a portfolio
    :param:
    :param:
    :param:
    :param:
    :param:
    :returns: A
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

