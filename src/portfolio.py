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

def portfolio_minimize_risk_esg(portfolio_covariance, esg_scores, x0, linear_constraint, bounds, options = None):
    """
    Function that will take different inputs including esg score data and compute the minimum risk of different portfolios 
    :param: A dataframe of the portfolio covariance matrix
    :param: A dataframe of esg scores of the different assets in portfolio
    :param: x0 argument that is the initial guess for the minimizer
    :param: Linear constraints for the minimizer
    :param: Bounds for the minimizer
    :param: Options for the minimizer
    :returns: A dataframe containing portfolio weight choice for minimumizing portfolio risk using esg scores
    """
    function = lambda weight: portfolio_risk(weights=weight, portfolio_covariance=portfolio_covariance)
    constraint_esg = {'type': 'eq', 'fun': lambda weight: np.dot(weight, esg_scores)}
    result = minimize(function, x0, method='Nelder-Mead', bounds=bounds, constraints=[linear_constraint, constraint_esg], options=options)

    return result.x

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

