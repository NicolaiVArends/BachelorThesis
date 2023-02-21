import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def portfolio_return(returns: pd.DataFrame, weights: pd.DataFrame):
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

def portfolio_risk(returns, weights: pd.DataFrame,):
    """
    Function that takes portfolio weigths and covariance matrix and computes the portfolio risk
    :param: A dataframe or numpy array with the portfolio weights
    :param: A dataframe of the portfolio covariance matrix
    :returns: A float of the computed portfolio risk
    """
    
    cov = portfolio_covariance(returns)
    return np.sqrt(np.dot(weights, np.dot(weights, cov)))

def portfolio_sharp_ratio(portfolio_returns, weights, portfolio_covariance):
    """
    Function that computes the sharp ratio by using the portfolio returns, weights and covariance from the functions portfolio_return() and portfolio_risk()
    :param: A dataframe containing the portfolio return
    :param: A dataframe or array containing the weighting of the portfolio
    :param: A dataframe containing the portfolio covariance
    :returns: A float of the sharp ratio for the given portfolio return, weight and covariance matrix
    """
    
    return portfolio_return(weights=weights, returns=portfolio_returns) / portfolio_risk(weights=weights, portfolio_covariance=portfolio_covariance)

def portfolio_minimize_risk_esg(portfolio_return, portfolio_covariance, esg_data, x0, linear_constraint, bounds, minimum_esg_score = 0, options = None):
    """
    Function that will take different inputs including esg score data and compute the minimum risk of different portfolios 
    :param: A dataframe of the portfolio covariance matrix
    :param: A dataframe of esg scores of the different assets in portfolio
    :param: x0 argument that is the initial guess for the minimizer
    :param: Linear constraints for the minimizer
    :param: Bounds for the minimizer
    :param: 
    :param: Options for the minimizer
    :returns: A dataframe containing portfolio weight choice for minimizing portfolio risk using esg scores
    """
    
    results = {'esg_val':[],
               'weights':[],
               'risk':[],
               'return':[]}
    
    function = lambda weight: portfolio_risk(weights=weight, portfolio_covariance=portfolio_covariance)
    constraint_esg = {'type': 'eq', 'fun': lambda weight: np.dot(weight, esg_data) - minimum_esg_score}
    result = minimize(function, 
                      x0, 
                      method='Nelder-Mead', 
                      bounds=bounds, 
                      constraints=[linear_constraint, constraint_esg], 
                      options=options)
   
    weights = list(result['x'])
    optimal_esg = np.dot(weights, esg_data)
    results['esg'].append(optimal_esg)
    results['weights'].append(weights)
    results['risk'].append(result['fun'])
    results['return'].append(np.dot(weights, portfolio_return.sum()))

    return results

def portfolio_max_sharp_ratio(portfolio_return, portfolio_covariance, esg_data, x0, linear_constraint, bounds, minimum_esg_score = 0, options = None):
    """
    Function that calculates the maximum sharp ratio using the portfolio sharp ratio function and doing a 
    :param: A dataframe of the portfolio covariance matrix
    :param: A dataframe of esg scores of the different assets in portfolio
    :param: x0 argument that is the initial guess for the minimizer
    :param: Linear constraints for the minimizer
    :param: Bounds for the minimizer
    :param: 
    :param: Options for the minimizer
    :returns: 
    """
    
    results = {'esg_val':[],
               'weights':[],
               'risk':[],
               'return':[]}
    
    function = lambda weight: portfolio_sharp_ratio(portfolio_returns=portfolio_return, weights=weight, portfolio_covariance=portfolio_covariance)
    constraint_esg = {'type': 'eq', 'fun': lambda weight: np.dot(weight, esg_data) - minimum_esg_score}
    result = minimize(function, 
                      x0, 
                      method='Nelder-Mead', 
                      bounds=bounds, 
                      constraints=[linear_constraint, constraint_esg], 
                      options=options)
    
    weights = list(result['x'])
    optimal_esg = np.dot(weights, esg_data)
    results['esg'].append(optimal_esg)
    results['weights'].append(weights)
    results['risk'].append(result['fun'])
    results['return'].append(np.dot(weights, portfolio_return.sum()))

    return result.x

def plot_portfolio_returns():



    plt.show();

    return
