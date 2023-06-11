import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from datetime import datetime as dt
from src import efficient_frontier
from scipy.optimize import Bounds, LinearConstraint, minimize
np.random.seed(42) #We give it a seed, so the ledoit wolf give the same output each time


def mean_return_annual(returns: pd.DataFrame, 
                       frequency: int = 12):
    """ This function makes annual mean on prices/return for portfolio.

    :param returns: Monthly stock price/returns data in the portfolio
    :param frequency: Multiplier for making the return from monthly to annual data, default is 12
    :returns: Annual portfolio mean
    """
    expected_annual_returns = returns.mean() * frequency
    return expected_annual_returns


def mean_return_monthly(returns: pd.DataFrame):
    """ This function makes mean on prices/return for portfolio.

    :param returns: Monthly stock price/returns data in the portfolio
    :returns: Monthly portfolio mean
    """
    return(returns.mean())


def covariance_matrix_monthly(returns: pd.DataFrame, ledoit_wolfe = True):
    """ This function makes monthly portfolio covariance matrix on monthly prices/return for the portfolio.

    :param returns: Monthly stock price/returns data in the portfolio
    :returns: Monthly portfolio covariance matrix
    """
    if ledoit_wolfe:
        cov_estimator = LedoitWolf()
        return cov_estimator.fit(returns).covariance_
    else:
        return returns.cov()


def covariance_matrix_annual(returns: pd.DataFrame,
                             ledoit_wolfe: bool == True, 
                             frequency: int = 12):
    """ This function makes Yearly portfolio covariance matrix on monthly prices/return for the portfolio.

    :param returns: Monthly stock price/returns data in the portfolio
    :returns: Monthly portfolio covariance matrix
    """
    if ledoit_wolfe:
        cov_estimator = LedoitWolf()
        return cov_estimator.fit(returns).covariance_*12
    else:
        return returns.cov()


def portfolio_return(returns: pd.DataFrame, 
                     weights: pd.DataFrame):
    """This function uses portfolio returns and weights to compute the portfolio return by doing dot product between returns and weights.

    Note: Parameters needs to be on same interval e.g. both yearly or monthly data.

    :param returns: Stock price/returns data in the portfolio
    :param weights: Portfolio weight allocation
    :returns: Computed portfolio return with given weight allocation
    """
    return np.dot(returns, weights.T)



def portfolio_std(port_cov: pd.DataFrame, 
                  weights: pd.DataFrame):
    """ This function takes portfolio weigths and covariance matrix and computes the portfolio standard deviation (risk).

    :param port_cov: Portfolio covariance matrix
    :param weights: Porfoltio weight allocation
    :returns: Computed portfolio standard deviation (risk)
    """
    return np.sqrt(np.dot(weights, np.dot(weights, port_cov)))


def esg_score_of_portfolio(weights_of_portfolio: pd.DataFrame, 
                           ESG_score: pd.DataFrame): #This calculates the esg_score of our portfolios
    """ This function takes weight allocation of portfolio and ESG scores, calculates and returns the specific esg_score of our portfolios.

    :param weights_of_portfolio: Porfoltio weight allocation
    :param ESG_score: Raw ESG scores 
    :returns: Specific esg_score of the portfolios
    """
    row_sums = (ESG_score.values * weights_of_portfolio).sum(axis = 1)
    result = pd.DataFrame({'ESG_score_of_portfolio': row_sums}, index = weights_of_portfolio.index)
    return (result)


