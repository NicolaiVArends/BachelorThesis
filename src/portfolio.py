import pandas as pd
import numpy as np
from datetime import datetime as dt
from src import efficient_frontier
from scipy.optimize import Bounds, LinearConstraint, minimize

def rolling_window_expected_return(returns, 
                                   start_year = 2003, 
                                   end_year = 2023, 
                                   window_size = 10):
    """

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
        rolling_window = returns[i*12:i*12+(12*window_size)]

        # calculate the expected return as a dataframe
        expected_annual_returns = mean_return_annual(rolling_window)

        # append the results of expected return and the years to list
        expected_return.append(expected_annual_returns)

    # make list of expected return into a dataframe
    for x in range(start_year+10, end_year+1):
        expected_year.append(dt(x,1,1))
  
    expected_return = pd.DataFrame(expected_return, index=expected_year)

    return expected_return


def rolling_window_efficient_frontier(returns, 
                                      bounds, 
                                      Sharpe_Type, 
                                      wanted_return = None , 
                                      maximum_risk = None, 
                                      window_size = 10,
                                      start_year=2003):

    """

    :param: 
    :param: 
    :param: 
    :returns: 
    """
    parameters = []
    for i in range(0, window_size + 1):
        sample_rolling_window = returns.loc['{}-02-01'.format(str(2003+i)):'{}-03-01'.format(str(2003+i+window_size))] #10 is ten years
        ret_port = mean_return_annual(sample_rolling_window)
        cov_port = covariance_matrix_annual(sample_rolling_window)
        parameters.append(efficient_frontier.calculate_efficient_frontier(ret_port, cov_port,bounds,Sharpe_Type,wanted_return, maximum_risk))
    return parameters

def efficient_frontier_solo(returns, bounds, Sharpe_Type,start_date,end_date, wanted_return = None, maximum_risk = None,monthly_or_yearly_mean = "yearly"):
    parameters = []
    sample_rolling_window = returns.loc['{}'.format(str(start_date)):'{}'.format(str(end_date))]
    if monthly_or_yearly_mean == "monthly":
        parameters.append(efficient_frontier.calculate_efficient_frontier(mean_return_monthly(sample_rolling_window),covariance_matric_monthly(sample_rolling_window),bounds,Sharpe_Type,wanted_return, maximum_risk))


    elif monthly_or_yearly_mean == "yearly":
        parameters.append(efficient_frontier.calculate_efficient_frontier(mean_return_annual(sample_rolling_window),covariance_matrix_annual(sample_rolling_window),bounds,Sharpe_Type,wanted_return, maximum_risk))
    else:
        return("monthly or yearly has to be either yearly or monthly")


    return parameters


def mean_return_annual(returns, 
                       frequency=12):
    """

    :param: 
    :param: 
    :param: 
    :returns: 
    """

    expected_annual_returns = returns.mean() * frequency
    return expected_annual_returns

def mean_return_monthly(returns):
    return(returns.mean())
def covariance_matric_monthly(returns):
    return(returns.cov())


def covariance_matrix_annual(returns, 
                             frequency=12):
    """
   
    :param: 
    :param: 
    :returns: 
    """
    covmatrix_annual = returns.cov() * frequency
    return covmatrix_annual


def portfolio_return(returns: pd.DataFrame, 
                     weights: pd.DataFrame):
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


def portfolio_std(port_cov, 
                  weights: pd.DataFrame):
    """
    Function that takes portfolio weigths and covariance matrix and computes the portfolio standard deviation (risk)
    :param: A dataframe or numpy array with the portfolio weights
    :param: A dataframe of the portfolio covariance matrix
    :returns: A float of the computed portfolio standard deviation (risk)
    """
    
    return np.sqrt(np.dot(weights, np.dot(weights, port_cov)))


def esg_score_of_portfolio(weights_of_portfolio: pd.DataFrame, 
                           ESG_score: pd.DataFrame): #This calculates the esg_score of our portfolios
    row_sums = (ESG_score.values * weights_of_portfolio).sum(axis = 1)
    result = pd.DataFrame({'ESG_score_of_portfolio': row_sums}, index = weights_of_portfolio.index)
    return(result)


#This function calculates the capital market-line return based on an accepted risk for our portfolio and a risk free rate
def capital_mark_line_returns(parameters: np.array,
                              risk_free_rate: float, 
                              accepted_risk: float ): 
    prt_exp_return_array = []
    prt_risk_array =  []
    portfolio_allocations = []
    returns = []
    risk = []
    cmle = []

    for i in range(len(parameters)):
        prt_exp_return_array.append(parameters[i][1]) 
        prt_risk_array.append(parameters[i][0])
    
    for i in range(len(parameters)):
        cmle.append(risk_free_rate + accepted_risk*((prt_exp_return_array[i]-risk_free_rate)/prt_risk_array[i]))
        portfolio_allocations.append((cmle[i]-risk_free_rate)/(prt_exp_return_array[i]-risk_free_rate)) #Portfolio allocations
    return(cmle,portfolio_allocations)




    
    
