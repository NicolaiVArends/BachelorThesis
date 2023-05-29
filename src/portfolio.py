import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf

from datetime import datetime as dt
from src import efficient_frontier
from scipy.optimize import Bounds, LinearConstraint, minimize


def rolling_window_expected_return(returns: pd.DataFrame, 
                                   start_year: int = 2003, 
                                   end_year: int = 2023, 
                                   window_size: int = 10):
    """ This function computes expected return of a portfolio for a given rolling window using the stock price/returns data.

    :param returns: Stock price/returns data in the portfolio
    :param start_year: Starting year of the stock return/price data in portfolio, default is 2003
    :param end_year: Ending year of the stock return/price data in portfolio, default is 2023
    :param window_size: Size of the rolling window in year, default is 10
    :returns: Expected return on rolling window for portfolio
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


def rolling_window_efficient_frontier(returns: pd.DataFrame, 
                                      bounds: Bounds, 
                                      Sharpe_Type: str, 
                                      wanted_return: float = None, 
                                      maximum_risk: float = None, 
                                      window_size: int = 10):
    """ This function calculates efficient frontier on a rolling window.

    :param returns: Stock price/returns data in the portfolio
    :param bounds: Bounds for the minimizer
    :param Sharpe_Type: Constraint that can be either "Wanted_return", "Maximum_risk", or "No_extra_constraint"
    :param wanted_return: Sets minimum limit for the wanted return as a constraint
    :param maximum_risk: Sets maximum limit of taken risk as a constraint
    :param window_size: Size of the rolling window in year, default is 10
    :returns: Calculated efficient frontier parameters on rolling window
    """
    parameters = []
    for i in range(0, window_size + 1):
        sample_rolling_window = returns.loc['{}-02-01'.format(str(2003+i)):'{}-03-01'.format(str(2003+i+window_size))] #10 is ten years
        ret_port = mean_return_annual(sample_rolling_window)
        cov_port = covariance_matrix_annual(sample_rolling_window)
        parameters.append(efficient_frontier.calculate_efficient_frontier(ret_port, 
                                                                          cov_port,
                                                                          bounds,
                                                                          Sharpe_Type,
                                                                          wanted_return,
                                                                          maximum_risk))
    return parameters


def efficient_frontier_solo(returns: pd.DataFrame, 
                            bounds: Bounds, 
                            Sharpe_Type,
                            start_date: int,
                            end_date: int, 
                            wanted_return: float = None, 
                            maximum_risk: float = None,
                            monthly_or_yearly_mean: str = "yearly",
                            ledoit_Wolf: bool = True):
    """ This function calculates the efficient frontier on one window time period.

    :param returns: Stock price/returns data in the portfolio
    :param bounds: Bounds for the minimizer
    :param Sharpe_Type: Constraint that can be either "Wanted_return", "Maximum_risk", or "No_extra_constraint"
    :param start_date: Starting year of the stock return/price data in portfolio
    :param end_date: Ending year of the stock return/price data in portfolio
    :param wanted_return: Sets minimum limit for the wanted return as a constraint, default is None
    :param maximum_risk: Sets maximum limit of taken risk as a constraint, default is None
    :param monthly_or_yearly_mean: Define if the data is yearly or monthly mean, default is "yearly"
    :returns: Parameters with the calculated efficient frontier data
    """    
    parameters = []
    sample_rolling_window = returns.loc['{}'.format(str(start_date)):'{}'.format(str(end_date))]
    if monthly_or_yearly_mean == "monthly":
        parameters.append(efficient_frontier.calculate_efficient_frontier(mean_return_monthly(sample_rolling_window),
                                                                          covariance_matric_monthly(sample_rolling_window,ledoit_Wolf),
                                                                          bounds,
                                                                          Sharpe_Type,
                                                                          wanted_return,
                                                                          maximum_risk))
    elif monthly_or_yearly_mean == "yearly":
        parameters.append(efficient_frontier.calculate_efficient_frontier(mean_return_annual(sample_rolling_window),
                                                                          covariance_matrix_annual(sample_rolling_window,ledoit_Wolf),
                                                                          bounds,
                                                                          Sharpe_Type,
                                                                          wanted_return,
                                                                          maximum_risk))
    else:
        return("monthly or yearly has to be either yearly or monthly")
    return parameters


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


def covariance_matric_monthly(returns: pd.DataFrame,
                            ledoit_wolfe: bool = True):
    """ This function makes monhtly portfolio covariance matrix on monthly prices/return for the portfolio.

    :param returns: Monthly stock price/returns data in the portfolio
    :returns: Monthly portfolio covariance matrix
    """
    #print(returns.cov().head().size)
    if ledoit_wolfe == True:
        return(LedoitWolf().fit(np.random.multivariate_normal(mean=returns.mean(),cov=returns.cov(),size=50)).covariance_) #https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
    else:
        return(returns.cov())


def covariance_matrix_annual(returns: pd.DataFrame,
                             ledoit_Wolf: bool == True, 
                             frequency: int = 12):
    """ This function makes yearly portfolio covariance matrix by taking in monthly prices/return for the portfolio.
   
    :param returns: Monthly stock price/returns data in the portfolio
    :param frequency: Multiplier for making the return from monthly to annual data, default is 12
    :returns: Yearly portfolio covariance matrix 
    """
    if ledoit_Wolf == True:
        return(LedoitWolf().fit(np.random.multivariate_normal(mean=returns.mean(),cov=returns.cov(),size=50)).covariance_* frequency)
    else:
        return(returns.cov()*frequency)


def portfolio_return(returns: pd.DataFrame, 
                     weights: pd.DataFrame):
    """This function uses portfolio returns and weights to compute the portfolio return by doing dot product between returns and weights.

    Note: Parameters needs to be on same interval e.g. both yearly or monthly data.

    :param returns: Stock price/returns data in the portfolio
    :param weights: Portfolio weight allocation
    :returns: Computed portfolio return with given weight allocation
    """
    return np.dot(returns, weights.T)


def portfolio_mean(returns: pd.DataFrame):
    """ This function computed the portfolio mean.

    :param returns: Monthly/Yearly stock price/returns data in the portfolio
    :returns: Monthly/Yearly portfolio mean
    """
    return returns.mean()


def portfolio_covariance(returns: pd.DataFrame):
    """ This function takes the returns of the different assets in a portfolio and computes the covariance matrix of it.

    :param returns: Monthly/Yearly stock price/returns data in the portfolio
    :returns: Monthly/Yearly portfolio covariance matrix
    """
    return returns.cov()


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


def capital_mark_line_returns(parameters: np.array,
                              risk_free_rate: float, 
                              accepted_risk: float): 
    """ This function takes parameters from capital market line/efficient frontier, calculates and return the returns in the capital market line based on the risk-free rate and an accepted risk level for the portfolios.

    :param parameters: Portfolio weight allocation
    :param risk_free_rate: Risk-free rate of capital market line
    :param accepted_risk: Limit/Level of accepted risk
    :returns: Return of the capital market line with an accepted risk
    """
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

