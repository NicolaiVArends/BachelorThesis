import pandas as pd
import numpy as np
import matplotlib as mpl
import statsmodels.api as stat
from src import portfolio

def calculate_portfolio_beta(market: pd.DataFrame, 
                             portfolio: pd.DataFrame, 
                             portfolio_weights: pd.DataFrame,
                             market_name: str):
    """ This function uses covariance and variance to calculate and return the portfolio betas.

    In this function we takes the price data for a benchmark market, the portfolio returns and the optimal weight allocation for each rolling window of the portfolio.
    It uses covariance and variance from NumPy to calculate the betas in the model and returns them as a the betas of each portfolio.

    :param market: Price data for benchmark market in the model
    :param portfolio: Returns data for the portfolio assets
    :param portfolio_weights: Optimal weight allocations as a dataframe for each calculated rolling window
    :param market_name: Name of the market you want to calculate betas with
    :returns: Betas for each portfolio 
    """
    betas = []
    for i in range(len(portfolio.columns)):
        betas.append(np.cov(market[portfolio.columns[i]],market[market_name])[0][1]/np.var(market['SPY']))
    beta_of_port = np.multiply(betas,portfolio_weights).values.sum()
    return (beta_of_port)

def capm_calc(market_expected_returns: pd.DataFrame,
              beta: float,
              risk_free_rate: float = 0.05):
    """ This function uses the capital assets price model (CAPM), calculates and returns the calculated model.

    In this function we takes the expected return data for a benchmark market, the risk free rate of the investment and the calculated betas of each portfolio.
    It uses and return the computed CAPM.

    :param market_expected_returns: Expected returns data for the benchmark market
    :param beta: Calculated betas of each portfolio from the CAPM
    :param risk_free_rate: Risk free rate allocation of the investment in procent, default is 0.05
    :returns: Computed CAPM model
    """
    print(f'Excpected return on investment is {100*(risk_free_rate+beta*(market_expected_returns-risk_free_rate))}%')

    return(risk_free_rate+beta*(market_expected_returns-risk_free_rate))

def jensens_alpha(expected_return: float, 
                  actual_return: float):
    """ This function uses expected return calculated from the CAPM and actual return from the portfolio, calculates and returns jensens alpha.

    In this function we uses Jensens alpha and the expected return calculated from the CAPM to compute jensens alpha.

    :param expected_return: Expected return from the portfolio given 
    :param actual_return: Actual return for portfolio
    :returns: Jensens Alpha
    """
    return (actual_return-expected_return)
  
def calculate_portfolio_beta_ols(market: pd.DataFrame, 
                                 portfolio: pd.DataFrame, 
                                 portfolio_weights: pd.DataFrame):
    """ This function uses ordinary least squares (OLS) to calculate and return the portfolio betas.

    In this function we takes the price data for a benchmark market, the portfolio returns and the optimal weight allocation for each rolling window of the portfolio.
    It uses OLS from statsmodel.api to calculate the betas in the model and returns them as a the betas of each portfolio.

    :param market: Price data for benchmark market in the model
    :param portfolio: Returns data for the portfolio assets
    :param portfolio_weights: Optimal weight allocations as a dataframe for each calculated rolling window
    :returns: Betas for each portfolio 
    """
    betas = []
    for i in range(len(portfolio.columns)):
        x = portfolio.columns[i]
        y = market['SPY']
        # adding the constant term
        x = stat.add_constant(x)
        # performing the regression
        # and fitting the model
        result = stat.OLS(y, x).fit()
        # printing the summary table
        print(result.summary())
        betas.append(result.params)
    beta_of_port = np.multiply(betas,portfolio_weights).values.sum()
    return(beta_of_port)
