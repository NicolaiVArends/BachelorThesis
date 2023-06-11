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
    :param market_name: Name of the market you want to calculate betas
    :returns: Betas for each portfolio 
    """
    betas = []
    for i in range(len(portfolio.columns)):
        beta = np.cov(np.stack([market[portfolio.columns[i]].to_numpy(),market[market_name].to_numpy()]),ddof=0)[0,1]/np.var(market[market_name])
        betas.append(beta)
    beta_of_port = np.sum(np.sum(np.multiply(betas,portfolio_weights)))
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
    print(f'Expected return on investment is {(100*(risk_free_rate+beta*(market_expected_returns-risk_free_rate))).round(5)}%')

    return(risk_free_rate+beta*(market_expected_returns-risk_free_rate))

def jensens_alpha(expected_return: float, 
                  actual_return: float):
    """ This function uses expected return calculated from the CAPM and actual return from the portfolio, calculates and returns Jensens alpha.

    In this function we use Jensens alpha and the expected return calculated from the CAPM to compute Jensens alpha.

    :param expected_return: Expected return from the portfolio given 
    :param actual_return: Actual return for portfolio
    :returns: Jensens Alpha
    """
    return (actual_return-expected_return)
  
def calculate_portfolio_beta_ols(market: pd.DataFrame, 
                                 portfolio: pd.DataFrame, 
                                 portfolio_weights: pd.DataFrame,
                                 market_name):
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
        x = market[market_name]
        x1 = stat.add_constant(x)
        y = market[portfolio.columns[i]]
        result = stat.OLS(y, x1).fit()
        betas.append(result.params[1])
    beta_of_port = np.multiply(betas,portfolio_weights).values.sum()
    return(beta_of_port)

