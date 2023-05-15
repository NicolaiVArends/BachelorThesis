import pandas as pd
import numpy as np
import matplotlib as mpl
import statsmodels.api as stat
from src.portfolio import *

def calculate_portfolio_beta(market,portfolio, portfolio_weights):
    
    
    betas = []
    for i in range(len(portfolio.columns)):

        betas.append(np.cov(market[portfolio.columns[i]],market['SPY'])[0][1]/np.var(market['SPY']))

    beta_of_port = np.multiply(betas,portfolio_weights).values.sum()

    return(beta_of_port)
def capm_calc(market_expected_returns,risk_free_rate,beta):
    print(f'Excpected return on investment is {100*(risk_free_rate+beta*(market_expected_returns-risk_free_rate))}%')

    return(risk_free_rate+beta*(market_expected_returns-risk_free_rate))
    

def alpha(expected_return,actual_return):
    return(actual_return-expected_return)
  