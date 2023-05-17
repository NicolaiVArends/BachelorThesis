import pandas as pd
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from src import portfolio

def sharp_ratio(portfolio_returns: pd.DataFrame, 
                weights: pd.DataFrame, 
                portfolio_covariance: pd.DataFrame):
    """This function computes the sharp ratio by using the portfolio returns, weights and covariance from the functions portfolio_return() and portfolio_risk().

    :param portfolio_returns: Portfolio returns
    :param weights: Weight allocation of the portfolio
    :param portfolio_covariance: Portfolio covariance matrix
    :returns: Sharp ratio for the given portfolio return, weight allocation and covariance matrix
    """
    return portfolio.portfolio_return(weights=weights, 
                            returns=portfolio_returns) / portfolio_std(port_cov=portfolio_covariance,
                                                                       weights=weights)


def check_sum(weight):
    return np.sum(weight)-1


def minimize_risk(port_covariance: pd.DataFrame,
                  x0,
                  bounds: Bounds):
    """ This function will take different inputs including portfolio covariance matrix and compute the minimum risk of different portfolios.
    
    :param port_covariance: Portfolio covariance matrix
    :param x0: Initial guess for the minimizer
    :param bounds: Bounds for the minimizer
    :returns: Portfolio weight choice for minimizing portfolio risk
    """
    function = lambda weight: portfolio_std(port_covariance, weights=weight)
    constraint = LinearConstraint(np.ones((port_covariance.shape[1],), dtype=int),1,1)
    options = {'xtol': 1e-07, 
               'gtol': 1e-07, 
               'barrier_tol': 1e-07, 
               'maxiter': 1000}
    result = minimize(function, 
                      x0,
                      method='SLSQP', 
                      bounds=bounds, 
                      constraints=constraint, 
                      options=options)
    return result.x


def maximize_sharp_ratio_wanted_returns(port_return: pd.DataFrame, 
                         port_covariance: pd.DataFrame,
                         x0,
                         bounds: Bounds,
                         wanted_return: float):
    """ This function will take different inputs including portfolio return and covariance matrix, to maximize the sharp ratio of different portfolios with a minimum return limit.
    
    :param port_return: Portfolio return
    :param port_covariance: Portfolio covariance matrix
    :param x0: Initial guess for the minimizer
    :param bounds: Bounds for the minimizer
    :param wanted_return: Sets minimum limit for the wanted return as a constraint
    :returns: Portfolio weight choice for maximizing sharp ratio with minimum limit of return
    """
    function = lambda weight: np.sqrt(np.dot(weight,np.dot(weight,port_covariance)))/port_return.dot(weight)
    bounds = bounds
    constraints = (LinearConstraint(np.ones((port_covariance.shape[1],), dtype=int),1,1),
                   {'type': 'eq',
                    'fun': lambda weight: wanted_return - weight@port_return})
                    #{'type': 'eq', 'fun': lambda weight: np.sqrt(np.dot(weight, np.dot(weight, port_covariance)))}) #Second restraint lets us define how high a return we want                                                          
    options = {'xtol': 1e-07, 
               'gtol': 1e-07, 
               'barrier_tol': 1e-07, 
               'maxiter': 1000}
    result = minimize(function, 
                      x0,
                      method='SLSQP', 
                      bounds=bounds, 
                      constraints=constraints, 
                      options=options)
    return result.x


def maximize_sharp_ratio_wanted_risk(port_return: pd.DataFrame, 
                         port_covariance: pd.DataFrame,
                         x0,
                         bounds: Bounds,
                         max_risk: float):
    """ This function will take different inputs including portfolio return and covariance matrix, to maximize the sharp ratio of different portfolios with a maximum risk limit.
    
    :param port_return: Portfolio return
    :param port_covariance: Portfolio covariance matrix
    :param x0: Initial guess for the minimizer
    :param bounds: Bounds for the minimizer
    :param max_risk: Sets maximum limit of taken risk as a constraint
    :returns: Portfolio weight choice for maximizing sharp ratio with maximum limit of risk
    """
    
    function = lambda weight: np.sqrt(np.dot(weight,np.dot(weight,port_covariance)))/port_return.dot(weight)
    bounds = bounds
    constraints = (LinearConstraint(np.ones((port_covariance.shape[1],), dtype=int),1,1),
                   {'type': 'eq',
                     'fun': lambda weight:  max_risk - np.sqrt(np.dot(weight, np.dot(weight, port_covariance)))},
                     {'type': 'ineq', 'fun': lambda weight: np.sqrt(np.dot(weight, np.dot(weight, port_covariance))) - max_risk})
                    #{'type': 'eq', 'fun': lambda weight: np.sqrt(np.dot(weight, np.dot(weight, port_covariance)))}) #Second restraint lets us define how high a return we want                                    
    options = {'xtol': 1e-07, 'gtol': 1e-07, 'barrier_tol': 1e-07, 'maxiter': 1000}
    result = minimize(function, 
                      x0,
                      method='SLSQP', 
                      bounds=bounds, 
                      constraints=constraints, 
                      options=options)
    return result.x


def maximize_sharp_ratio_no_spec(port_return: pd.DataFrame, 
                         port_covariance: pd.DataFrame,
                         x0,
                         bounds: Bounds):
    """ This function will take different inputs including portfolio return and covariance matrix, to maximize the sharp ratio of different portfolios with no extra constraints.
    
    :param port_return: Portfolio return
    :param port_covariance: Portfolio covariance matrix
    :param x0: Initial guess for the minimizer
    :param bounds: Bounds for the minimizer
    :returns: Portfolio weight choice for maximizing sharp ratio with no extra constraints
    """
    function = lambda weight: np.sqrt(np.dot(weight,np.dot(weight,port_covariance)))/port_return.dot(weight)
    bounds = bounds
    constraints = (LinearConstraint(np.ones((port_covariance.shape[1],), dtype=int),1,1))
                    #{'type': 'eq', 'fun': lambda weight: np.sqrt(np.dot(weight, np.dot(weight, port_covariance)))}) #Second restraint lets us define how high a return we want                                           
    options = {'xtol': 1e-07, 
               'gtol': 1e-07, 
               'barrier_tol': 1e-07, 
               'maxiter': 1000}
    result = minimize(function, 
                      x0,
                      method='SLSQP', 
                      bounds=bounds, 
                      constraints=constraints, 
                      options=options)
   
    return result.x


def calculate_efficient_frontier(ret_port: pd.DataFrame, 
                                 cov_port: pd.DataFrame,
                                 bounds: Bounds,
                                 Sharpe_Type,
                                 wanted_return = None, 
                                 max_risk = None):
    """ This function will take different inputs including portfolio return and covariance matrix, to maximize the sharp ratio of different portfolios with no extra constraints.
    
    :param ret_port: Portfolio return
    :param cov_port: Portfolio covariance matrix
    :param bounds: Bounds for the minimizer
    :param Sharpe_Type: Constraint that can be either "Wanted_return", "Maximum_risk", or "No_extra_constraint"
    :param wanted_return: Sets minimum limit for the wanted return as a constraint
    :param max_risk: Sets maximum limit of taken risk as a constraint
    :returns: Calculated efficient frontier including optimal return and risk for sharp ratio and points for plotting the efficient frontier
    """
    sr_opt_set = set()

    #Create x0, the first guess at the values of each asset's weight.
    w0 = np.linspace(start=1, stop=0, num=cov_port.shape[1])
    x0 = w0/np.sum(w0)
 
    #These are the weights of the assets in the portfolio with the lowest level of risk possible. taken from https://towardsdatascience.com/portfolio-optimization-with-scipy-aa9c02e6b937
    w_minr = minimize_risk(cov_port, x0, bounds)
    opt_risk_ret = portfolio.portfolio_return(ret_port, w_minr)
    opt_risk_vol = portfolio_std(cov_port, w_minr)
    print(f'Min. Risk = {opt_risk_vol*100:.3f}% => Return: {(opt_risk_ret*100):.3f}%  Sharpe Ratio = {opt_risk_ret/opt_risk_vol:.2f}')

    #These are the weights of the assets in the portfolio with the highest Sharpe ratio.
    #Here we chose whether we want to use risk or return as an ectra constraint

    if Sharpe_Type == "Maximum_risk":    
        w_sr_top = maximize_sharp_ratio_wanted_risk(ret_port,cov_port,x0,bounds,max_risk)
    elif Sharpe_Type == "Wanted_return":
        w_sr_top = maximize_sharp_ratio_wanted_returns(ret_port,cov_port,x0,bounds,wanted_return)
    elif Sharpe_Type == "No_extra_constraint":
        w_sr_top = maximize_sharp_ratio_no_spec(ret_port,cov_port,x0,bounds)
    else:
        print("wrong constraint type")
        return False

    opt_sr_ret = portfolio.portfolio_return(ret_port, w_sr_top)
    opt_sr_vol = portfolio_std(cov_port, w_sr_top)
    print(f'Max. Sharpe Ratio = {opt_sr_ret/opt_sr_vol:.2f} => Return: {(opt_sr_ret*100):.2f}%  Risk: {opt_sr_vol*100:.3f}%')

    frontier_y = np.linspace(-0.3, opt_sr_ret*3, 50)
    frontier_x = []

    x0 = w_sr_top
    for possible_return in frontier_y:
        cons = ({'type':'eq', 'fun': check_sum},
                {'type':'eq', 'fun': lambda w: portfolio.portfolio_return(ret_port, w) - possible_return})

        #Define a function to calculate volatility
        fun = lambda weights: portfolio_std(cov_port, weights)
        result = minimize(fun,
                          x0, 
                          method='SLSQP', 
                          bounds=bounds, 
                          constraints=cons)
        frontier_x.append(result['fun'])

    frontier_x = np.array(frontier_x)
    dt_plot = pd.DataFrame(sr_opt_set, columns=['vol', 'ret'])
    vol_opt = dt_plot['vol'].values
    ret_opt = dt_plot['ret'].values
    sharpe_opt = ret_opt/vol_opt

    return opt_sr_vol, opt_sr_ret, opt_risk_vol,  opt_risk_ret, frontier_x, frontier_y, w_sr_top


def capital_market_line(max_sr_return: float, 
                        max_sr_risk: float):
    """ This function takes the return and risk in the maximum sharp ratio, to compute the capital market line (CML) slope and axis for plotting.
    
    :param max_sr_return: Return in the maximum sharp ratio
    :param max_sr_risk: Risk in the maximum sharp ratio
    :returns: Capital market line (CML) slope, x and y axis for plotting later
    """

    slope = max_sr_return/max_sr_risk
    cml_x_axis = np.linspace(0-0.1,1,50)
    cml_y_axis = slope*cml_x_axis+0.01

    return slope, cml_x_axis, cml_y_axis


def weights_of_portfolio(stocks: pd.DataFrame, 
                         parameters: np.array):
    """ This function takes the stocks and parameters for the weight allocation, sets up in one dataframe that it returns.

    :param stocks: Portfolio covariance matrix
    :param parameters: Weight allocations as NumPy array
    :returns: Weight allocation of each stock in the portfolio
    """
    weight_array = []
    column_names = stocks.columns.values
    for i in range(len(parameters)):
        weight_array.append(parameters[i][6])
    df = pd.DataFrame(data =  weight_array, columns = column_names)
    return(df)


def portfolio_std(port_cov: pd.DataFrame, 
                  weights: pd.DataFrame):
    """ This function takes portfolio weigths and covariance matrix and computes the portfolio standard deviation (risk).

    :param port_cov: Portfolio covariance matrix
    :param weights: Weight allocation
    :returns: Computed portfolio standard deviation (risk)
    """
    
    return np.sqrt(np.dot(weights, np.dot(weights, port_cov)))

