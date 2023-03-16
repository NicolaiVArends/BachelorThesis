import pandas as pd
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from src.portfolio import *

def sharp_ratio(portfolio_returns, weights, portfolio_covariance):
    """
    Function that computes the sharp ratio by using the portfolio returns, weights and covariance from the functions portfolio_return() and portfolio_risk()
    :param: A dataframe containing the portfolio return
    :param: A dataframe or array containing the weighting of the portfolio
    :param: A dataframe containing the portfolio covariance
    :returns: A float of the sharp ratio for the given portfolio return, weight and covariance matrix
    """
    
    return portfolio_return(weights=weights, 
                            returns=portfolio_returns) / portfolio_std(port_cov=portfolio_covariance,
                                                                       weights=weights)

def minimize_risk(port_return, 
                    port_covariance, 
                    esg_data, 
                    x0,
                    linear_constraint, 
                    bounds, 
                    options = None):
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
    
    results = {'esg':[],
               'weights':[],
               'risk':[],
               'return':[]}
    
    function = lambda weight: portfolio_std(port_cov=port_covariance, weights=weight)
    constraint_esg = {'type': 'eq', 'fun': lambda weight: np.dot(weight, esg_data)}
    result = minimize(function, 
                      x0, 
                      method='trust-constr', 
                      bounds=bounds, 
                      constraints=[linear_constraint, constraint_esg], 
                      options=options)
   
    optimal_weights = list(result['x'])
    optimal_esg = np.dot(optimal_weights, esg_data)
    results['esg'].append(optimal_esg)
    results['weights'].append(optimal_weights)
    results['risk'].append(result['fun'])
    results['return'].append(np.dot(optimal_weights, port_return.sum()))

    return results

def maximize_sharp_ratio(port_return, 
                    port_covariance, 
                    esg_data, 
                    x0, 
                    linear_constraint, 
                    bounds, 
                    options = None):
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
    
    results = {'esg':[],
               'weights':[],
               'risk':[],
               'return':[]}
    
    function = lambda weight: sharp_ratio(portfolio_returns=port_return, weights=weight, portfolio_covariance=port_covariance)
    constraint_esg = {'type': 'eq', 'fun': lambda weight: np.dot(weight, esg_data)}
    result = minimize(function, 
                      x0, 
                      method='trust-constr', 
                      bounds=bounds, 
                      constraints=[linear_constraint, constraint_esg], 
                      options=options)
    
    optimal_weights = list(result['x'])
    optimal_esg = np.dot(optimal_weights, esg_data)
    results['esg'].append(optimal_esg)
    results['weights'].append(optimal_weights)
    results['risk'].append(result['fun'])
    results['return'].append(np.dot(optimal_weights, port_return.sum()))

    return results


def ef1(ret_port, cov_port):
    bounds = Bounds(-2, 5)

    #Create x0, the first guess at the values of each asset's weight.
    w0 = np.linspace(start=1, stop=0, num=cov_port.shape[1])
    x0 = w0/np.sum(w0)
    # All weights between 0 and 1
    # The second boundary is the sum of weights.
    linear_constraint = LinearConstraint(np.ones((cov_port.shape[1],), dtype=int),1,1)
    options = {'xtol': 1e-07, 'gtol': 1e-07, 'barrier_tol': 1e-07, 'maxiter': 1000}
 
    #These are the weights of the assets in the portfolio with the lowest level of risk possible.
    w_minr = minimize_risk(cov_port, x0, linear_constraint, bounds)
    opt_risk_ret = portfolio_return(ret_port,w_minr)
    opt_risk_vol = portfolio_std(cov_port, w_minr)
    print(f'Min. Risk = {opt_risk_vol*100:.3f}% => Return: {(opt_risk_ret*100):.3f}%  Sharpe Ratio = {opt_risk_ret/opt_risk_vol:.2f}')

    #These are the weights of the assets in the portfolio with the highest Sharpe ratio.
    w_sr_top = maximize_sharp_ratio(ret_port,cov_port, x0, linear_constraint, bounds, options)
    opt_sr_ret = portfolio_return(ret_port, w_sr_top)
    opt_sr_vol = portfolio_std(cov_port, w_sr_top)
    print(f'Max. Sharpe Ratio = {opt_sr_ret/opt_sr_vol:.2f} => Return: {(opt_sr_ret*100):.2f}%  Risk: {opt_sr_vol*100:.3f}%')

    frontier_y = np.linspace(-0.3, opt_sr_ret*3, 50)
    frontier_x = []

    x0 = w_sr_top
    for possible_return in frontier_y:
        cons = ({'type':'eq', 'fun': check_sum},
                {'type':'eq', 'fun': lambda w: portfolio_return(ret_port, w) - possible_return})

        #Define a function to calculate volatility
        fun = lambda w: np.sqrt(np.dot(w,np.dot(w,cov_port)))
        result = minimize(fun,x0,method='SLSQP', bounds=bounds, constraints=cons, callback=callbackF)
        frontier_x.append(result['fun'])

    frontier_x = np.array(frontier_x)
    dt_plot = pd.DataFrame(sr_opt_set, columns=['vol', 'ret'])
    vol_opt = dt_plot['vol'].values
    ret_opt = dt_plot['ret'].values
    sharpe_opt = ret_opt/vol_opt

    return opt_sr_vol, opt_sr_ret, opt_risk_vol,  opt_risk_ret, frontier_x, frontier_y, w_sr_top

def calculate_efficient_frontier_esg(returns, port_covariance, esg_data):
    """
    
    :param:
    :param: 
    :returns: 
    """

    w0 = np.linspace(start=1, stop=0, num=port_covariance.shape[1])
    x0 = w0/np.sum(w0)
    linear_constraint = LinearConstraint(np.ones((port_covariance.shape[1],), dtype=int),1,1)
    bounds = Bounds(-2, 5)
    options = {'xtol': 1e-07, 'gtol': 1e-07, 'barrier_tol': 1e-07, 'maxiter': 1000}

    # compute the optimal return and risk for risk aversion
    results_risk = minimize_risk(returns, port_covariance, esg_data, x0, linear_constraint, bounds, options)
    min_risk_return = results_risk['returns']
    min_risk_risk = results_risk['risk']

    # compute the maximum sharp ratio for risk and return
    results_sr = max_sharp_ratio(returns, port_covariance, esg_data, x0, linear_constraint, bounds, options)
    max_sr_return = results_sr['returns']
    max_sr_risk = results_sr['risk']

    # compute efficient frontier for esg score
    results_risk = minimize_risk(returns, port_covariance, esg_data, x0, linear_constraint, bounds, options)
    min_risk_opt_esg = results_risk['esg']
    min_risk_esg = results_risk['risk']

    # compute the axis
    frontier_x_axis = np.linspace(-0.3, max_sr_risk*3, 50)
    frontier_y_axis = np.linspace(-0.3, max_sr_return*3, 50)

    return min_risk_return, min_risk_risk, max_sr_return, max_sr_risk, min_risk_opt_esg, min_risk_esg, frontier_x_axis, frontier_y_axis

def capital_market_line(max_sr_return, max_sr_risk):
    """
    
    :param:
    :param: 
    :returns: 
    """

    slope = max_sr_return/max_sr_risk
    cml_x_axis = np.linspace(0-0.1,1,50)
    cml_y_axis = slope*cml_x_axis+0.01

    return slope, cml_x_axis, cml_y_axis
