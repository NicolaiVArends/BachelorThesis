import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, LinearConstraint
from mpl_toolkits import mplot3d
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

def max_sharp_ratio(port_return, 
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

def plot_efficient_frontier_return(max_sr_return, 
                               max_sr_risk, 
                               frontier_x_axis, 
                               frontier_y_axis,
                               cml_x_axis,
                               cml_y_axis, 
                               mpl_style='default',
                               title='Efficient Frontier'):
    """
    Function that plot and shows a 2D graph for 
    :param: 
    :param:
    :param:
    :param: 
    :param:
    :param: 
    :param:
    :param: 
    :returns: 
    """
    
    mpl.style.use(mpl_style)
    plt.title(title)
    #plt.xlabel('Portfolio Risk')
    #plt.ylabel('Portfolio Return')
    plt.xlim([min(frontier_x_axis), max(frontier_x_axis)])
    plt.ylim([min(frontier_y_axis), max(frontier_y_axis)])
    plt.plot(frontier_x_axis, frontier_y_axis)
    plt.plot(max_sr_risk, max_sr_return, marker='o')
    plt.plot(cml_x_axis, cml_y_axis, label=f'CML')
    plt.legend()
    plt.show()

    return None

def plot_efficient_frontier_return_3D(max_sr_return, 
                               max_sr_risk,
                               frontier_x_axis, 
                               frontier_y_axis,
                               cml_x_axis,
                               cml_y_axis, 
                               mpl_style='default',
                               title='Efficient Frontier 3D'):
    """
    
    :param:
    :param:
    :param:
    :param: 
    :param:
    :param: 
    :param:
    :param: 
    :returns: 
    """

    mpl.style.use(mpl_style)
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.set_title(title)
    #ax.set_xlabel('Portfolio risk')
    #ax.set_ylabel('Portfolio returns')
    #ax.set_zlabel('ESG')
    ax.scatter(frontier_x_axis, frontier_y_axis, esg, cmap ='viridis')
    ax.grid()
    plt.legend()
    plt.show()

    return None


def plot_efficient_frontier_esg_2D(max_sr_return, 
                               max_sr_risk, 
                               frontier_x_axis, 
                               frontier_y_axis,
                               cml_x_axis,
                               cml_y_axis, 
                               mpl_style='default',
                               title='Efficient Frontier'):
    """
    Function that plot and shows a 2D graph for 
    :param: 
    :param:
    :param:
    :param: 
    :param:
    :param: 
    :param:
    :param: 
    :returns: 
    """
    
    mpl.style.use(mpl_style)
    plt.title(title)
    #plt.xlabel('Portfolio Risk')
    #plt.ylabel('Portfolio Return')
    plt.xlim([min(frontier_x_axis), max(frontier_x_axis)])
    plt.ylim([min(frontier_y_axis), max(frontier_y_axis)])
    plt.plot(frontier_x_axis, frontier_y_axis)
    plt.plot(max_sr_risk, max_sr_return, marker='o')
    plt.plot(cml_x_axis, cml_y_axis, label=f'CML')
    plt.legend()
    plt.show()

    return None

def plot_efficient_frontier_esg_3D(max_sr_return, 
                               max_sr_risk,
                               frontier_x_axis, 
                               frontier_y_axis,
                               cml_x_axis,
                               cml_y_axis, 
                               mpl_style='default',
                               title='Efficient Frontier 3D'):
    """
    
    :param:
    :param:
    :param:
    :param: 
    :param:
    :param: 
    :param:
    :param: 
    :returns: 
    """

    mpl.style.use(mpl_style)
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.set_title(title)
    #ax.set_xlabel('Portfolio risk')
    #ax.set_ylabel('Portfolio returns')
    #ax.set_zlabel('ESG')
    ax.scatter(frontier_x_axis, frontier_y_axis, esg, cmap ='viridis')
    ax.grid()
    plt.legend()
    plt.show()

    return None

