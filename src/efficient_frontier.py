import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, LinearConstraint
from mpl_toolkits import mplot3d
from src.portfolio import portfolio_return, portfolio_covariance, portfolio_risk, portfolio_minimize_risk_esg, portfolio_max_sharp_ratio

def calculate_efficient_frontier_esg(returns, covariance, esg_data):
    """
    
    :param:
    :param: 
    :returns: 
    """

    w0 = np.linspace(start=1, stop=0, num=covariance.shape[1])
    x0 = w0/np.sum(w0)
    linear_constraint = LinearConstraint(np.ones((covariance.shape[1],), dtype=int),1,1)
    bounds = Bounds(-2, 5)
    options = {'xtol': 1e-07, 'gtol': 1e-07, 'barrier_tol': 1e-07, 'maxiter': 1000}

    # compute the optimal return and risk for risk aversion
    minimum_weights_esg = portfolio_minimize_risk_esg(returns, covariance, esg_data, x0, linear_constraint, bounds, options)
    min_risk_return = portfolio_return(returns, minimum_weights_esg)
    min_risk_risk = portfolio_risk(covariance, minimum_weights_esg)

    # compute the maximum sharp ratio for risk and return
    max_sr_weights_esg = portfolio_max_sharp_ratio(returns, covariance, esg_data, x0, linear_constraint, bounds,options)
    max_sr_return = portfolio_return(returns, max_sr_weights_esg)
    max_sr_risk = portfolio_risk(covariance, max_sr_weights_esg)

    frontier_x_axis = []
    frontier_y_axis = []#np.linspace(-0.3, *3, 50)
    
    return min_risk_return, min_risk_risk, max_sr_return, max_sr_risk, frontier_x_axis, frontier_y_axis

def calculate_capital_market_line(max_sr_return, max_sr_risk):
    """
    
    :param:
    :param: 
    :returns: 
    """

    slope = max_sr_return/max_sr_risk
    cml_x_axis = np.linspace(0-0.1,1,50)
    cml_y_axis = slope*cml_x_axis+0.01

    return slope, cml_x_axis, cml_y_axis

def plot_efficient_frontier_2D(max_sr_return, 
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

def plot_efficient_frontier_3D(max_sr_return, 
                               max_sr_risk,
                               frontier_x_axis, 
                               frontier_y_axis,
                               cml_x_axis,
                               cml_y_axis, 
                               mpl_style='default',
                               title='Efficient Frontier'):
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

    plt.figure()
    plt.title(title)
    plt.axes(projection ='3d')
    #plt.plot(frontier_x_axis, frontier_y_axis, esg_data, cmap ='viridis', edgecolor ='green')
    plt.legend()
    plt.show()

    return None




