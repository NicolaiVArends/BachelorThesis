import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def calculate_efficient_frontier(portfolio_return, portfolio_cov):
    """
    
    :param:
    :param: 
    :returns: 
    """



    return

def calculate_capital_market_line():
    """
    
    :param:
    :param: 
    :returns: 
    """
    
    slope =
    cml_x_axis =
    cml_y_axis =

    return slope, cml_x_axis, cml_y_axis

def plot_efficient_frontier_2D(optimal_sharp_ratio_return, optimal_sharp_ratio_risk, frontier_x_axis, frontier_y_axis):
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
    mpl.style.use()
    plt.title('', fontsize=12)
    plt.xlabel('')
    plt.ylabel('')
    plt.xlim([])
    plt.ylim([])
    plt.plot(optimal_sharp_ratio_risk, optimal_sharp_ratio_return)
    plt.plot(frontier_x_axis, frontier_y_axis, marker='o')
    plt.legend()
    plt.show()

    return None

def plot_efficient_frontier_3D(optimal_portfolio_return, optimal_portfolio_risk, ):
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
    plt.axes(projection ='3d')
    #plt.plot(x, y, z, cmap ='viridis', edgecolor ='green')
    plt.title('Surface plot geeks for geeks')
    plt.show()

    return None




