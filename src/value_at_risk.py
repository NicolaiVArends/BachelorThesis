import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.portfolio import portfolio_mean, portfolio_std
from scipy.stats import norm

def calculate_var(port_mean, port_std, initial_investment = 100000, confidence_level = 0.05):
    """
    Function that 
    :param: 
    :returns: 
    """

    cut_off = norm.ppf(confidence_level, port_mean, port_std)
    value_at_risk = initial_investment - cut_off

    return value_at_risk

def calculate_cvar():
    """
    Function that 
    :param: 
    :returns: 
    """


    return

def plot_var(value_at_risk,
             num_days,
             mpl_style = 'default',
             title='Value at Risk'):
    """
    Function that 
    :param: 
    :returns: 
    """

    var_array = []
    for x in range(1, num_days+1):    
        var_array.append(np.round(value_at_risk * np.sqrt(x), 2))

    mpl.style.use(mpl_style)
    plt.plot(var_array)
    plt.title(title)
    plt.ylabel("VaR")
    plt.xlabel("Time")
    plt.legend()
    plt.show()

    return None

def plot_cvar(conditional_value_at_risk,
              mpl_style = 'default',
              title='Value at Risk'):
    """
    Function that 
    :param: 
    :returns: 
    """

    mpl.style.use(mpl_style)
    plt.plot()
    plt.title(title)
    plt.ylabel("VaR")
    plt.xlabel("Time")
    plt.legend()
    plt.show()




    return None



