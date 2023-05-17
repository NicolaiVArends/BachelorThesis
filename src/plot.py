import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets
from mpl_toolkits import mplot3d

def plot_cummulative_portfolio_returns(returns: pd.DataFrame,
                           mpl_style: str = 'default',
                           title: str = 'Portfolio cummulative returns'):
    """ This function uses raw price data to plot and show a 2D graph of portfolio cummulative returns.

    Note: The function will not return anything but will show a plot when running.

    :param returns: Raw price/returns that is not in percent already
    :param mpl_style: Matplotlib style of the plot, default is 'default'
    :param title: Title of the commulative plot, default is 'Portfolio cummulative returns'
    """
    returns_pct_cumm = returns.pct_change().dropna().cumsum()
    returns_pct_cumm['PortfolioMean'] = returns_pct_cumm.mean(numeric_only=True, axis=1)
    mpl.style.use(mpl_style)
    for asset in returns_pct_cumm:
        plt.plot(returns_pct_cumm[asset], alpha=0.4)
    plt.plot(returns_pct_cumm['PortfolioMean'], color='black')
    plt.title(title)
    plt.ylabel("Returns")
    plt.xlabel("Time")
    plt.legend(returns_pct_cumm)
    plt.show()
    return None


def plot_efficient_frontier(parameters,
                            start_year: int,
                            end_year: int,
                            risk_free_rate: float,
                            plot_cml: bool = True,
                            plot_max_sharp: bool = False,
                            mpl_style: str = 'default'):
    """ This function will plot and show a 2D graph of the efficient frontier with options showing capital market line and/or point of max sharp ratio.

    Note: The function will not return anything but will show a plot of the efficient frontier when running.

    :param parameters: Calculated efficient frontier data
    :param start_year: Start year of the data that is going to be plotted 
    :param end_year: End year of the data that is going to be plotted 
    :param risk_free_rate: Risk free rate of investment
    :param plot_cml: Boolean to enable or disable plotting of capital market line, default is True
    :param plot_max_sharp: Boolean to enable or disable plotting of max sharp ratio, default is False
    :param mpl_style: Matplotlib style of the plot, default is 'default'
    """
    mpl.style.use(mpl_style)
    plt.xlabel('Risk/Volatility')
    plt.ylabel('Expected Return')
    colors = ['r','b','k','m','g','c', 'lightslategrey', "darkcyan", "purple", "orange", "olive"]
    for i, x in enumerate(parameters):
        opt_sr_vol, opt_sr_ret, opt_risk_vol,  opt_risk_ret, frontier_x, frontier_y, _ = x
        if plot_max_sharp:
            plt.title('Efficient Frontier with Max Sharp')
            plt.xlim([0.0,0.4])
            plt.ylim([-0.2,0.7])
            if i % 3 != 0:
                continue
            plt.plot(opt_sr_vol,  opt_sr_ret, marker='o', color = f'{colors[i]}', markersize=8, label=f' {start_year} - {end_year} Max Sharp Ratio')
            plt.plot(frontier_x, frontier_y, linestyle='--', color = f'{colors[i]}', linewidth=2, label=f'{start_year}-{end_year} Efficient Frontier')
        elif plot_cml == True:
            plt.title('Efficient Frontier with Max Sharp')
            plt.xlim([0.0,0.4])
            plt.ylim([-0.2,0.7])
            cm_x = np.linspace(0,0.5,100)
            cm_y = (risk_free_rate + cm_x*((opt_sr_ret-risk_free_rate)/opt_sr_vol))
            if i % 3 != 0:
                continue
            plt.plot(cm_x,cm_y, color='k', linewidth = 2, label = 'Capital Market Line')
            plt.plot(opt_sr_vol,  opt_sr_ret, marker='o', color = f'{colors[i]}', markersize=8, label=f'{start_year}-{end_year} Max Sharp Ratio')
            plt.plot(frontier_x, frontier_y, linestyle='--', color = f'{colors[i]}', linewidth=2, label=f'{start_year}-{end_year} Efficient Frontier')
        else:
            plt.title('Efficient Frontier with Minimum Risk')
            plt.xlim([0.1,0.2])
            plt.ylim([-0.2,0.4])
            if i % 3 != 0:
                continue
            plt.plot(opt_risk_vol,  opt_risk_ret, marker='o', color = f'{colors[i]}', markersize=8, label=f'20{3+i:02d}-20{13+i:02d} Minimum Risk')
            plt.plot(frontier_x, frontier_y, linestyle='--', color = f'{colors[i]}', linewidth=2, label=f'20{3+i:02d}-20{13+i:02d} Efficient Frontier') 
        #plt.plot(frontier_x, frontier_y, linestyle='--', color = f'{colors[i]}', linewidth=2, label=f'20{3+i:02d}-20{13+i:02d} Efficient Frontier') 
    plt.legend()
    plt.show()
    return None

def plot_efficient_frontier_cml(parameters,
                                start_year: int,
                                end_year: int,
                                risk_free_rate: float,
                                plot_cml: bool = True,
                                risk: float = 0.1,
                                mpl_style: str ='default'):
    """
    Function that plot and shows a 2D graph for

    Note: The function will not return anything but will show a plot of the efficient frontier when running.

    :param: 
    :param:
    :param:
    :param: 
    :param:
    :param: 
    :param:
    :param: 
    """
    mpl.style.use(mpl_style)
    plt.xlabel('Risk/Volatility')
    plt.ylabel('Expected Return')
    colors = ['r','b','k','m','g','c', 'lightslategrey', "darkcyan", "purple", "orange", "olive"]
    for i, x in enumerate(parameters):
        opt_sr_vol, opt_sr_ret, opt_risk_vol,  opt_risk_ret, frontier_x, frontier_y, _ = x
        if plot_cml == True:
            plt.title('Efficient Frontier with Max Sharp')
            plt.xlim([0.0,0.4])
            plt.ylim([-0.2,0.7])      
            cm_x = np.linspace(0,0.5,100)
            cm_y = (risk_free_rate + cm_x*((opt_sr_ret-risk_free_rate)/opt_sr_vol))
            cm_yy = (risk_free_rate + risk*((opt_sr_ret-risk_free_rate)/opt_sr_vol))
            if i % 3 != 0:
                continue
            plt.plot(cm_x,cm_y, color='k', linewidth = 2, label = 'Capital Market Line')
            plt.plot(opt_sr_vol,  opt_sr_ret, marker='o', color = f'{colors[i]}', markersize=8, label=f'{start_year:02d}-{end_year:02d} Max Sharp Ratio')
            plt.plot(frontier_x, frontier_y, linestyle='--', color = f'{colors[i]}', linewidth=2, label=f'{start_year:02d}-{end_year:02d} Efficient Frontier')
            plt.plot(risk,cm_yy,  marker="o", markersize=5, markeredgecolor="k", markerfacecolor="k", label = 'wanted return')
        else:
            plt.title('Efficient Frontier with Minimum Risk')
            plt.xlim([0.1,0.2])
            plt.ylim([-0.2,0.4])
            if i % 3 != 0:
                continue
            plt.plot(opt_risk_vol,  opt_risk_ret, marker='o', color = f'{colors[i]}', markersize=8, label=f'20{3+i:02d}-20{13+i:02d} Minimum Risk')
            plt.plot(frontier_x, frontier_y, linestyle='--', color = f'{colors[i]}', linewidth=2, label=f'20{3+i:02d}-20{13+i:02d} Efficient Frontier') 
        plt.plot(frontier_x, frontier_y, linestyle='--', color = f'{colors[i]}', linewidth=2, label=f'20{3+i:02d}-20{13+i:02d} Efficient Frontier') 
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

