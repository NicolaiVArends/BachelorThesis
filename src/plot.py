import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def plot_cummulative_portfolio_returns(returns: pd.DataFrame,
                           mpl_style='default',
                           title='Portfolio cummulative returns'):
    """
    Function that uses return data to plot portfolio returns performance
    :param: 
    :param: 
    :param: 
    :returns: 
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

