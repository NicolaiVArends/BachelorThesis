import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets
import seaborn as sns
sns.set_theme()
from mpl_toolkits import mplot3d

def plot_cummulative_portfolio_returns(returns: pd.DataFrame,
                           mpl_style: str = 'default',
                           title: str = 'Portfolio cummulative returns'):
    """ This function uses raw price data to plot and show a 2D graph of portfolio cummulative returns.

    NOTE: The function will not return anything but will show a plot when running.

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
    """ This function will plot and show a 2D graph of the efficient frontier with options showing capital market line and/or the max sharp ratio.

    NOTE: The function will not return anything but will show a plot of the efficient frontier when running.

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
            plt.xlim([0.0,0.3])
            plt.ylim([-0.5,0.5])
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
    """ This function will plot and show a 2D graph of the efficient frontier with capital market line and the max sharp ratio.

    NOTE: The function will not return anything but will show a plot of the efficient frontier when running.

    :param parameters: Calculated efficient frontier data
    :param start_year: Start year of the data that is going to be plotted 
    :param end_year: End year of the data that is going to be plotted 
    :param risk_free_rate: Risk free rate of investment
    :param plot_cml: Boolean to enable or disable plotting of capital market line, default is True
    :param risk: Acceptable risk level, default is 0.1
    :param mpl_style: Matplotlib style of the plot, default is 'default'
    """
    mpl.style.use(mpl_style)
    plt.xlabel('Risk/Volatility')
    plt.ylabel('Expected Return')
    colors = ['r','b','k','m','g','c', 'lightslategrey', "darkcyan", "purple", "orange", "olive"]
    for i, x in enumerate(parameters):
        opt_sr_vol, opt_sr_ret, opt_risk_vol,  opt_risk_ret, frontier_x, frontier_y, _ = x
        if plot_cml == True:
            plt.title('Efficient Frontier with Max Sharp')
            plt.xlim([0.0,0.3])
            plt.ylim([-0.5,0.5])      
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


def compare_returns(dates: pd.DataFrame, 
                    portfolio_returns: pd.DataFrame,
                    market_returns: pd.DataFrame):
    """ This function will plot 

    NOTE: The function will not return anything but will show a plot

    :param dates: 
    :param portfolio_returns:
    :param market_returns:
    """
    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    for i in range(len(dates)):
        plt.bar(dates[i], portfolio_returns[i], width=0.4, label='Portfolio Returns')
        plt.bar(dates[i], market_returns[i], width=0.4, label='Market Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.title('Portfolio Returns vs. Market Returns')
    plt.legend()
    # Rotating x-axis labels for better readability
    plt.xticks(rotation=45)
    # Displaying the chart
    plt.tight_layout()
    plt.show()
    return(None)


def compare_returns1(dates: pd.DataFrame, 
                    portfolio_returns: pd.DataFrame,
                    market_returns: pd.DataFrame):
    """ This function will plot that will compare returns from the benchmark market to the portfolio returns.

    NOTE: The function will not return anything but will show a plot

    :param dates: Return dates af a list computed from the backtest
    :param portfolio_returns:
    :param market_returns:
    """
    # Set the width of the bars
    bar_width = 0.2

    # Create an array of indices to use as x-axis positions
    x = np.arange(len(dates))

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(18, 6))
    portfolio_bars = ax.bar(x - bar_width/2, portfolio_returns, width=bar_width, label='Portfolio Returns')
    market_bars = ax.bar(x + bar_width/2, market_returns, width=bar_width, label='Market Returns')

    # Add a line separating positive and negative values
    ax.axhline(0, color='black', linewidth=1)

    # Add mesh background
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    # Add labels, title, and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    ax.set_title('Portfolio Returns vs. Market Returns')
    ax.set_xticks(x)
    ax.set_xticklabels(dates)
    ax.legend()
    # Add data labels above each bar
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize='small')

    autolabel(portfolio_bars)
    autolabel(market_bars)
    # Adjust spacing and layout
    fig.tight_layout()
    # Show the plot
    plt.show()

def compare_cumulative_returns(dates: list, 
                    portfolio_cumulative_returns: list,
                    market_cumulative_returns: list):
    """ This function will plot that will compare returns from the benchmark market to the portfolio returns.

    NOTE: The function will not return anything but will show a plot

    :param dates: Return dates as a list
    :param portfolio_returns: Portfolio returns as a list over time
    :param market_returns: Market returns as a list over time
    """
    # Set the width of the bars
    bar_width = 0.2

    # Create an array of indices to use as x-axis positions
    x = np.arange(len(dates))

    # Plotting the bar chart
    fig, ax = plt.subplots(figsize=(18, 6))
    portfolio_bars = ax.bar(x - bar_width/2, portfolio_cumulative_returns, width=bar_width, label='Portfolio cumulative  Returns')
    market_bars = ax.bar(x + bar_width/2, market_cumulative_returns, width=bar_width, label='Market cumulative Returns')

    # Add a line separating positive and negative values
    ax.axhline(0, color='black', linewidth=1)

    # Add mesh background
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    # Add labels, title, and legend
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    ax.set_title('Portfolio cumulative Returns vs. Market cumulative Returns')
    ax.set_xticks(x)
    ax.set_xticklabels(dates)
    ax.legend()
    # Add data labels above each bar
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize='small')

    autolabel(portfolio_bars)
    autolabel(market_bars)
    # Adjust spacing and layout
    fig.tight_layout()
    # Show the plot
    plt.show()


def plot_portfolio_betas(dates: list, 
                         portfolio_betas: list):
    """ This function will plot portfolio betas development over time using betas and return dates computed in backtesting().

    NOTE: The function will not return anything but will show a plot

    :param dates: Return dates as a list over time
    :param portfolio_betas: Portfolio betas over time
    """
    plt.figure(figsize=(14,8))
    bars = plt.bar(dates, portfolio_betas, color='dodgerblue', edgecolor='black')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Beta', fontsize=12)
    plt.title('Portfolio Betas Over Time', fontsize=16)
    plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels for better visibility
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)  # Add y-axis gridlines
    sns.despine()  # Remove top and right spines for a cleaner look
    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def plot_portfolio_esg_scores(dates: list, 
                              portfolio_esg_scores: list):
    """ This function will plot portfolio esg score development over time using portfolio esg scores and return dates computed in backtesting().

    NOTE: The function will not return anything but will show a plot

    :param dates: Return dates as a list over time
    :param portfolio_esg_scores: Portfolio ESG scoring over time
    """
    portfolio_esg_scores = [df.iloc[0, 0] for df in portfolio_esg_scores]

    plt.figure(figsize=(14,8))
    bars = plt.bar(dates, portfolio_esg_scores, color='seagreen', edgecolor='black')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio ESG Scores', fontsize=12)
    plt.title('Portfolio ESG Scores Over Time', fontsize=16)
    plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels for better visibility
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)  # Add y-axis gridlines
    sns.despine()  # Remove top and right spines for a cleaner look
    # Add data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

