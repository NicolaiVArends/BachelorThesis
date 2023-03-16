import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


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