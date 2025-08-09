import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def calculate_performance_metrics(returns, benchmark_returns):
    if len(returns) == 0:
        return {
            'Annualized Return': np.nan,
            'Sharpe Ratio': np.nan,
            'Information Ratio': np.nan,
            'Max Drawdown': np.nan
        }
    
    # Annualized Return
    annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
    
    # Annualized Volatility
    annualized_volatility = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio
    sharpe_ratio = annualized_return / annualized_volatility
    
    # Information Ratio
    active_return = returns - benchmark_returns
    if active_return.std() == 0:
        information_ratio = np.nan
    else:
        information_ratio = active_return.mean() / active_return.std() * np.sqrt(252)
    
    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Information Ratio': information_ratio,
        'Max Drawdown': max_drawdown
    }

def backtest_strategy(signal_df, sp500_returns_df, signal_names, output_dir):
    # Ensure the Date index is set for both DataFrames
    signal_df.index = pd.to_datetime(signal_df.index)
    sp500_returns_df.index = pd.to_datetime(sp500_returns_df.index)

    # Debug prints
    print("Signal DataFrame head:\n", signal_df.head())
    print("SP500 Returns DataFrame head:\n", sp500_returns_df.head())
    print("Signal DataFrame Date Range: ", signal_df.index.min(), " to ", signal_df.index.max())
    print("SP500 Returns DataFrame Date Range: ", sp500_returns_df.index.min(), " to ", sp500_returns_df.index.max())

    performance_metrics = {}

    # Merge the signal and SP500 data on the Date index
    combined_df = signal_df.join(sp500_returns_df, how='inner', lsuffix='_signal', rsuffix='')

    # Debug print
    print("Combined DataFrame head after join:\n", combined_df.head())
    print("Combined DataFrame length: ", len(combined_df))

    plt.figure(figsize=(14, 8))
    
    for signal in signal_names:
        # Calculate strategy returns using matrix multiplication
        combined_df['Strategy_Return'] = np.where(
            combined_df[signal] == 'Buy', combined_df['SP500_Return'],
            np.where(combined_df[signal] == 'Sell', -combined_df['SP500_Return'], 0)
        )

        # Calculate cumulative returns
        combined_df['Cumulative_Strategy_Return'] = (1 + combined_df['Strategy_Return']).cumprod()
        combined_df['Cumulative_SP500_Return'] = (1 + combined_df['SP500_Return']).cumprod()

        # Plot the cumulative returns
        plt.plot(combined_df.index, combined_df['Cumulative_Strategy_Return'], label=f'{signal} Cumulative Return', linewidth=2)

        # Calculate performance metrics
        metrics = calculate_performance_metrics(combined_df['Strategy_Return'], combined_df['SP500_Return'])
        performance_metrics[signal] = metrics

    # Plot the SP500 cumulative return
    plt.plot(combined_df.index, combined_df['Cumulative_SP500_Return'], label='SP500 Cumulative Return', color='cyan', linewidth=2)
    
    plt.title('Cumulative Returns: Strategies vs. SP500')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(output_dir, 'cumulative_returns_comparison.png'))
    plt.show()

    # Save the cumulative return table
    cumulative_returns_df = combined_df[['Cumulative_Strategy_Return', 'Cumulative_SP500_Return']]
    cumulative_returns_df.to_csv(os.path.join(output_dir, 'cumulative_returns.csv'), index=True)

    # Save the performance metrics
    performance_metrics_df = pd.DataFrame(performance_metrics).T
    performance_metrics_df.to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=True)

    return cumulative_returns_df, performance_metrics_df