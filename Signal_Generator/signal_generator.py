import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_signal_table_vol(predictions_df):
    signal_table = pd.DataFrame(index=predictions_df.index)
    
    for column in predictions_df.columns:
        if column == 'Real' or column == 'SP500_Return':
            signal_table[column] = predictions_df[column]
        else:
            signal_table[column + '_Signal'] = predictions_df.apply(
                lambda row: 'Buy' if row[column] < row['Real'] else 'Sell', axis=1
            )
    
    return signal_table

def generate_signal_table_sp500(predictions_df):
    signal_table = pd.DataFrame(index=predictions_df.index)
    
    for column in predictions_df.columns:
        if column == 'Real' or column == 'SP500_Return':
            signal_table[column] = predictions_df[column]
        else:
            signal_table[column + '_Signal'] = predictions_df.apply(
                lambda row: 'Buy' if row[column] > row['Real'] else 'Sell', axis=1
            )
    
    return signal_table


def plot_signals_vol(predictions_df, signal_table, model_names, output_dir):
    fig, axs = plt.subplots(len(model_names), 1, figsize=(14, 20))
    
    for i, name in enumerate(model_names):
        ax = axs[i]
        
        # Plot original VIX values and predictions
        ax.plot(predictions_df.index, predictions_df['Real'], label='Original VIX', color='black', linestyle='--', linewidth=1.5)
        ax.plot(predictions_df.index, predictions_df[name], label=f'{name} Predictions', color='blue', linewidth=1.5)
        
        # Plot signals
        buy_signals = signal_table[signal_table[f'{name}_Signal'] == 'Buy'].index
        sell_signals = signal_table[signal_table[f'{name}_Signal'] == 'Sell'].index
        
        ax.vlines(buy_signals, ymin=predictions_df['Real'].min(), ymax=predictions_df['Real'].max(), color='green', linestyle='--', label='Buy Signal')
        ax.vlines(sell_signals, ymin=predictions_df['Real'].min(), ymax=predictions_df['Real'].max(), color='red', linestyle='--', label='Sell Signal')
        
        ax.set_title(f'Volatility Strategy: {name} Predictions vs. Original VIX with Signals')
        ax.set_xlabel('Date')
        ax.set_ylabel('VIX')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volatility_predictions_with_signals.png'))
    plt.show()



def plot_signals_sp500(predictions_df, signal_table, model_names, output_dir):
    fig, axs = plt.subplots(len(model_names), 1, figsize=(14, 20))
    
    for i, name in enumerate(model_names):
        ax = axs[i]
        
        # Plot original VIX values and predictions
        ax.plot(predictions_df.index, predictions_df['Real'], label='Original VIX', color='black', linestyle='--', linewidth=1.5)
        ax.plot(predictions_df.index, predictions_df[name], label=f'{name} Predictions', color='blue', linewidth=1.5)
        
        # Plot signals
        buy_signals = signal_table[signal_table[f'{name}_Signal'] == 'Buy'].index
        sell_signals = signal_table[signal_table[f'{name}_Signal'] == 'Sell'].index
        
        ax.vlines(buy_signals, ymin=predictions_df['Real'].min(), ymax=predictions_df['Real'].max(), color='green', linestyle='--', label='Buy Signal')
        ax.vlines(sell_signals, ymin=predictions_df['Real'].min(), ymax=predictions_df['Real'].max(), color='red', linestyle='--', label='Sell Signal')
        
        ax.set_title(f'SP500 Strategy: {name} Predictions vs. Original VIX with Signals')
        ax.set_xlabel('Date')
        ax.set_ylabel('VIX')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sp500_predictions_with_signals.png'))
    plt.show()


    import pandas as pd

import pandas as pd

def generate_majority_signal_table(signal_table, model_names, signal_column_name='Majority_Signal'):
    majority_signal_table = pd.DataFrame(index=signal_table.index)
    
    for index, row in signal_table.iterrows():
        votes = row[[f'{name}_Signal' for name in model_names]]
        
        # Count the number of Buy and Sell votes
        buy_votes = (votes == 'Buy').sum()
        sell_votes = (votes == 'Sell').sum()
        
        # Determine the majority signal
        if buy_votes > sell_votes:
            majority_signal_table.at[index, signal_column_name] = 'Buy'
        elif sell_votes > buy_votes:
            majority_signal_table.at[index, signal_column_name] = 'Sell'
        else:
            majority_signal_table.at[index, signal_column_name] = 'Hold'  # In case of a tie, choose 'Hold' or another strategy
    
    # Include Real and SP500_Return columns if present
    for col in ['Real', 'SP500_Return']:
        if col in signal_table.columns:
            majority_signal_table[col] = signal_table[col]

    return majority_signal_table


import matplotlib.pyplot as plt
import os

def plot_vix_combined_signal(vix_combined_signal_table, output_dir):
    plt.figure(figsize=(14, 10))
    
    # Plot original VIX values
    plt.plot(vix_combined_signal_table.index, vix_combined_signal_table['Real'], label='Original VIX', color='black', linestyle='--', linewidth=1.5)
    
    # Plot signals
    buy_signals = vix_combined_signal_table[vix_combined_signal_table['vix_combined_signal'] == 'Buy'].index
    sell_signals = vix_combined_signal_table[vix_combined_signal_table['vix_combined_signal'] == 'Sell'].index
    
    plt.vlines(buy_signals, ymin=vix_combined_signal_table['Real'].min(), ymax=vix_combined_signal_table['Real'].max(), color='green', linestyle='--', label='Buy Signal')
    plt.vlines(sell_signals, ymin=vix_combined_signal_table['Real'].min(), ymax=vix_combined_signal_table['Real'].max(), color='red', linestyle='--', label='Sell Signal')
    
    plt.title('VIX Combined Strategy: Original VIX with Signals')
    plt.xlabel('Date')
    plt.ylabel('VIX')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'vix_combined_strategy_with_signals.png'))
    plt.show()


def plot_sp500_combined_signal(sp_combined_signal_table, output_dir):
    plt.figure(figsize=(14, 10))
    
    # Plot original SP500 returns
    plt.plot(sp_combined_signal_table.index, sp_combined_signal_table['Real'], label='SP500 Return', color='black', linestyle='--', linewidth=1.5)
    
    # Plot signals
    buy_signals = sp_combined_signal_table[sp_combined_signal_table['sp_combined_signal'] == 'Buy'].index
    sell_signals = sp_combined_signal_table[sp_combined_signal_table['sp_combined_signal'] == 'Sell'].index
    
    plt.vlines(buy_signals, ymin=sp_combined_signal_table['Real'].min(), ymax=sp_combined_signal_table['Real'].max(), color='green', linestyle='--', label='Buy Signal')
    plt.vlines(sell_signals, ymin=sp_combined_signal_table['Real'].min(), ymax=sp_combined_signal_table['Real'].max(), color='red', linestyle='--', label='Sell Signal')
    
    plt.title('SP500 Combined Strategy: SP500 Returns with Signals')
    plt.xlabel('Date')
    plt.ylabel('SP500 Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'sp500_combined_strategy_with_signals.png'))
    plt.show()

def combine_and_plot_signals(vix_combined_signal_table, sp_combined_signal_table, output_dir):
    combined_signal_table = pd.DataFrame(index=vix_combined_signal_table.index)
    
    for index, row in vix_combined_signal_table.iterrows():
        vix_signal = row['vix_combined_signal']
        sp_signal = sp_combined_signal_table.at[index, 'sp_combined_signal']
        
        if vix_signal == sp_signal:
            combined_signal_table.at[index, 'combined_signal'] = vix_signal
        else:
            combined_signal_table.at[index, 'combined_signal'] = 'Hold'  # Only trade when both agree
    
    # Include Real and SP500_Return columns if present
    combined_signal_table['Real_VIX'] = vix_combined_signal_table['Real']
    combined_signal_table['Real_SP500'] = sp_combined_signal_table['Real']
    
    fig, ax1 = plt.subplots(figsize=(14, 10))
    
    # Plot original VIX values
    ax1.plot(combined_signal_table.index, combined_signal_table['Real_VIX'], label='Original VIX', color='black', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('VIX', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Plot combined signals on primary y-axis
    buy_signals = combined_signal_table[combined_signal_table['combined_signal'] == 'Buy'].index
    sell_signals = combined_signal_table[combined_signal_table['combined_signal'] == 'Sell'].index
    ax1.vlines(buy_signals, ymin=combined_signal_table['Real_VIX'].min(), ymax=combined_signal_table['Real_VIX'].max(), color='green', linestyle='--', label='Buy Signal')
    ax1.vlines(sell_signals, ymin=combined_signal_table['Real_VIX'].min(), ymax=combined_signal_table['Real_VIX'].max(), color='red', linestyle='--', label='Sell Signal')
    
    # Create secondary y-axis for SP500 returns
    ax2 = ax1.twinx()
    ax2.plot(combined_signal_table.index, combined_signal_table['Real_SP500'], label='SP500 Return', color='blue', linestyle='-', linewidth=1.5)
    ax2.set_ylabel('SP500 Return', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    fig.tight_layout()
    plt.title('Combined Strategy: VIX and SP500 Returns with Signals')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'combined_strategy_with_signals.png'))
    plt.show()
    
    return combined_signal_table

