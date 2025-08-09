import pandas as pd

def simple_data_prep(data):
    # Convert 'Unnamed: 0' to date and set as index
    data['Unnamed: 0'] = pd.to_datetime(data['Unnamed: 0'])
    data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    data.set_index('Date', inplace=True)

    # Forward fill missing values
    data.fillna(method='ffill', inplace=True)

    # Calculate return from SP500 and drop NaNs
    if 'S&P_500' in data.columns:
        data['SP500_Return'] = data['S&P_500'].pct_change()
        data.dropna(subset=['SP500_Return'], inplace=True)
        data.drop(columns='S&P_500', inplace=True)

    # Convert data to weekly
    data = data.resample('W').last()

    # reset index
    data.reset_index(inplace=True)
    

    # Create a df without the SP500_Return
    data_no_ret = data.drop(columns='SP500_Return', errors='ignore')
    

    #save the data without SP500_Return
    data_no_ret.to_csv('data_no_ret.csv')

    # Create a df without VIX
    data_no_vix = data.drop(columns='VIX', errors='ignore')
    # save the data without VIX
    data_no_vix.to_csv('data_no_vix.csv')

    # Create a df without the SP500_Return and VIX
    data_no_ret_vix = data.drop(columns=['SP500_Return', 'VIX'], errors='ignore')
    
    # Return all dataframes
    return data, data_no_ret, data_no_vix, data_no_ret_vix



