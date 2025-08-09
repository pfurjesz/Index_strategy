import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# append the parent directory to the path
import sys
sys.path.append('..')

from Predictor_models.cnn_model import CNN
from Predictor_models.rnn_model import RNN
from Predictor_models.gru_model import GRU
from Predictor_models.lstm_model import LSTM

def predictor_module(data, model_names, target_variable, output_dir):
    # Drop rows with missing values
    data.dropna(inplace=True)

    # Extract the features and target variable
    X = data.drop(columns=[target_variable]).values
    y = data[target_variable].values

    # Normalize the features and target
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Split the data into training and testing sets
    split_idx = len(X_scaled) // 2
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Instantiate the models
    input_size = X_train.shape[1]
    all_models = {
        'CNN': CNN(input_size),
        'RNN': RNN(input_size),
        'GRU': GRU(input_size),
        'LSTM': LSTM(input_size)
    }

    models = {name: all_models[name] for name in model_names}

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizers = {name: optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) for name, model in models.items()}

    # Early stopping parameters
    patience = 20
    min_delta = 0.0001

    # Train and evaluate each model
    predictions_full_df = pd.DataFrame(index=range(len(y_scaled)))
    predictions_train_df = pd.DataFrame(index=range(len(y_train)))
    predictions_test_df = pd.DataFrame(index=range(len(y_test)))

    n_epochs = 500
    window_size = 10

    for name, model in models.items():
        optimizer = optimizers[name]

        best_loss = float('inf')
        patience_counter = 0

        # Train the model
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            if name in ['RNN', 'GRU', 'LSTM']:
                X_input = X_train.view(X_train.size(0), -1, input_size)  # Reshape for RNN, GRU, LSTM
            else:
                X_input = X_train
            outputs = model(X_input)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Early stopping check
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_input)
                val_loss = criterion(val_outputs, y_train)
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} for {name}")
                    break

        # Make rolling predictions on the full dataset
        predictions_full = []
        model.eval()
        with torch.no_grad():
            for i in range(len(X_scaled) - window_size + 1):
                X_batch = torch.tensor(X_scaled[i:i+window_size], dtype=torch.float32)
                if name in ['RNN', 'GRU', 'LSTM']:
                    X_batch = X_batch.view(X_batch.size(0), -1, input_size)
                pred = model(X_batch).mean().item()
                predictions_full.append(pred)

        # Pad the predictions to match the length of the full dataset
        predictions_full = [np.nan] * (window_size - 1) + predictions_full
        predictions_full_df[name] = predictions_full

        # Make rolling predictions on train set
        predictions_train = []
        model.eval()
        with torch.no_grad():
            for i in range(len(X_train) - window_size + 1):
                X_batch = X_train[i:i+window_size]
                if name in ['RNN', 'GRU', 'LSTM']:
                    X_batch = X_batch.view(X_batch.size(0), -1, input_size)
                pred = model(X_batch).mean().item()
                predictions_train.append(pred)

        # Pad the predictions to match the length of y_train
        predictions_train = [np.nan] * (window_size - 1) + predictions_train
        predictions_train_df[name] = predictions_train

        # Make rolling predictions on test set
        predictions_test = []
        model.eval()
        with torch.no_grad():
            for i in range(len(X_test) - window_size + 1):
                X_batch = X_test[i:i+window_size]
                if name in ['RNN', 'GRU', 'LSTM']:
                    X_batch = X_batch.view(X_batch.size(0), -1, input_size)
                pred = model(X_batch).mean().item()
                predictions_test.append(pred)

        # Pad the predictions to match the length of y_test
        predictions_test = [np.nan] * (window_size - 1) + predictions_test
        predictions_test_df[name] = predictions_test

    # Inverse scale the predictions and y values for plotting
    predictions_full_df = predictions_full_df.apply(lambda col: scaler_y.inverse_transform(col.values.reshape(-1, 1)).flatten())
    predictions_train_df = predictions_train_df.apply(lambda col: scaler_y.inverse_transform(col.values.reshape(-1, 1)).flatten())
    predictions_test_df = predictions_test_df.apply(lambda col: scaler_y.inverse_transform(col.values.reshape(-1, 1)).flatten())
    y_train_inverse = scaler_y.inverse_transform(y_train.numpy().reshape(-1, 1)).flatten()
    y_test_inverse = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()
    y_scaled_inverse = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    # Include real values in the DataFrame
    predictions_full_df['Real'] = y_scaled_inverse
    predictions_train_df['Real'] = y_train_inverse
    predictions_test_df['Real'] = y_test_inverse

    # Plotting the results for full dataset
    plt.figure(figsize=(14, 10))
    plt.plot(y_scaled_inverse, label='Original Data', color='black', linestyle='--', linewidth=1.5)
    colors = ['blue', 'green', 'red', 'purple']
    for i, name in enumerate(model_names):
        plt.plot(predictions_full_df[name], label=f'{name} Predictions', color=colors[i], linewidth=1.5)
    plt.title('Full Dataset: Model Predictions vs. Original Data')
    plt.xlabel('Time')
    plt.ylabel(target_variable)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'full_predictions.png'))
    plt.show()

    # Plotting the results for train set
    plt.figure(figsize=(14, 10))
    plt.plot(y_train_inverse, label='Original Data', color='black', linestyle='--', linewidth=1.5)
    for i, name in enumerate(model_names):
        plt.plot(predictions_train_df[name], label=f'{name} Predictions', color=colors[i], linewidth=1.5)
    plt.title('Train Set: Model Predictions vs. Original Data')
    plt.xlabel('Time')
    plt.ylabel(target_variable)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'train_predictions.png'))
    plt.show()

    # Plotting the results for test set
    plt.figure(figsize=(14, 10))
    plt.plot(y_test_inverse, label='Original Data', color='black', linestyle='--', linewidth=1.5)
    for i, name in enumerate(model_names):
        plt.plot(predictions_test_df[name], label=f'{name} Predictions', color=colors[i], linewidth=1.5)
    plt.title('Test Set: Model Predictions vs. Original Data')
    plt.xlabel('Time')
    plt.ylabel(target_variable)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'test_predictions.png'))
    plt.show()

    # Save the predictions to CSV files
    predictions_full_df.to_csv(os.path.join(output_dir, 'full_predictions.csv'), index=False)
    predictions_train_df.to_csv(os.path.join(output_dir, 'train_predictions.csv'), index=False)
    predictions_test_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)

    return predictions_full_df, predictions_train_df, predictions_test_df

# Example usage
if __name__ == "__main__":
    # Assuming df is the DataFrame with the data
    data = df
    model_names = ['CNN', 'RNN', 'GRU', 'LSTM']
    target_variable = 'VIX'
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    predictions_full_df, predictions_train_df, predictions_test_df = predictor_module(data, model_names, target_variable, output_dir)
    print(predictions_full_df.head())
    print(predictions_train_df.head())
    print(predictions_test_df.head())
