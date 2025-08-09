import sys
from pathlib import Path
parent_dir = Path().resolve().parent
sys.path.append(str(parent_dir))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class WindowGenerator():
    def __init__(self, input_width, label_width, shift, input_columns=None, label_columns=None, all_columns=None):
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.train_label_indices = {name: i for i, name in enumerate(all_columns)}

        self.input_columns = input_columns
        if input_columns is not None:
            self.input_columns_indices = {name: i for i, name in enumerate(input_columns)}
        self.train_input_indices = {name: i for i, name in enumerate(all_columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.input_columns is not None:
            inputs = tf.stack([inputs[:, :, self.train_input_indices[name]] for name in self.input_columns], axis=-1)
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.train_label_indices[name]] for name in self.label_columns], axis=-1)
        return inputs, labels

    def make_dataset(self, data, shuffle=False, batchsize=500):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data, targets=None, sequence_length=self.total_window_size,
            sequence_stride=1, sampling_rate=1, shuffle=shuffle, batch_size=batchsize
        )
        ds = ds.map(self.split_window)
        return ds

class Autoencoder(tf.keras.models.Model):
    def __init__(self, num_timesteps, num_inputs, num_hidden, kernel_size, pooling):
        super(Autoencoder, self).__init__()
        self.num = num_timesteps
        self.lb = kernel_size
        self.pooling = pooling

        encoder_input = tf.keras.Input(shape=(num_timesteps, num_inputs), name="input")
        x = tf.keras.layers.Conv1D(filters=num_hidden, kernel_size=kernel_size, activation=None, use_bias=True, padding='causal')(encoder_input)
        x = layers.MaxPooling1D(self.pooling, strides=self.pooling, padding='same')(x)
        self.encoder = tf.keras.Model(inputs=encoder_input, outputs=x)
        
        decoder_input = tf.keras.Input(shape=(int(num_timesteps/self.pooling), num_hidden))
        y = tf.keras.layers.Conv1DTranspose(filters=num_inputs, kernel_size=kernel_size, strides=self.pooling, activation=None, use_bias=True, padding='same')(decoder_input)
        self.decoder = tf.keras.Model(inputs=decoder_input, outputs=y)

    def call(self, input):
        u = self.encoder(input)
        decoded = self.decoder(u)
        return decoded

def main_ae(signals, targets, data):
    tf.random.set_seed(1234)

    all_columns = signals.copy()
    all_columns.append(targets)
    all_noise_col = [f'{col}_noise' for col in all_columns]
    normalizer = StandardScaler()

    date4Fig = data['Date']
    df_ = data.set_index(data['Date']).drop(columns='Date')
    dat = df_[all_columns]

    df = pd.DataFrame(normalizer.fit_transform(dat), columns=all_columns)
    df['Date'] = date4Fig
    df = df.set_index(df['Date']).drop(columns='Date')

    df_n = df + 1.0 * np.random.normal(0, 1, df.shape)
    df[all_noise_col] = df_n[all_columns].copy()
    n = len(df)
    nr_true_col = len(all_columns)

    lb = 30
    pooling = 1
    window = WindowGenerator(input_width=lb, label_width=lb, shift=0, input_columns=all_noise_col, label_columns=all_columns, all_columns=df.columns)
    td = window.make_dataset(df, shuffle=True)
    train_data = td.take(2)
    val_data = td.skip(2)

    model = Autoencoder(num_timesteps=lb, num_inputs=nr_true_col, num_hidden=2, kernel_size=25, pooling=pooling)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=50, decay_rate=0.97, staircase=True)
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.SGD(learning_rate=lr_schedule), metrics=[tf.metrics.MeanSquaredError()])
    model.run_eagerly = True
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='min')
    history = model.fit(train_data, validation_data=val_data, epochs=400, callbacks=[early_stopping])
    model.summary()

    fig, axs = plt.subplots()
    axs.plot(history.history['loss'])
    axs.plot(history.history['val_loss'])
    axs.legend(['training loss', 'validation loss'])

    window_orig = WindowGenerator(input_width=lb, label_width=lb, shift=0, input_columns=all_columns, label_columns=all_columns, all_columns=df.columns)
    fd = window_orig.make_dataset(df, shuffle=False)

    y_pred = model.predict(fd)
    u_true = np.concatenate([x for x, y in fd], axis=0)
    y_true = np.concatenate([y for x, y in fd], axis=0)
    mse_FD = ((y_pred - y_true) ** 2).mean()
    print('mse_FD ' + str(mse_FD))

    plt.figure(figsize=(1 * 6, (nr_true_col + 1) * 2))
    mp = -1
    du = pd.DataFrame(y_true[:, mp, :], index=df.index[lb - 1:])
    dpred = pd.DataFrame(y_pred[:, mp, :], index=df.index[lb - 1:])

    for index, signal in enumerate(all_columns):
        signal = signal.lstrip('_')
        plt.subplot(nr_true_col + 1, 1, index + 1)
        plt.plot(du.iloc[:, index], '-', linewidth=1)
        plt.plot(dpred.iloc[:, index], '-', linewidth=1)
        plt.legend([f'{signal} true', f'{signal} prediction'])

    middle = model.encoder(u_true)
    plt.subplot(nr_true_col + 1, 1, nr_true_col + 1)
    plt.plot((pd.DataFrame(middle[:, mp, :], index=df.index[lb - 1:])))
    plt.legend(['middle layer'])
    plt.show()

    pd.merge(dpred, dat.iloc[lb - 1:, :], on="Date").to_excel('./data/test/data_preprocessed.xlsx')


