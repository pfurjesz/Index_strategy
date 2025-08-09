import tensorflow as tf
from tensorflow.keras import layers


class Autoencoder(tf.keras.models.Model):
  def __init__(self, num_timesteps, num_inputs, num_hidden, kernel_size, pooling):
    super(Autoencoder, self).__init__()
    self.num = num_timesteps
    self.lb = kernel_size
    self.pooling =pooling

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