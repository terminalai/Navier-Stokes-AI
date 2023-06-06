import tensorflow as tf
keras = tf.keras

from keras.layers import *


class FourierIntegralLayer(Layer):
    """
    Implements the fourier integral transform, the heart of the fourier neural operator
    """
    def __init__(self, k_max=16, **kwargs):
        super().__init__(**kwargs)

        self.k_max = k_max
        self.kernel = None

        self.input_dim = 1
        self.output_dim = 1

    def build(self, input_shape):
        self.input_dim = len(input_shape) - 2  # -1 for channels, -1 for batch dimension
        self.output_dim = input_shape[-1]

        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.kernel = tf.Variable(
            tf.complex(initializer((self.k_max, self.output_dim, self.output_dim)),
                       initializer((self.k_max, self.output_dim, self.output_dim))),
            shape=tf.TensorShape((self.k_max, self.output_dim, self.output_dim)),
            dtype=tf.complex64
        )

    def call(self, inputs, *args, **kwargs):
        batch_size = tf.shape(inputs)[0]
        n = tf.shape(inputs)[1]

        x = tf.cast(inputs, dtype=tf.complex64)
        x = tf.transpose(x, (0, 2, 1))

        x = tf.signal.fft(x)[:, :, :self.k_max]  # fourier transform
        x = tf.transpose(x, (0, 2, 1))

        # todo do a legit matmul
        x = tf.linalg.matmul(
            self.kernel,
            tf.repeat(tf.expand_dims(x, axis=-1), self.output_dim, axis=-1)
        )  # apply fourier kernel
        x = tf.reduce_sum(x, axis=-1)

        x = tf.transpose(x, (0, 2, 1))
        x = tf.concat([x, tf.zeros((batch_size, self.output_dim, n - self.k_max), dtype=tf.complex64)], axis=-1)
        x = tf.signal.ifft(x)  # inverse fourier transform

        x = tf.transpose(x, (0, 2, 1))
        return tf.cast(x, dtype=tf.float32)


class FourierLayer(Layer):
    """
    Implements the fourier layer
    """
    def __init__(self, activation="swish", k_max=16, **kwargs):
        super().__init__(**kwargs)

        self.dim = 1  # the number of dimensions of the problem
        self.k_max = k_max

        self.linear_transform = None  # the linear transform W
        self.fourier_integral_layer = None  # the heart of the fourier neural operator

        self.activation = Activation(activation)

    def build(self, input_shape):
        self.dim = len(input_shape) - 2  # -1 for channels, -1 for batch dimension
        self.linear_transform = Dense(input_shape[-1])
        self.fourier_integral_layer = FourierIntegralLayer(k_max=self.k_max)

    def call(self, inputs, *args, **kwargs):
        return self.activation(
            self.linear_transform(inputs) + self.fourier_integral_layer(inputs)
        )
