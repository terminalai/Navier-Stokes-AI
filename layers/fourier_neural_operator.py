import tensorflow as tf
keras = tf.keras

from keras.layers import *
from keras.models import *


class FourierIntegralLayer(Layer):
    """
    Implements the fourier integral transform, the heart of the fourier neural operator
    todo: implement 2d and 3d
    """
    def __init__(self, k_max=16, mlp_hidden_units=10, activation="swish", **kwargs):
        super().__init__(**kwargs)

        self.k_max = k_max
        self.kernel = None

        self.input_dim = 1
        self.output_dim = 1
        self.num_params = 0

        self.real_mlp = None
        self.complex_mlp = None

        self.activation = activation
        self.mlp_hidden_units = mlp_hidden_units

    def build(self, input_shape):
        self.num_params = input_shape[0][1]

        input_shape = input_shape[-1]
        self.input_dim = len(input_shape) - 2  # -1 for channels, -1 for batch dimension
        self.output_dim = input_shape[-1]

        self.real_mlp = Sequential([
            Dense(
                self.mlp_hidden_units,
                activation=self.activation,
                input_shape=(self.num_params,)
            ),
            Dense(self.k_max * self.output_dim * self.output_dim)
        ])

        self.complex_mlp = Sequential([
            Dense(
                self.mlp_hidden_units,
                activation=self.activation,
                input_shape=(self.num_params,)
            ),
            Dense(self.k_max * self.output_dim * self.output_dim)
        ])

    def call(self, inputs, *args, **kwargs):
        parameters, f = inputs

        # getting shape of inputs
        batch_size = tf.shape(f)[0]
        n = tf.shape(f)[1]

        # converting inputs into complex numbers
        x = tf.cast(f, dtype=tf.complex64)
        x = tf.transpose(x, (0, 2, 1))

        # fourier transform
        x = tf.signal.fft(x)[:, :, :self.k_max]
        x = tf.transpose(x, (0, 2, 1))

        # build kernel
        real_kernel = tf.reshape(self.real_mlp(parameters), (-1, self.k_max, self.output_dim, self.output_dim))
        complex_kernel = tf.reshape(self.complex_mlp(parameters), (-1, self.k_max, self.output_dim, self.output_dim))
        kernel = tf.complex(real_kernel, complex_kernel)

        # todo do a legit matmul
        x = tf.linalg.matmul(
            kernel,
            tf.repeat(tf.expand_dims(x, axis=-1), self.output_dim, axis=-1)
        )  # apply fourier kernel
        x = tf.reduce_sum(x, axis=-1)

        # inverse fourier transform
        x = tf.transpose(x, (0, 2, 1))
        x = tf.concat([x, tf.zeros((batch_size, self.output_dim, n - self.k_max), dtype=tf.complex64)], axis=-1)
        x = tf.signal.ifft(x)

        x = tf.transpose(x, (0, 2, 1))
        return tf.cast(x, dtype=tf.float32)


class FourierLayer(Layer):
    """
    Implements the fourier layer
    """
    def __init__(self, activation="swish", k_max=16, **kwargs):
        super().__init__(**kwargs)

        self.dim = 1  # the number of dimensions of the problem
        self.num_params = 0  # the number of non-function parameters in the problem

        self.k_max = k_max

        self.linear_transform = None  # the linear transform W
        self.fourier_integral_layer = None  # the heart of the fourier neural operator

        self.activation = Activation(activation)

    def build(self, input_shape):
        self.num_params = input_shape[0][1]

        input_shape = input_shape[-1]
        self.dim = len(input_shape) - 2  # -1 for channels, -1 for batch dimension
        self.linear_transform = Dense(input_shape[-1])
        self.fourier_integral_layer = FourierIntegralLayer(k_max=self.k_max)

    def call(self, inputs, *args, **kwargs):
        parameters, f = inputs
        return self.activation(
            self.linear_transform(f) + self.fourier_integral_layer(inputs, activation=self.activation)
        )
