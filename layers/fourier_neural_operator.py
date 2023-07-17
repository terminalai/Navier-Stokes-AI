import tensorflow as tf
keras = tf.keras

from keras.layers import *
from keras.models import *


class FourierIntegralLayer(Layer):
    """
    Implements the fourier integral transform, the heart of the fourier neural operator
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
        if not isinstance(input_shape[0], tf.TensorShape):
            self.num_params = 0
        else:
            self.num_params = input_shape[0][1]
            input_shape = input_shape[-1]

        self.input_dim = len(input_shape) - 2  # -1 for channels, -1 for batch dimension
        self.output_dim = input_shape[-1]

        kernel_dim = self.k_max ** self.input_dim * self.output_dim * self.output_dim
        self.real_mlp = Sequential([
            Dense(
                self.mlp_hidden_units,
                activation=self.activation,
                input_shape=(max(self.num_params, 1),),
                kernel_initializer="zeros"
            ),
            Dense(kernel_dim)
        ])

        self.complex_mlp = Sequential([
            Dense(
                self.mlp_hidden_units,
                activation=self.activation,
                input_shape=(max(self.num_params, 1),),
                kernel_initializer="zeros"
            ),
            Dense(kernel_dim)
        ])

    def call(self, inputs, *args, **kwargs):
        if self.num_params == 0:
            f = inputs
            parameters = tf.zeros((tf.shape(f)[0], 1))
        else:
            parameters, f = inputs

        # getting shape of inputs
        batch_size = tf.shape(f)[0]
        n = tf.shape(f)[1]

        # converting inputs into complex numbers
        x = f  # tf.cast(f, dtype=tf.complex64)

        if self.input_dim == 1:
            x = tf.transpose(x, (0, 2, 1))

            # fourier transform
            x = tf.signal.rfft(x)[:, :, :self.k_max]

            # build kernel
            real_kernel = tf.reshape(self.real_mlp(parameters), (-1, self.k_max, self.output_dim, self.output_dim))
            complex_kernel = tf.reshape(self.complex_mlp(parameters), (-1, self.k_max, self.output_dim, self.output_dim))
            kernel = tf.complex(real_kernel, complex_kernel)

            # my excessive knowledge and love of einstein notation is finally useful
            x = tf.einsum("bxio,bix->box", kernel, x)

            # inverse fourier transform
            x = tf.concat([x, tf.zeros((batch_size, self.output_dim, n - self.k_max), dtype=tf.complex64)], axis=-1)
            x = tf.signal.irfft(x)

            x = tf.transpose(x, (0, 2, 1))
        elif self.input_dim == 2:
            x = tf.transpose(x, (0, 3, 1, 2))

            # fourier transform
            x = tf.signal.rfft2d(x)[:, :, :self.k_max, :self.k_max]

            # build kernel
            real_kernel = tf.reshape(self.real_mlp(parameters), (-1, self.k_max, self.k_max, self.output_dim, self.output_dim))
            complex_kernel = tf.reshape(self.complex_mlp(parameters), (-1, self.k_max, self.k_max, self.output_dim, self.output_dim))
            kernel = tf.complex(real_kernel, complex_kernel)

            # my excessive knowledge and love of einstein notation is finally useful
            x = tf.einsum("bxyio,bixy->boxy", kernel, x)

            # inverse fourier transform
            x = tf.pad(x, tf.constant([[0, 0], [0, 0], [0, f.shape[1] - self.k_max], [0, f.shape[2] - self.k_max]]))
            x = tf.signal.irfft2d(x)

            x = tf.transpose(x, (0, 2, 3, 1))

        return tf.cast(x, dtype=tf.float32)


class FourierLayer(Layer):
    """
    Implements the fourier layer
    """
    def __init__(self, activation="swish", k_max=16, mlp_hidden_units=16, **kwargs):
        super().__init__(**kwargs)

        self.dim = 1  # the number of dimensions of the problem
        self.num_params = 0  # the number of non-function parameters in the problem

        self.k_max = k_max  # the number of modes to truncate
        self.mlp_hidden_units = mlp_hidden_units  # the number of hidden units used to parameterise the fourier kernel

        self.linear_transform = None  # the linear transform W
        self.fourier_integral_layer = None  # the heart of the fourier neural operator

        self.activation = Activation(activation)

    def build(self, input_shape):
        if not isinstance(input_shape[0], tf.TensorShape):
            self.num_params = 0
        else:
            self.num_params = input_shape[0][1]
            input_shape = input_shape[-1]

        self.dim = len(input_shape) - 2  # -1 for channels, -1 for batch dimension
        self.linear_transform = Dense(input_shape[-1])
        self.fourier_integral_layer = FourierIntegralLayer(
            mlp_hidden_units=self.mlp_hidden_units,
            k_max=self.k_max,
            activation=self.activation
        )

    def call(self, inputs, *args, **kwargs):
        if self.num_params == 0:
            f = inputs
        else:
            parameters, f = inputs

        return self.activation(
            self.linear_transform(f) +
            self.fourier_integral_layer(inputs)
        )


class FactorisedFourierLayer(Layer):
    """
    Implements the factorised fourier layer (Tran et al., 2022)
    """

    def __init__(self, activation="swish", k_max=16, mlp_hidden_units=16, weight_sharing=False, **kwargs):
        super().__init__(**kwargs)

        self.dim = 1  # the number of dimensions of the problem
        self.num_params = 0  # the number of non-function parameters in the problem

        self.k_max = k_max  # the number of modes to truncate
        self.mlp_hidden_units = mlp_hidden_units  # the number of hidden units used to parameterise the fourier kernel

        self.weight_sharing = weight_sharing  # should the weights of the fourier networks be shared

        self.linear_transform = None  # the linear transform W1
        self.linear_transform_2 = None  # the linear transform W2

        # the heart of the fourier neural operator
        self.fourier_integral_layer = None
        self.fourier_integral_layer_2 = None

        self.activation = activation

    def build(self, input_shape):
        if not isinstance(input_shape[0], tf.TensorShape):
            self.num_params = 0
        else:
            self.num_params = input_shape[0][1]
            input_shape = input_shape[-1]

        self.dim = len(input_shape) - 2  # -1 for channels, -1 for batch dimension

        self.linear_transform = Dense(input_shape[-1], activation=self.activation)
        self.linear_transform_2 = Dense(input_shape[-1], activation=self.activation)

        self.fourier_integral_layer = FourierIntegralLayer(
            mlp_hidden_units=self.mlp_hidden_units,
            k_max=self.k_max,
            activation=self.activation
        )

        if self.weight_sharing:
            self.fourier_integral_layer_2 = self.fourier_integral_layer
        else:
            self.fourier_integral_layer_2 = FourierIntegralLayer(
                mlp_hidden_units=self.mlp_hidden_units,
                k_max=self.k_max,
                activation=self.activation
            )

    def call(self, inputs, *args, **kwargs):
        if self.num_params == 0:
            x = inputs
        else:
            parameters, x = inputs

        if self.dim == 1:
            x = self.fourier_integral_layer(inputs)
        elif self.dim == 2:
            # first dimension
            x1 = tf.reshape(x, (-1, x.shape[1] * x.shape[2], x.shape[3]))
            x1 = self.fourier_integral_layer(x1 if self.num_params == 0 else [parameters, x1])
            x1 = tf.reshape(x1, (-1, x.shape[1], x.shape[2], x.shape[3]))

            # second dimension
            x2 = tf.transpose(x, (0, 2, 1, 3))

            shape = x2.shape
            x2 = tf.reshape(x2, (-1, shape[1] * shape[2], shape[3]))
            x2 = self.fourier_integral_layer(x2 if self.num_params == 0 else [parameters, x2])
            x2 = tf.reshape(x2, (-1, shape[1], shape[2], shape[3]))

            x2 = tf.transpose(x2, (0, 2, 1, 3))

            # adding them together
            x = x1 + x2

        return self.linear_transform_2(self.linear_transform(x)) + (inputs if self.num_params == 0 else inputs[-1])
