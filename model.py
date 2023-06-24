import random
import numpy as np
import tensorflow as tf
keras = tf.keras

from keras.models import *
from keras.layers import *

from layers import FourierLayer


def FourierNeuralOperator2(num_params, input_shape,
                          mlp_hidden_units=1, k_max=16, dim=64, num_layers=4,
                          activation="swish", periodic=False, initial_embedding=False):
    """
    Implements the fourier neural operator (Li et al. 2021, https://arxiv.org/pdf/2010.08895.pdf)
    :param num_params: The number of constant parameters (e.g. viscosity, density)
    :param input_shape: The shape of the input
    :param mlp_hidden_units: The number of hidden units used to parameterise the fourier kernel
    :param k_max: The number of modes to use
    :param dim: The number of dimensions to expand to
    :param num_layers: The number of fourier layers to use
    :param activation: The activation function to use
    :param periodic: Are the boundary conditions periodic?
    :param initial_embedding: Should the parameter used be embedded initially?
    :return: Returns the fourier neural operator model
    """

    parameters = Input((num_params,), name="parameter_input")
    function = Input(input_shape, name="function_input")

    if periodic:
        x = tf.pad(function, [[0, 0], [0, 2], [0, 0]], "CONSTANT")  # pad for non-periodic BCs
    else:
        x = function

    x = Dense(dim, name="function_embedding")(x)  # project to higher dimension

    # use or don't use an initial embedding
    if initial_embedding:
        x2 = Dense(dim, activation=activation, name="function_embedding_1")(parameters)
        x2 = Dense(dim, name="function_embedding_2")(x2)
        x2 = tf.repeat(tf.expand_dims(x2, axis=1), x.shape[1], axis=1)

        x = x + x2

    # applying fourier layers
    for i in range(num_layers):
        x = FourierLayer(k_max=k_max, activation=activation, mlp_hidden_units=mlp_hidden_units)([parameters, x])

    # projecting back to original dimension
    x = Dense(256, activation=activation, name="output_projection_1")(x)
    x = Dense(input_shape[-1], name="output_projection_2")(x)

    if periodic:
        x = x[:, :-2]  # remove padding

    return Model(inputs=(parameters, function), outputs=x)


class FourierNeuralOperator(Model):
    def __init__(
            self,
            num_params,
            input_shape,
            mlp_hidden_units=1,
            k_max=16,
            dim=64,
            num_layers=4,
            activation="swish",
            periodic=False,
            initial_embedding=False,
            physics_loss=None,
            *args, **kwargs
    ):
        super(FourierNeuralOperator, self).__init__(*args, **kwargs)

        self.num_params = num_params
        self.mlp_hidden_units = mlp_hidden_units
        self.k_max = k_max
        self.dim = dim
        self.num_layers = num_layers
        self.activation = activation
        self.periodic = periodic
        self.initial_embedding = initial_embedding
        self.physics_loss = physics_loss

        self.function_embedding = Dense(self.dim, name="function_embedding")

        if self.num_params > 0:
            self.parameter_embedding = Sequential(
                [
                    Dense(self.dim, activation=self.activation),
                    Dense(self.dim)
                ], name="parameter_embedding"
            )

        self.output_projection = Sequential(
            [
                Dense(256, activation=self.activation),
                Dense(input_shape[-1])
            ], name="output_projection"
        )

        self.fourier_layers = [
            FourierLayer(k_max=self.k_max, activation=self.activation, mlp_hidden_units=self.mlp_hidden_units)
            for _ in range(self.num_layers)
        ]

        self.physics_loss_tracker = keras.metrics.Mean(name='physics_loss')

    def call(self, inputs, training=None, mask=None):
        if self.num_params == 0:
            parameters = None
            function = inputs
        else:
            parameters, function = inputs

        if self.periodic:
            x = tf.pad(function, [[0, 0], [0, 2], [0, 0]], "CONSTANT")  # pad for non-periodic BCs
        else:
            x = function

        x = self.function_embedding(x)  # project to higher dimension

        # you can only have an initial embedding for the parameters if you provide them
        assert (self.initial_embedding and self.num_params > 0) or not self.initial_embedding

        # use or don't use an initial embedding
        if self.initial_embedding:
            x2 = self.parameter_embedding(parameters)
            x2 = tf.repeat(tf.expand_dims(x2, axis=1), x.shape[1], axis=1)

            x = x + x2

        # applying fourier layers
        for layer in self.fourier_layers:
            if parameters is None:
                x = layer(x)
            else:
                x = layer([parameters, x])

        # projecting back to original dimension
        x = self.output_projection(x)

        if self.periodic:
            x = x[:, :-2]  # remove padding

        return x

    def train_step(self, data):
        if self.physics_loss is not None:
            x = data

            with tf.GradientTape() as tape:
                y = self(x, training=True)  # Forward pass

                # Compute the loss value
                # (the loss function is configured in `compile()`)
                loss = self.physics_loss(x, y)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update metrics
            self.physics_loss_tracker.update_state(loss)

            # Return a dict mapping metric names to current value
            return {m.name: m.result() for m in self.metrics}
        else:
            super().train_step(data)


if __name__ == "__main__":
    from losses import physics_loss

    """
    # test the model by making it learn the differential operator
    x_train = []
    order_lst = []
    y_train = []
    for i in tqdm.trange(2 ** 16):
        coefficients = [random.uniform(-5, 5) / (0.05 * j + 1) for j in range(20)]
        y = np.polynomial.polynomial.Polynomial(coefficients)(np.arange(-5 / 500, 1.2, 1 / 500))

        x_train.append(y[5:505])

        order = random.randint(1, 5)
        order_lst.append(order)

        y_train.append(np.diff(y, n=order)[5 - order:500 + 5 - order] * (100 ** (0.5 * order + 1)))

    x_train, order_lst, y_train = np.array(x_train), np.array(order_lst), np.array(y_train)
    x_test, order_test, y_test = x_train[9 * len(x_train) // 10:], \
        order_lst[9 * len(x_train) // 10:], y_train[9 * len(x_train) // 10:]
    x_train, order_train, y_train = x_train[:9 * len(x_train) // 10], \
        order_lst[:9 * len(x_train) // 10], y_train[:9 * len( x_train) // 10]
    """

    ds = tf.data.Dataset.range(2000).map(lambda x: tf.random.uniform((500, 1)))

    # build the model
    model = FourierNeuralOperator(
        num_params=0,
        input_shape=(500, 1),
        physics_loss=physics_loss(lambda lst: lst[0] + lst[1] * lst[2] - 0.01 * lst[3], 2, 1 / 500, 0.01)
    )

    # train the model
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4))
    history = model.fit(
        ds, epochs=20, batch_size=64
    )

    model.save_weights("model.h5")
