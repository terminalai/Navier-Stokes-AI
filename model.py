import random
import numpy as np
import tensorflow as tf
keras = tf.keras

from keras.models import *
from keras.layers import *

from layers import FourierLayer


def FourierNeuralOperator(num_params, input_shape,
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


if __name__ == "__main__":
    import tqdm

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

    # build the model
    model = FourierNeuralOperator(num_params=1, input_shape=(500, 1))
    model.summary()

    # train the model
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), loss="mse")

    history = model.fit(
        (order_train, x_train), y_train,
        epochs=50, batch_size=64,
        validation_data=((order_test, x_test), y_test)
    )
