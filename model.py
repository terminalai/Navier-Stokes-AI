import random
import numpy as np
import tensorflow as tf
keras = tf.keras

from keras.layers import *
from keras.models import *

from layers import FourierLayer


def FourierNeuralOperator(num_params, input_shape, k_max=16, dim=64, num_layers=4, activation="swish"):
    parameters = Input((num_params,))
    function = Input(input_shape)

    x = Dense(2 * dim, activation=activation)(function)
    x = Dense(dim)(x)

    for i in range(num_layers):
        x = FourierLayer(k_max=k_max, activation=activation)([parameters, x])

    x = Dense(input_shape[-1])(x)

    return Model(inputs=(parameters, function), outputs=x)


if __name__ == "__main__":
    model = FourierNeuralOperator(num_params=1, input_shape=(500, 1))
    model.summary()

    # test the model by making it learn the differential operator
    x_train = []
    y_train = []
    for i in range(2048):
        coefficients = [random.uniform(0, 1) for j in range(10)]
        y = [sum([coefficients[j] * (x / 500) ** j for j in range(10)]) for x in range(500)]

        differentiated_coefficients = [random.uniform(0, 1) * j for j in range(10)]
        dy_dx = [sum([differentiated_coefficients[j] * (x / 500) ** (j - 1) for j in range(1, 10)]) for x in range(500)]

        x_train.append(y)
        y_train.append(dy_dx)

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test, y_test = x_train[1536:], y_train[1536:]
    x_train, y_train = x_train[:1536], y_train[:1536]

    # train the model
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), loss="mse")
    model.fit(
        (np.ones((len(x_train),)), x_train), y_train,
        epochs=20, batch_size=16,
        validation_data=((np.ones((len(x_test),)), x_test), y_test)
    )
