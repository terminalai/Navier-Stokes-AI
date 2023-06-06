import random
import numpy as np
import tensorflow as tf
keras = tf.keras

from keras.layers import *
from keras.models import *

from layers import FourierLayer


def FourierNeuralOperator(input_shape, k_max=16, dim=64, num_layers=4, activation="swish"):
    inputs = Input(input_shape)
    x = Dense(2 * dim, activation=activation)(inputs)
    x = Dense(dim)(x)

    for i in range(num_layers):
        x = FourierLayer(k_max=k_max, activation=activation)(x)

    x = Dense(input_shape[-1])(x)

    model = Model(inputs=inputs, outputs=x)
    return model


if __name__ == "__main__":
    model = FourierNeuralOperator(input_shape=(500, 1))

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
    model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))
