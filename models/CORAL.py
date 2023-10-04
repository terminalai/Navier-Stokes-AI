import keras_core
import keras_core.ops as ops

import tensorflow as tf

from keras_core.layers import *
from keras_core.models import *

from layers import SIREN


class CORALAutoencoder(Model):
    def __init__(
        self,
        latent_size,
        optimizer="adamw",
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.latent_size = latent_size
        self.optimizer = keras_core.optimizers.get(optimizer)

        self.INR = None
        self.latent = None

    def build(self, input_shape):
        output_dim = input_shape[1][-1]

        self.INR = SIREN(
            widths=[256] * 5,
            omega_0=30.0,
            output_dim=output_dim,
            use_latent=True,
            name="siren",
        )
        self.latent = tf.Variable(initial_value=ops.zeros((input_shape[0][0], self.latent_size)), dtype=tf.float32)

    def call(self, inputs, training=False):
        return self.INR(inputs)

    def train_step(self, data):
        (points, values), _ = data  # split into the positions of the points and their values
        batch_size = ops.shape(points)[0]

        # initialise latent
        self.latent.assign(ops.zeros((batch_size, self.latent_size)))
        for i in range(100):
            with tf.GradientTape() as tape:
                y_pred = self((points, self.latent), training=False)
                loss = ops.mean(ops.square(y_pred - values))

            # compute gradients
            gradients = tape.gradient(loss, self.latent)

            # update latent code
            self.latent.assign_add(-0.001 * gradients)   # self.optimizer.apply(gradients, latent)

        (x, y), _ = data  # split into the positions of the points and their values
        with tf.GradientTape() as tape:
            y_pred = self((x, self.latent), training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply(gradients, trainable_vars)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


if __name__ == "__main__":
    import keras_core

    import numpy as np
    from scipy.io import loadmat

    model = SIREN(
        widths=[256, 256, 256, 256, 256],
        output_dim=1,
        use_latent=True
    )

    model = CORALAutoencoder(latent_size=256)
    model((tf.random.normal((16, 1024, 1)), tf.random.normal((16, 256))))
    model.summary()

    model.compile(
        optimizer=keras_core.optimizers.AdamW(learning_rate=5e-3), loss="mse"
    )

    data = loadmat("../burgers_data_R10.mat")

    x = data['a'].astype("float32")  # input
    y = data['u'].astype("float32")  # target

    x = np.expand_dims(x, axis=-1)[:, ::8192 // 1024]
    y = np.expand_dims(y, axis=-1)[:, ::8192 // 1024]

    x_train, y_train = x[:-len(x) // 8], y[:-len(y) // 8]
    x_test, y_test = x[-len(x) // 8:], y[-len(y) // 8:]

    pts_train = tf.repeat(
        tf.range(len(x_train[0]), dtype="float32")[tf.newaxis, ..., tf.newaxis] / 1024.0,
        len(x_train), axis=0
    )

    model.fit((pts_train, x_train), x_train, epochs=100, batch_size=16, validation_data=(x_test, y_test))
