import keras_core.ops as ops
import tensorflow as tf
from keras_core.models import *

from layers import SIREN


class CORALAutoencoder(Model):
    def __init__(
        self,
        latent_size=256,
        widths=(256,) * 5,
        omega_0=30.0,
        alpha=1e-3,
        steps=100,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.latent_size = latent_size
        self.widths = widths
        self.omega_0 = omega_0

        self.alpha = alpha
        self.steps = steps

        self.INR = None
        self.latent = None

    def build(self, input_shape):
        output_dim = input_shape[1][-1]

        self.INR = SIREN(
            widths=self.widths,
            omega_0=self.omega_0,
            output_dim=output_dim,
            use_latent=True,
            name="siren",
        )
        self.latent = tf.Variable(
            initial_value=ops.zeros((input_shape[0][0], self.latent_size)),
            trainable=True,
            dtype=tf.float32,
            name="latent_code"
        )

    def call(self, inputs, training=False, **kwargs):
        return self.INR(inputs, training=training, **kwargs)

    def encode(self, inputs, training=False, **kwargs):
        points, values = inputs  # split into the positions of the points and their values
        batch_size = ops.shape(points)[0]

        # initialise latent
        self.latent.assign(ops.zeros((batch_size, self.latent_size)))
        for i in range(self.steps):
            with tf.GradientTape() as tape:
                y_pred = self((points, self.latent), training=training, **kwargs)
                loss = ops.mean(ops.square(y_pred - values))

            # compute gradients
            gradients = tape.gradient(loss, self.latent)

            # update latent code
            self.latent.assign_add(-self.alpha * gradients)

        return self.latent

    def train_step(self, data):
        (x, y), _ = data  # split into the positions of the points and their values
        batch_size = ops.shape(x)[0]

        # initialise latent
        latent_constant = ops.zeros((batch_size, self.latent_size))
        self.latent.assign(latent_constant)  # we need a variable to nab the gradients
        with tf.GradientTape() as tape:
            for i in range(self.steps):
                with tf.GradientTape() as tape2:  # inner loop for optimising the latent code
                    y_pred = self((x, self.latent), training=True)
                    loss = ops.mean(ops.square(y_pred - y))

                # compute gradients
                gradients = tape2.gradient(loss, self.latent)

                # update latent code
                self.latent.assign_add(-self.alpha * gradients)
                latent_constant = latent_constant - self.alpha * gradients

            y_pred = self((x, latent_constant), training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # update weights
        self.optimizer.apply(gradients, trainable_vars)

        # update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        self.encode(data[0])  # getting the latent

        (x, y), _ = data  # split into the positions of the points and their values
        y_pred = self((x, self.latent), training=False)
        loss = self.compute_loss(y=y, y_pred=y_pred)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class CORAL(Model):
    def __init__(
        self,
        latent_model: Model,
        input_autoencoder: CORALAutoencoder,
        output_autoencoder: CORALAutoencoder,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.latent_model = latent_model

        self.input_autoencoder = input_autoencoder
        self.output_autoencoder = output_autoencoder

    def call(self, inputs, training=False, use_latent=False):
        if not use_latent:
            latent = self.input_autoencoder.encode(inputs)
        else:
            latent = inputs

        new_latent = self.latent_model(latent)

        if not use_latent:
            output = self.output_autoencoder((inputs[0], new_latent))
        else:
            output = new_latent

        return output

    def train_step(self, data):
        x, y = data  # x is the input latent, y is the output latent
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True, use_latent=True)
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

    def test_step(self, data):
        x, y = data  # x is the input latent, y is the output latent

        y_pred = self(x, training=False, use_latent=True)
        loss = self.compute_loss(y=y, y_pred=y_pred)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


if __name__ == "__main__":
    import os
    os.environ["KERAS_BACKEND"] = "tensorflow"

    import keras_core

    import numpy as np
    from scipy.io import loadmat

    model = CORALAutoencoder(
        latent_size=256,
        alpha=1e-3,
        steps=3
    )
    model((tf.random.normal((16, 1024, 1)), tf.random.normal((16, 256))))
    model.summary()

    model.compile(
        optimizer=keras_core.optimizers.AdamW(learning_rate=1e-3), loss="mse"
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

    model.fit((pts_train, x_train), x_train, epochs=50, batch_size=16)
    model.save_weights("input_encoder.keras.h5")

    model2 = CORALAutoencoder(
        latent_size=256,
        alpha=1e-3,
        steps=3
    )
    model2((tf.random.normal((16, 1024, 1)), tf.random.normal((16, 256))))
    model2.summary()

    model2.compile(
        optimizer=keras_core.optimizers.AdamW(learning_rate=5e-3), loss="mse"
    )
    model2.fit((pts_train, y_train), y_train, epochs=50, batch_size=16)
    model.save_weights("output_encoder.keras.h5")
