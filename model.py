import random
import numpy as np
import tensorflow as tf
keras = tf.keras

from keras.models import *
from keras.layers import *

from layers import FourierLayer


class FourierNeuralOperator(Model):
    def __init__(
            self,
            num_params,
            input_shape,
            num_outputs=-1,
            k_max=16,
            dim=64,
            num_layers=4,
            activation="swish",
            periodic=False,
            physics_loss=None,
            *args, **kwargs
    ):
        super(FourierNeuralOperator, self).__init__(*args, **kwargs)

        self.num_params = num_params
        self.k_max = k_max
        self.dim = dim
        self.num_layers = num_layers
        self.num_outputs = input_shape[-1] if num_outputs < 0 else num_outputs
        self.activation = activation
        self.periodic = periodic
        self.physics_loss = physics_loss

        self.function_embedding = Dense(self.dim, name="function_embedding")

        self.output_projection = Sequential(
            [
                Dense(256, activation=self.activation),
                Dense(self.num_outputs)
            ], name="output_projection"
        )

        self.fourier_layers = [
            FourierLayer(k_max=self.k_max, activation=self.activation, mlp_hidden_units=1)
            for _ in range(self.num_layers)
        ]

        self.physics_loss_tracker = keras.metrics.Mean(name='physics_loss')

    def call(self, inputs, training=None, mask=None):
        if self.num_params == 0:
            function = inputs
        else:
            parameters, function = inputs

        if self.periodic:
            x = tf.pad(function, [[0, 0], [0, 2], [0, 0]], "CONSTANT")  # pad for non-periodic BCs
        else:
            x = function

        if self.num_params > 0:  # add parameters
            x = tf.concat([x, tf.repeat(tf.expand_dims(parameters, axis=1), tf.shape(x)[1], axis=1)], axis=-1)

        x = self.function_embedding(x)  # project to higher dimension

        # applying fourier layers
        for layer in self.fourier_layers:
            x = layer(x)

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


    def _f(t):
        return t * t * t * (t * (t * 6 - 15) + 10)


    def generate_perlin_noise_2d(batch_size, shape, res):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = tf.meshgrid(tf.range(0, res[0], delta[0]),
                           tf.range(0, res[1], delta[1]), indexing='ij')
        grid = tf.stack(grid, axis=-1)
        grid = grid - tf.floor(grid)
        grid = tf.cast(grid, tf.float32)

        angles = tf.random.uniform(shape=(batch_size, res[0] + 1, res[1] + 1), maxval=2 * np.pi)
        gradients = tf.stack((tf.cos(angles), tf.sin(angles)), axis=-1)

        gradients = tf.repeat(tf.repeat(gradients, repeats=d[0], axis=1), repeats=d[1], axis=2)
        g00 = gradients[:, :-d[0], :-d[1]]
        g10 = gradients[:, d[0]:, :-d[1]]
        g01 = gradients[:, :-d[0], d[1]:]
        g11 = gradients[:, d[0]:, d[1]:]

        # Ramps
        n00 = tf.reduce_sum(tf.stack((grid[:, :, 0], grid[:, :, 1]), axis=-1) * g00, axis=3)
        n10 = tf.reduce_sum(tf.stack((grid[:, :, 0] - 1, grid[:, :, 1]), axis=-1) * g10, axis=3)
        n01 = tf.reduce_sum(tf.stack((grid[:, :, 0], grid[:, :, 1] - 1), axis=-1) * g01, axis=3)
        n11 = tf.reduce_sum(tf.stack((grid[:, :, 0] - 1, grid[:, :, 1] - 1), axis=-1) * g11, axis=3)

        # Interpolation
        t = _f(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        return 6.21908435118 * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)  # for a std dev of 1


    ds = tf.data.Dataset.range(256).map(
        lambda x: (10 ** -tf.random.uniform((64, 1), 0, 3), generate_perlin_noise_2d(64, (512, 1), (32, 1)))
    )

    # build the model
    model = FourierNeuralOperator(
        num_params=1,
        input_shape=(512, 1),
        periodic=True,
        physics_loss=physics_loss(
            lambda params, lst: lst[0] + lst[1] * lst[2] - params[:, 0] * lst[3], 2, 1 / 512, 0.01, num_params=1
        )
    )

    # train the model
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=4e-3,
        decay_steps=512,
        decay_rate=0.85
    )

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr))
    history = model.fit(ds, epochs=100)

    model.save_weights("model.h5")
