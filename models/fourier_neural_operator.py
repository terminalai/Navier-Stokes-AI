import random
import numpy as np
import tensorflow as tf
keras = tf.keras

from keras.models import *
from keras.layers import *

from layers import FourierLayer, FactorisedFourierLayer


class FourierNeuralOperator(Model):
    def __init__(
            self,
            num_params,
            input_shape,
            num_outputs=-1,
            k_max=16,
            dim=64,
            num_layers=4,
            mlp_hidden_units=-1,
            activation="swish",
            fourier_layer=FourierLayer,
            size=None,
            periodic=False,
            physics_loss=None,
            *args, **kwargs
    ):
        super(FourierNeuralOperator, self).__init__(*args, **kwargs)

        self.num_params = num_params
        self.k_max = k_max
        self.dim = dim
        self.num_layers = num_layers
        self.mlp_hidden_units = mlp_hidden_units
        self.num_outputs = input_shape[-1] if num_outputs < 0 else num_outputs
        self.activation = activation
        self.size = size
        self.periodic = periodic
        self.physics_loss = physics_loss

        # checking some input conditions
        assert size is None or len(input_shape) - 1 == len(size)

        self.function_embedding = Dense(self.dim, name="function_embedding")

        self.output_projection = Sequential(
            [
                Dense(256, activation=self.activation),
                Dense(self.num_outputs)
            ], name="output_projection"
        )

        self.fourier_layers = [
            fourier_layer(
                k_max=self.k_max,
                activation=self.activation,
                mlp_hidden_units=max(self.mlp_hidden_units, 1)
            ) for _ in range(self.num_layers)
        ]

        if physics_loss is not None:
            self.physics_loss_tracker = keras.metrics.Mean(name='physics_loss')

    def call(self, inputs, training=None, mask=None):
        if self.num_params == 0:
            function = inputs
        else:
            parameters, function = inputs

        if not self.periodic:
            x = tf.pad(function, [[0, 0], [0, 2], [0, 0]], "CONSTANT")  # pad for non-periodic BCs
        else:
            x = function

        if self.num_params > 0:  # add parameters
            x = tf.concat([x, tf.repeat(tf.expand_dims(parameters, axis=1), tf.shape(x)[1], axis=1)], axis=-1)

        if self.size is not None:  # adding coordinates
            coordinates = tf.meshgrid(*[tf.range(0, dim, dim/tf.shape(x)[1]) for dim in self.size])
            coordinates = [
                tf.repeat(
                    tf.expand_dims(
                        tf.expand_dims(
                            tf.cast(y, dtype=tf.float32), axis=-1
                        ), axis=0
                    ), tf.shape(x)[0], axis=0
                ) for y in coordinates
            ]

            x = tf.concat([x] + coordinates, axis=-1)

        x = self.function_embedding(x)  # project to higher dimension

        # applying fourier layers
        for layer in self.fourier_layers:
            if self.mlp_hidden_units > 0 and self.num_params > 0:
                x = layer([parameters, x])
            else:
                x = layer(x)

        # projecting back to original dimension
        x = self.output_projection(x)

        if not self.periodic:
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
            return super().train_step(data)
