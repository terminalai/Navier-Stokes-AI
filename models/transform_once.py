import keras_core.ops as ops
import tensorflow as tf
from keras_core.initializers import RandomNormal
from keras_core.layers import *
from keras_core.models import *


class MLPMixer(Layer):
    def __init__(
            self,
            activation=ops.sin,
            signal_length=1024,
            name=None,
            **kwargs
    ):  # not actually an MLPMixer
        super().__init__(name=name, **kwargs)

        self.activation = Activation(activation)
        self.signal_length = signal_length

        self.channel_mixing = None
        self.spatial_mixing = None

    def build(self, input_shape):
        self.channel_mixing = self.add_weight(
            shape=(input_shape[-2], input_shape[-2], input_shape[-1]),
            initializer=RandomNormal(0, ops.sqrt(self.signal_length / (input_shape[-1] * input_shape[-2])))
        )

        self.spatial_mixing = self.add_weight(
            shape=(input_shape[-1], input_shape[-1], input_shape[-2]),
            initializer=RandomNormal(0, ops.sqrt(1 / input_shape[-1]))
        )

        """
        self.channel_mixing = Dense(
            input_shape[-1],
            activation=self.activation,
            kernel_initializer=RandomNormal(0, ops.sqrt(self.signal_length / (input_shape[-1] * input_shape[-2]))),
            use_bias=False
        )

        self.mode_mixing = Dense(
            input_shape[-2],
            kernel_initializer=RandomNormal(0, ops.sqrt(1 / input_shape[-1])),
            use_bias=False
        )
        """

    def call(self, inputs, **kwargs):
        x = ops.einsum("bix,iox->box", inputs, self.channel_mixing)
        x = self.activation(x)
        x = ops.einsum("bix,xyi->biy", x, self.spatial_mixing)

        return x + inputs


# todo make the model work for the non 2D case
class TransformOnce(Model):
    def __init__(
            self,
            model,
            num_params,
            input_shape,
            num_outputs=-1,
            k_max=64,
            dim=64,
            num_layers=4,
            activation=lambda x: x * ops.cos(x),
            size=None,
            periodic=(False,),
            *args, **kwargs
    ):
        super(TransformOnce, self).__init__(*args, **kwargs)

        self.model = model

        self.num_params = num_params
        self.k_max = k_max
        self.dim = dim
        self.num_layers = num_layers
        self.num_outputs = input_shape[-1] if num_outputs < 0 else num_outputs
        self.activation = activation
        self.size = size
        self.periodic = periodic

        # checking some input conditions
        assert size is None or len(input_shape) - 1 == len(size)

        self.function_embedding = Dense(self.dim, name="function_embedding")

        self.output_embedding = Sequential(
            [
                Dense(
                    512,
                    activation=activation,
                    kernel_initializer=RandomNormal(0, 1e-8),
                    use_bias=False
                ),
                Dense(
                    self.num_outputs,
                    kernel_initializer=RandomNormal(0, 1e-8),
                    use_bias=False
                )
            ], name="output_embedding"
        )

    def call(self, inputs, training=False, mask=None):
        if self.num_params == 0:
            function = inputs
        else:
            parameters, function = inputs

        padding = [[0, 0]] + [[0, 10 * (not x)] for x in self.periodic] + [[0, 0]]
        x = ops.pad(function, padding, "constant")  # pad for non-periodic BCs

        if self.num_params > 0:  # add parameters
            x = ops.concatenate([x, ops.repeat(ops.expand_dims(parameters, axis=1), ops.shape(x)[1], axis=1)], axis=-1)

        if self.size is not None:  # adding coordinates
            coordinates = ops.meshgrid(
                *[ops.arange(0, self.size[i], self.size[i] / ops.shape(x)[-i - 2]) for i in range(len(self.size))]
            )
            coordinates = [
                ops.repeat(
                    ops.expand_dims(
                        ops.expand_dims(
                            ops.cast(y, dtype="float32"), axis=-1
                        ), axis=0
                    ), ops.shape(x)[0], axis=0
                ) for y in coordinates
            ]

            x = ops.concatenate([x] + coordinates, axis=-1)

        x = self.function_embedding(x)  # project to higher dimension

        # convert to k-space via DCT
        x = ops.transpose(x, (0, 2, 1))
        x = tf.signal.dct(x, n=self.k_max, norm="ortho")

        # apply stuff on the modes
        x = self.model(x)

        # process modes a bit more
        x = ops.transpose(x, (0, 2, 1))
        x = self.output_embedding(x)
        x = ops.transpose(x, (0, 2, 1))

        if not training or True:
            # convert back into n-space
            x = tf.signal.idct(x, n=function.shape[1], norm="ortho")
            x = ops.transpose(x, (0, 2, 1))

            # removing padding
            for i in range(len(self.periodic)):
                if i == 0 and not self.periodic[i]:
                    x = x[:, :-10]  # remove padding

                if i == 1 and not self.periodic[i]:
                    x = x[:, :, :-10]  # remove padding

        return x

    def train_step(self, data):
        x, y = data

        # convert y into DCT modes
        y = ops.transpose(y, (0, 2, 1))
        modes = tf.signal.dct(y, n=self.k_max, norm="ortho")

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # forward pass

            # compute the loss value
            loss = self.compute_loss(y=modes, y_pred=y_pred)

        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # update weights
        self.optimizer.apply(gradients, trainable_vars)

        # update the metrics.
        # metrics are configured in `compile()`.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(modes, y_pred)

        # return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


if __name__ == "__main__":
    import keras_core

    import numpy as np
    from scipy.io import loadmat

    model = TransformOnce(
        model=Sequential(
            [
                Flatten(),
                Dense(1024, activation="gelu"),
                Dense(64 * 64),
                Reshape((64, 64,))
                # MLPMixer() for _ in range(4)
            ]
        ),
        num_params=0,
        input_shape=(1024, 1),
        k_max=64,
        periodic=[True]
    )
    model.call(tf.random.normal(shape=(1, 1024, 1)))
    model.summary()

    model.compile(
        optimizer=keras_core.optimizers.AdamW(learning_rate=1e-3), loss="mse"
    )

    data = loadmat("../burgers_data_R10.mat")

    x = data['a'].astype("float32")  # input
    y = data['u'].astype("float32")  # target

    x = np.expand_dims(x, axis=-1)[:, ::8192//1024]
    y = np.expand_dims(y, axis=-1)[:, ::8192//1024]

    x_train, y_train = x[:-len(x)//8], y[:-len(y)//8]
    x_test, y_test = x[-len(x)//8:], y[-len(y)//8:]

    model.fit(x_train, y_train, epochs=100, batch_size=16, validation_data=(x_test, y_test))
