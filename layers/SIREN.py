import keras_core.ops as ops

from keras_core.models import *
from keras_core.layers import *
from keras_core.initializers import *


class SIREN(Model):
    def __init__(
        self,
        widths=(64,),
        omega_0=30.0,
        output_dim=1,
        use_latent=False,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.widths = widths
        self.omega_0 = omega_0
        self.output_dim = output_dim

        self.use_latent = use_latent

        # hidden layers
        first_layer = Dense(
            widths[0],
            kernel_initializer=RandomUniform(-omega_0*ops.sqrt(6.0/widths[0]), omega_0*ops.sqrt(6.0/widths[0]))
        )

        self.mlp = [first_layer] + [
            Dense(
                width,
                kernel_initializer=RandomUniform(-ops.sqrt(6.0/width), ops.sqrt(6.0/width))
            ) for width in widths[1:]
        ]

        # output layer
        self.output_projection = Dense(output_dim)

        # modulation
        if self.use_latent:
            self.latent_mlp = [
                Dense(width) for width in widths
            ]

    def call(self, inputs, *args, **kwargs):
        if self.use_latent: x, latent = inputs
        else: x = inputs

        num_pts = ops.shape(x)[1]

        for i in range(len(self.mlp)):
            # apply dense layer
            x = self.mlp[i](x)

            if self.use_latent:
                # apply modulation from latent
                x = x + self.omega_0 * ops.repeat(
                    ops.expand_dims(
                        self.latent_mlp[i](latent), axis=1
                    ), num_pts, axis=1
                )

            # apply activation function
            x = ops.sin(x)

        return self.output_projection(x)


if __name__ == "__main__":
    import keras_core
    import tensorflow as tf

    image = tf.io.decode_image(
        tf.io.read_file(
            r"C:\Users\jedli\OneDrive - NUS High School\Documents\Computing Studies\computed_tomography\logs\
            regularly_masked_sinograms\images\ground_truth\1.png"
        )
    )
    image = tf.cast(image, dtype=tf.float32)
    image = (image - tf.math.reduce_mean(image)) / tf.math.reduce_std(image)

    width = tf.cast(image.shape[0], dtype=tf.float32)
    height = tf.cast(image.shape[1], dtype=tf.float32)

    X, Y = tf.meshgrid(
        (tf.cast(tf.range(image.shape[0]), dtype=tf.float32) / width * 2 - 1),
        (tf.cast(tf.range(image.shape[1]), dtype=tf.float32) / height * 2 - 1)
    )
    X, Y = tf.reshape(X, (-1, 1)), tf.reshape(Y, (-1, 1))

    x = tf.cast(tf.concat([X, Y], axis=-1), dtype=tf.float32)
    y = tf.cast(tf.reshape(image, (-1, 1)), dtype=tf.float32)

    temp = tf.concat([x, y], axis=-1)
    temp = tf.random.shuffle(temp)

    x_train, y_train = temp[..., :2], temp[..., 2:]

    latent = tf.repeat(tf.random.normal((1, 10,)), len(x_train), axis=0)

    y_true = tf.reshape(y, image.shape)

    model = SIREN(
        widths=[256, 256, 256, 256, 256],
        output_dim=1
    )
    model.compile(optimizer=keras_core.optimizers.AdamW(learning_rate=1e-3), loss="mse")

    model.fit(x_train, y_train, epochs=20, batch_size=64)

    y_pred = model.predict(x)
    y_pred = tf.reshape(y_pred, image.shape)

    x = tf.Variable(x)
    with tf.GradientTape() as tape:
        y = model(x)

    grad = tape.gradient(y, x)

    grad_1 = tf.reshape(grad[:, 0], image.shape)
    grad_2 = tf.reshape(grad[:, 1], image.shape)
