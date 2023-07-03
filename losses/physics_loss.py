import tensorflow as tf

keras = tf.keras


def physics_loss(f, max_order, sizes, periodicity, num_params=0):
    """
    Implements the physics-based loss from (Shi et al. 2022, https://arxiv.org/pdf/2206.09418.pdf)
    TODO do something more like (Li et al. 2022, https://arxiv.org/pdf/2111.03794.pdf)
    :param f: A function that computes the PDE and sets it to 0
    :param max_order: The maximum orders of derivatives to compute
    :param sizes: The sizes of the spatial dimensions
    :param periodicity: Is the function periodic along this dimension?
    :param num_params: The number of real parameters the model is looking at
    :return: Returns the loss function
    """

    def loss(inputs, y):
        if len(max_order) == 1:  # 1d
            lst = [None for _ in range(max_order[0] + 1)]
            lst[0] = y
            for order in range(1, max_order + 1):
                y = lst[order - 1]
                lst[order] = (tf.concat([y[:, 1:, :], tf.expand_dims(y[:, 0, :], axis=1)], axis=1) - y) * sizes[0]
        elif len(max_order) == 2:  # 2d
            lst = [[None for _ in range(max_order[1] + 1)] for _ in range(max_order[0] + 1)]

            lst[0][0] = y
            for i in range(1, max_order[0] + 1):
                y = lst[i - 1][0]
                lst[i][0] = (tf.concat([y[:, 1:], tf.expand_dims(y[:, 0], axis=1)], axis=1) - y) * sizes[0]

            for i in range(max_order[0] + 1):
                for j in range(1, max_order[1] + 1):
                    y = lst[i][j - 1]
                    lst[0][j] = (tf.concat([y[:, :, 1:], tf.expand_dims(y[:, :, 0], axis=2)], axis=2) - y) * sizes[1]
        else:
            raise NotImplementedError("3D or higher physics loss is not yet implemented")

        if num_params > 0:
            return tf.math.reduce_mean(tf.square(f(inputs, lst)))
        else:
            return tf.math.reduce_mean(tf.square(f(lst)))

    return loss
