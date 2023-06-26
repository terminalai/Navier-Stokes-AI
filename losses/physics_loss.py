import tensorflow as tf
keras = tf.keras


def physics_loss(f, max_order, dx, dt, num_params=0):
    """
    Implements the mean squared residual loss (Shi et al. 2022, https://arxiv.org/pdf/2206.09418.pdf)
    :param f: A function that computes the PDE and sets it to 0
    :param max_order: The maximum order of derivative to compute
    :param dx: The size of the spatial discretion
    :param dt: The size of the timestep
    :param num_params: The number of real parameters the model is looking at
    :return: Returns the loss function
    """
    def loss(x, y):
        if num_params > 0:
            params, x = x

        lst = [(y - x) / dt, y]
        for order in range(1, max_order + 1):
            y = tf.concat([y[:, 1:, :], tf.expand_dims(y[:, 0, :], axis=1)], axis=1) - y
            lst.append(y / dx)

        if num_params > 0:
            return tf.math.reduce_mean(tf.square(f(params, lst)))
        else:
            return tf.math.reduce_mean(tf.square(f(lst)))

    return loss
