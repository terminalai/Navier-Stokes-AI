import tensorflow as tf
from utils import BoundaryCondition, process_bc

keras = tf.keras


def physics_loss(f, max_order, sizes, bc, num_params=0):
    """
    Implements the physics-based loss from (Shi et al. 2022, https://arxiv.org/pdf/2206.09418.pdf)
    TODO do something more like (Li et al. 2022, https://arxiv.org/pdf/2111.03794.pdf)
    :param f: A function that computes the PDE and sets it to 0
    :param max_order: The maximum orders of derivatives to compute
    :param sizes: The sizes of the spatial dimensions
    :param bc: The boundary condition along the ith dimension.
    :param num_params: The number of real parameters the model is looking at
    :return: Returns the loss function
    """

    def loss(inputs, y):
        temp = y
        shape = y.shape[1:]

        if len(max_order) == 1:  # 1d
            lst = [None for _ in range(max_order[0] + 1)]

            # boundary condition handling
            y = process_bc(inputs, sizes, bc, y)

            # actual differentiation
            lst[0] = y
            for order in range(1, max_order[0] + 1):
                y = lst[order - 1]
                if bc[0] == BoundaryCondition.PERIODIC:
                    lst[order] = (tf.concat([y[:, 1:], tf.expand_dims(y[:, 0], axis=1)], axis=1) - y) * shape[0]
                else:
                    lst[order] = tf.concat([
                        (y[:, 1:2] - y[:, 0:1]) * shape[0],
                        (y[:, 2:] - y[:, :-2]) * shape[0] / 2,
                        (y[:, -2:-1] - y[:, -3:-2]) * shape[0]
                    ])
        elif len(max_order) == 2:  # 2d
            lst = [[None for _ in range(max_order[1] + 1)] for _ in range(max_order[0] + 1)]

            # boundary condition handling
            y = process_bc(inputs, sizes, bc, y)

            lst[0][0] = y  # compute 1st row of derivatives first
            for i in range(1, max_order[0] + 1):
                y = lst[i - 1][0]
                if bc[0] == BoundaryCondition.PERIODIC:
                    lst[i][0] = (tf.concat([y[:, 1:], tf.expand_dims(y[:, 0], axis=1)], axis=1) - y) * shape[0]
                else:
                    lst[i][0] = tf.concat([
                        (y[:, 1:2] - y[:, 0:1]) * shape[0],
                        (y[:, 2:] - y[:, :-2]) * shape[0] / 2,
                        (y[:, -2:-1] - y[:, -3:-2]) * shape[0]
                    ], axis=1)

                # lst[i][0] = (tf.concat([y[:, 1:], tf.expand_dims(y[:, 0], axis=1)], axis=1) - y) * shape[0]

            # propagate down
            for i in range(max_order[0] + 1):
                for j in range(1, max_order[1] + 1):
                    y = lst[i][j - 1]
                    if bc[0] == BoundaryCondition.PERIODIC:
                        lst[i][j] = (tf.concat([y[:, :, 1:], tf.expand_dims(y[:, :, 0], axis=2)], axis=2) - y) * shape[1]
                    else:
                        lst[i][j] = tf.concat([
                            (y[:, :, 1:2] - y[:, :, 0:1]) * shape[1],
                            (y[:, :, 2:] - y[:, :, :-2]) * shape[1] / 2,
                            (y[:, :, -2:-1] - y[:, :, -3:-2]) * shape[1]
                        ], axis=2)
        else:
            raise NotImplementedError("3D or higher physics loss is not yet implemented")

        # we assume this function will handle the uneven indexing :)
        return tf.math.reduce_mean(tf.square(f(inputs, temp, lst)))

    return loss
