from enum import Enum
import tensorflow as tf


# types of boundary conditions
class BoundaryCondition(Enum):
    DIRICHLET = 1  # (DIRICHLET, u)
    NEUMANN = 2  # (NEUMANN, u_x)
    PERIODIC = 3  # (PERIODIC)
    ROBIN = 4  # (ROBIN, k, c) -> u + k * u_x = c
    # CAUCHY = 5


def process_boundary_condition_1d(bc, values, dx, side=-1):  # spaces are so that pycharm renders the docstirng properly
    """
    Computes the boundary conditions and applies them to the input. The input format for boundary conditions is:

    - DIRICHLET, u

    - NEUMANN, u_x

    - PERIODIC

    - ROBIN, k, c -> u + k * u_x = c

    - todo implement CAUCHY

    :param bc: The boundary condition to be applied. The format is as shown above.
    :param values: The value(s) of the function right next to the boundary.
    :param dx: The discretion size of this domain.
    :param side: The side of the domain to apply the boundary condition to. (1 for the right boundary, -1 for the left boundary)
    :return:
    """

    # ensure side is either 1 or -1
    assert side == 1 or side == -1, "parameter side should only be either 1 or -1"

    if bc[0] == BoundaryCondition.DIRICHLET:
        return bc[1]
    elif bc[0] == BoundaryCondition.NEUMANN:
        return side * bc[1] * dx + values
    elif bc[0] == BoundaryCondition.ROBIN:
        return (bc[2] + side * values / dx) / (1 + side * bc[1] / dx)


def compute_boundary(inputs, bc, size, shape, bc_value, side=-1):
    dx = size / shape
    if bc[0] == BoundaryCondition.DIRICHLET:
        return bc[1](inputs)
    elif bc[0] == BoundaryCondition.NEUMANN:
        return bc[1](inputs) * dx - bc_value * side
    elif bc[0] == BoundaryCondition.ROBIN:  # todo do robin correctly
        return (dx * bc[2](inputs) + side * bc[1](inputs) * bc_value) / (1 + bc[1](inputs) * side)


def process_bc(inputs, sizes, bc, y):
    shape = y.shape[1:]

    # todo handle BCs for coupled PDEs
    if len(sizes) == 1:  # 1d
        if bc[0] != BoundaryCondition.PERIODIC:
            temp = tf.ones((tf.shape(y)[0], 1, tf.shape(y)[2]))

            # computing for left BC
            if bc[0][0] is not None:
                left_bc = compute_boundary(inputs, bc[0][0], sizes[0], shape[0], y[:, 1], side=-1)
                y = tf.concat([left_bc * temp, y[:, 1:]], axis=1)

            # computing for right BC
            if bc[0][1] is not None:
                right_bc = compute_boundary(inputs, bc[0][1], sizes[0], shape[0], y[:, -1], side=1)
                y = tf.concat([y[:, :-1], right_bc * temp], axis=1)
    elif len(sizes) == 2:  # 2d
        if bc[0] != BoundaryCondition.PERIODIC:
            temp = tf.ones((tf.shape(y)[0], 1, tf.shape(y)[2], tf.shape(y)[3]))

            # computing for left BC
            if bc[0][0] is not None:
                left_bc = compute_boundary(inputs, bc[0][0], sizes[0], shape[0], y[:, 1], side=-1)
                y = tf.concat([left_bc * temp, y[:, 1:]], axis=1)

            # computing for right BC
            if bc[0][1] is not None:
                right_bc = compute_boundary(inputs, bc[0][1], sizes[0], shape[0], y[:, -1], side=1)
                y = tf.concat([y[:, :-1], right_bc * temp], axis=1)

        if bc[1] != BoundaryCondition.PERIODIC:
            temp = tf.ones((tf.shape(y)[0], tf.shape(y)[1], 1, tf.shape(y)[3]))

            # computing for left BC
            if bc[1][0] is not None:
                left_bc = compute_boundary(inputs, bc[1][0], sizes[1], shape[1], y[:, :, 1], side=-1)
                y = tf.concat([left_bc * temp, y[:, :, 1:]], axis=2)

            # computing for right BC
            if bc[1][1] is not None:
                right_bc = compute_boundary(inputs, bc[1][1], sizes[1], shape[1], y[:, :, -1], side=1)
                y = tf.concat([y[:, :, :-1], right_bc * temp], axis=2)

    return y
