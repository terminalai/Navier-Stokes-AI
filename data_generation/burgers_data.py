import random

import numpy as np

from utils import BoundaryCondition
from data_generation import method_of_lines, generate_random_functions_with_bc
from data_generation.tfrecord_utils import tf_serialize_example


def burgers_data_generator():
    while True:
        nu = 0.1

        L = 1

        dt = 0.001
        T = np.linspace(0, 1, int(1 / dt))

        # defining the PDE
        f = lambda t, u, u_x, u_xx: nu * u_xx - u * u_x

        # defining the boundary conditions
        left_boundary = random.uniform(-1, 1)
        right_boundary = random.uniform(-1, 1)
        bc = (
            (BoundaryCondition.NEUMANN, left_boundary),
            (BoundaryCondition.NEUMANN, right_boundary)
        )

        u0 = generate_random_functions_with_bc(32, 2048, bc)

        for ic in u0:
            output = method_of_lines(
                f, ic, L, T, bc=bc, max_order=2
            )

            yield tf_serialize_example(ic, output[1:])

            # np.save(f"../data/ics/ic_{count}.npy", ic)
            # np.save(f"../data/sol/sol_{count}.npy", output[1:])
