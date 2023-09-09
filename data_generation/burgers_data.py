import random

import numpy as np

from data_generation import method_of_lines, generate_random_functions_with_bc
from utils import BoundaryCondition


def burgers_data_generator(count):
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
        (BoundaryCondition.DIRICHLET, left_boundary),
        (BoundaryCondition.DIRICHLET, right_boundary)
    )

    u0 = generate_random_functions_with_bc(32, 2048, bc)

    for ic in u0:
        output = method_of_lines(
            f, ic, L, T, bc=bc, max_order=2
        )

        np.save(f"../data/ics/ic_{count}.npy", ic)
        np.save(f"../data/sol/sol_{count}.npy", output[1:])

        count += 1

    print(f"Completed {count} - {count+32}")
    return 0


