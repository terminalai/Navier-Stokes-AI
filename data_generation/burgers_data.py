import random

import numpy as np

from data_generation import generate_random_functions, method_of_lines
from utils import BoundaryCondition


def data_generator(count):
    mu = 0.01

    L_x = 1
    dx = 1 / 2048
    N_x = int(L_x / dx)
    X = np.linspace(0, L_x, N_x)

    L_t = 1
    dt = 0.001
    N_t = int(L_t / dt)
    T = np.linspace(0, L_t, N_t)

    left_boundary = random.uniform(-1, 1)
    right_boundary = random.uniform(-1, 1)

    u0 = generate_random_functions(
        num=32,
        resolution=2048,
        points=np.array(((0, left_boundary), (1, right_boundary)))
    )

    # defining the PDE
    f = lambda u, u_x, u_xx: mu * u_xx - u * u_x

    for ic in u0:
        output = method_of_lines(
            f, ic, L_x, T,
            bc=(
                (BoundaryCondition.DIRICHLET, left_boundary),
                (BoundaryCondition.DIRICHLET, right_boundary)
            )
        )

        np.save(f"../data/ics/ic_{count}.npy", ic)
        np.save(f"../data/sol/sol_{count}.npy", output[1:])
        # yield serialize_example(ic, output[1:])

        count += 1

    return 0
