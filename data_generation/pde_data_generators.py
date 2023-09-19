import random

import numpy as np

from utils import BoundaryCondition
from data_generation import method_of_lines, generate_random_functions_with_bc
from data_generation.tfrecord_utils import tf_serialize_example


RESOLUTION = 2048
NUM_SAME_BC = 32


# solves the burgers equation
def burgers_equation(bc, total=10000):
    count = 0
    while (count := count + 1) < total:
        # defining the PDE
        if(count%10==0):
            print(count)
        nu = 0.1
        f = lambda t, u, u_x, u_xx: nu * u_xx - u * u_x

        # defining the domain to solve on
        L = 1

        dt = 0.001
        T = np.linspace(0, 1, int(1 / dt))

        # defining the boundary conditions
        left_boundary = random.uniform(-1, 1)
        right_boundary = random.uniform(-1, 1)

        if bc != BoundaryCondition.PERIODIC:  # if its dirichlet or neumann
            bc = (
                (bc, left_boundary),
                (bc, right_boundary)
            )

        u0 = generate_random_functions_with_bc(NUM_SAME_BC, RESOLUTION, bc)

        for ic in u0:
            output = method_of_lines(
                f, ic, L, T, bc=bc, max_order=2
            )

            yield tf_serialize_example(ic, output[1:])


# solves the fisher-kpp equation
def fisher_kpp_equation(bc, total=10000):
    count = 0
    while (count := count + 1) < total:
        r = 5

        # defining the PDE
        f = lambda t, u, u_x, u_xx: r * u * (1 - u) + u_xx

        # defining the domain to solve on
        L = 1

        dt = 0.001
        T = np.linspace(0, 1, int(1 / dt))

        # defining the boundary conditions
        left_boundary = random.uniform(-1, 1)
        right_boundary = random.uniform(-1, 1)

        if bc != BoundaryCondition.PERIODIC:  # if its dirichlet or neumann
            bc = (
                (bc, left_boundary),
                (bc, right_boundary)
            )

        u0 = generate_random_functions_with_bc(NUM_SAME_BC, RESOLUTION, bc)

        for ic in u0:
            output = method_of_lines(
                f, ic, L, T, bc=bc, max_order=2
            )

            yield tf_serialize_example(ic, output[1:])


# solves the ZPK equation
def zpk_equation(bc, total=10000):
    count = 0
    while (count := count + 1) < total:
        beta = 5

        # defining the PDE
        f = lambda t, u, u_x, u_xx: beta ** 2 / 2 * u * (1 - u) * np.exp(-beta * (1 - u)) + u_xx

        # defining the domain to solve on
        L = 1

        dt = 0.001
        T = np.linspace(0, 1, int(1 / dt))

        # defining the boundary conditions
        left_boundary = random.uniform(-1, 1)
        right_boundary = random.uniform(-1, 1)

        if bc != BoundaryCondition.PERIODIC:  # if its dirichlet or neumann
            bc = (
                (bc, left_boundary),
                (bc, right_boundary)
            )

        u0 = generate_random_functions_with_bc(NUM_SAME_BC, RESOLUTION, bc)

        for ic in u0:
            output = method_of_lines(
                f, ic, L, T, bc=bc, max_order=2
            )

            yield tf_serialize_example(ic, output[1:])
