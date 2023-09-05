import numpy as np

from scipy.integrate import odeint

from data_generation.gaussian_process import generate_random_functions
from utils.boundary_conditions import process_boundary_condition_1d, BoundaryCondition


def method_of_lines(eqn, u0, L, T, bc):
    """
    A generic solver for PDEs that can be expressed in the form u_t = f(x, u, u_x, u_xx, ...) using the Method of Lines.
    :param eqn: The function f(u, u_x, u_xx, ...) that describes the PDE.
    :param u0: The initial condition for the PDE.
    :param L: The size of the domain in which the PDE is being solved.
    :param T: The timesteps for which the solution should be provided.
    :param bc: The boundary conditions of the PDE.
    :return: Returns the solution to the PDE at the timesteps provided in T.
    """

    dx = L / len(u0)

    def f(u, t):
        if bc == BoundaryCondition.PERIODIC:
            # spatial derivative in the Fourier domain
            u_hat = np.fft.fft(u)
            u_hat_x = 1j * k * u_hat
            u_hat_xx = -k ** 2 * u_hat

            # switching in the spatial domain
            u_x = np.fft.ifft(u_hat_x)
            u_xx = np.fft.ifft(u_hat_xx)
        else:
            # boundary conditions
            u[0] = process_boundary_condition_1d(bc[0], u[1], dx, side=-1)
            u[-1] = process_boundary_condition_1d(bc[1], u[-2], dx, side=1)

            u_x = np.concatenate([
                (u[1:2] - u[0:1]) / dx,
                (u[2:] - u[:-2]) / dx / 2,
                (u[-2:-1] - u[-3:-2]) / dx
            ])
            u_xx = np.concatenate([
                (u[0:1] - 2 * u[1:2] + u[2:3]) / (dx ** 2),
                (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx ** 2),
                (u[-2:-1] - 2 * u[-3:-2] + u[-4:-3]) / (dx ** 2)
            ])

        u_t = eqn(u, u_x, u_xx)
        u_t[0] = 0
        u_t[-1] = 0
        return u_t.real

    # solve it like a set of ODEs
    return odeint(f, u0, T, mxstep=5000)


if __name__ == "__main__":
    import random

    mu = 0.01

    L_x = 1
    dx = 0.001
    N_x = int(L_x / dx)
    X = np.linspace(0, L_x, N_x)

    L_t = 1
    dt = 0.001
    N_t = int(L_t / dt)
    T = np.linspace(0, L_t, N_t)

    k = 2 * np.pi * np.fft.fftfreq(N_x, d=dx)

    u0 = generate_random_functions(num=100, resolution=8192)

    # defining the PDE
    f = lambda u, u_x, u_xx: mu * u_xx - u * u_x

    import time

    start_time = time.time()
    output = method_of_lines(
        f, u0[0], L_x, T,
        bc=((BoundaryCondition.DIRICHLET, 0), (BoundaryCondition.DIRICHLET, 0))
    )
    print(time.time() - start_time)
