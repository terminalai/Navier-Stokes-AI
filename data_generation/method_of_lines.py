import numpy as np

from scipy.integrate import odeint

from data_generation.gaussian_process import generate_random_functions, generate_random_functions_with_bc
from utils.boundary_conditions import process_boundary_condition_1d, BoundaryCondition


def method_of_lines(eqn, u0, L, T, bc, k=30, max_order=2):
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
            u_hat_xxx = -1j * k ** 3 * u_hat

            # switching in the spatial domain
            u_x = np.fft.ifft(u_hat_x)
            u_xx = np.fft.ifft(u_hat_xx)
            u_xxx = np.fft.ifft(u_hat_xxx)
        else:
            # boundary conditions
            u[0] = process_boundary_condition_1d(bc[0], u[1], dx, side=-1)
            u[-1] = process_boundary_condition_1d(bc[1], u[-2], dx, side=1)

            u_x = np.concatenate([
                (u[1:2] - u[0:1]) / dx,
                (u[2:] - u[:-2]) / dx / 2,
                (u[-2:-1] - u[-3:-2]) / dx
            ])

            if max_order > 1:
                u_xx = np.concatenate([
                    (u[0:1] - 2 * u[1:2] + u[2:3]) / (dx ** 2),
                    (u[2:] - 2 * u[1:-1] + u[:-2]) / (dx ** 2),
                    (u[-2:-1] - 2 * u[-3:-2] + u[-4:-3]) / (dx ** 2)
                ])

            if max_order > 2:
                u_xxx = np.concatenate([
                    (-2.5*u[0:1] + 9 * u[1:2] - 12 * u[2:3] + 7 * u[3:4] - 1.5 * u[4:5]) / (dx ** 3),
                    (-2.5*u[1:2] + 9 * u[2:3] - 12 * u[3:4] + 7 * u[4:5] - 1.5 * u[5:6]) / (dx ** 3),
                    (-0.5 * u[4:] + u[3:-1] - u[1:-3] + 0.5 * u[:-4]) / (dx ** 3),
                    (2.5*u[-2:-1] - 9 * u[-3:-2] + 12 * u[-4:-3] - 7 * u[-5:-4] + 1.5 * u[-6:-5]) / (dx ** 3),
                    (2.5*u[-3:-2] - 9 * u[-4:-3] + 12 * u[-5:-4] - 7 * u[-6:-5] + 1.5 * u[-7:-6]) / (dx ** 3)
                ])

            if max_order > 3:
                u_xxxx = np.concatenate([
                    (u[0:1] - 4 * u[1:2] + 6 * u[2:3] - 4 * u[3:4] + u[4:5]) / (dx ** 4),
                    (u[1:2] - 4 * u[2:3] + 6 * u[3:4] - 4 * u[4:5] + u[5:6]) / (dx ** 4),
                    (u[4:] - 4 * u[3:-1] + 6 * u[2:-2] + - 4 * u[1:-3] + u[:-4]) / (dx ** 4),
                    (u[-2:-1] - 4 * u[-3:-2] + 6 * u[-4:-3] - 4 * u[-5:-4] + u[-6:-5]) / (dx ** 4),
                    (u[-3:-2] - 4 * u[-4:-3] + 6 * u[-5:-4] - 4 * u[-6:-5] + u[-7:-6]) / (dx ** 4)
                ])

        if max_order == 0:
            u_t = eqn(t, u)
        elif max_order == 1:
            u_t = eqn(t, u, u_x)
        elif max_order == 2:
            u_t = eqn(t, u, u_x, u_xx)
        elif max_order == 3:
            u_t = eqn(t, u, u_x, u_xx, u_xxx)
        elif max_order == 4:
            u_t = eqn(t, u, u_x, u_xxx, u_xxxx)
        else:
            raise Exception("Only PDEs up to the 4th order are supposed!")

        u_t[0] = 0
        u_t[-1] = 0
        return u_t.real

    # solve it like a set of ODEs
    return odeint(f, u0, T)


if __name__ == "__main__":
    r = 5
    D = 1

    L_x = 1
    dx = 1/1000
    N_x = int(L_x / dx)
    X = np.linspace(0, L_x, N_x)

    L_t = 1
    dt = 0.001
    N_t = int(L_t / dt)
    T = np.linspace(0, L_t, N_t)

    k = 2 * np.pi * np.fft.fftfreq(N_x, d=dx)

    u0 = generate_random_functions_with_bc(
        num=100, resolution=1000,
        bc=((BoundaryCondition.NEUMANN, -1), (BoundaryCondition.NEUMANN, 2))
    ) * 3

    # defining the PDE
    f = lambda t, u, u_x, u_xx: r * u * (1 - u) + D * u_xx

    import time

    start_time = time.time()
    output = method_of_lines(
        f, u0[0], L_x, T,
        bc=((BoundaryCondition.NEUMANN, -1), (BoundaryCondition.NEUMANN, 2)),
        max_order=2
    )
    print(time.time() - start_time)
