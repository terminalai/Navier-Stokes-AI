import random
import scipy

import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, List, Tuple


def from_coefficients(coefficients: np.array, kmax: int = 30, dx: int = 0) -> Callable[[np.array], np.array]:
    def apply(x):
        output = np.zeros(x.shape)
        for i in range(kmax):
            output += coefficients[i] * (i + 1) ** dx * np.sin((i + 1) + np.pi / 2 + (i + 1) * x)

        for i in range(kmax, 2 * kmax + 1):
            output += coefficients[i] * i ** dx * np.cos(i * np.pi / 2 + i * x)

        return output

    return apply


def get_coefficients(f: Callable[[np.array], np.array], init: np.array, kmax: int = 30) -> np.array:
    def objective(coefficients):
        f_guess = from_coefficients(coefficients, kmax=kmax)
        return np.mean(np.square(f_guess(collocation_points) - f(collocation_points)))

    collocation_points = np.arange(0, 2 * np.pi, 1 / (2 * kmax + 1))
    return scipy.optimize.minimize(objective, init).x


def burgers_equation(coefficients: np.array, viscosity: float = 1e-5, kmax: int = 30):
    u = from_coefficients(coefficients, kmax=kmax)
    ux = from_coefficients(coefficients, kmax=kmax, dx=1)  # first-order derivative
    uxx = from_coefficients(coefficients, kmax=kmax, dx=2)  # second-order derivative

    return lambda x: viscosity * uxx(x) - u(x) * ux(x)


def solve_burgers(init_coefficient: np.array, viscosity: float = 1e-5, time: float = 1.0,
                  kmax: int = 30, dt: float = 0.01) -> List[Tuple[float, Callable[[np.array], np.array]]]:
    coefficients = init_coefficient
    # get_coefficients(ic, np.zeros((2 * kmax + 1,)), kmax=kmax)

    t = 0
    solutions = []
    for i in range(int(time / dt)):
        u = from_coefficients(coefficients, kmax=kmax)
        solutions.append((t, u))

        ut = burgers_equation(coefficients, viscosity=viscosity)

        unew = lambda x: u(x) + ut(x) * dt
        coefficients = get_coefficients(unew, coefficients, kmax=kmax)

        t += dt
        print(t)

    u = from_coefficients(coefficients, kmax=kmax)
    solutions.append((t, u))
    return solutions


if __name__ == "__main__":
    kmax = 30

    coefficients = np.array([(2 * random.uniform(0, 1) - 1) / (3 * (x % (kmax + 1)) + 1)
                             for x in range(2 * kmax + 1)])
    ic = from_coefficients(coefficients, kmax=kmax)

    solutions = solve_burgers(
        coefficients, viscosity=1e-2, dt=0.001, time=0.1, kmax=kmax
    )

    for t, sol in solutions:
        plt.plot(sol(np.arange(0, 2 * np.pi, 0.01)))

    plt.show()
