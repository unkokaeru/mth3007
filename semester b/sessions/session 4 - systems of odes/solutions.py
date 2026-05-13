"""Solutions for Session 4 - Systems of ODEs."""

from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"


def explicit_euler_system(
    derivative_function: Callable[
        [float, npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ],
    initial_values: npt.NDArray[np.float64],
    time_start: float,
    time_end: float,
    time_step: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve a system of ODEs using the explicit Euler method.

    The explicit Euler method for systems uses the vector approximation:
    y_{n+1} = y_n + h * g(t_n, y_n)

    Args:
        derivative_function: Function g(t, y) that computes dy/dt as a vector.
        initial_values: Initial condition vector y(t_0).
        time_start: Starting time t_0.
        time_end: Ending time t_max.
        time_step: Time step size h (Delta t).

    Returns:
        Tuple of (time_values, solution_values) where solution_values has
        shape (number_of_steps + 1, number_of_equations).
    """
    number_of_steps = int((time_end - time_start) / time_step)
    time_values = np.linspace(time_start, time_end, number_of_steps + 1)
    number_of_equations = len(initial_values)
    solution_values = np.zeros((number_of_steps + 1, number_of_equations))
    solution_values[0] = initial_values

    for step_index in range(number_of_steps):
        current_time = time_values[step_index]
        current_values = solution_values[step_index]
        derivatives = derivative_function(current_time, current_values)
        solution_values[step_index + 1] = current_values + time_step * derivatives

    return time_values, solution_values


def lorenz_derivatives(
    _time: float,
    state: npt.NDArray[np.float64],
    sigma: float = 10.0,
    beta: float = 2.666667,
    rho: float = 28.0,
) -> npt.NDArray[np.float64]:
    """Compute the derivatives for the Lorenz system.

    dx/dt = -sigma*x + sigma*y
    dy/dt = rho*x - y - x*z
    dz/dt = -beta*z + x*y

    Args:
        _time: Current time (unused, system is autonomous).
        state: Current state vector [x, y, z].
        sigma: Lorenz parameter sigma.
        beta: Lorenz parameter beta.
        rho: Lorenz parameter rho.

    Returns:
        Derivative vector [dx/dt, dy/dt, dz/dt].
    """
    state_x, state_y, state_z = state
    dxdt = -sigma * state_x + sigma * state_y
    dydt = rho * state_x - state_y - state_x * state_z
    dzdt = -beta * state_z + state_x * state_y
    return np.array([dxdt, dydt, dzdt])


def plot_lorenz_x_vs_t(
    time_values: npt.NDArray[np.float64],
    solution_values: npt.NDArray[np.float64],
    time_step: float,
) -> None:
    """Plot x(t) for the Lorenz system.

    Args:
        time_values: Array of time values.
        solution_values: Solution array with shape (n, 3) for [x, y, z].
        time_step: Integration time step used.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_values, solution_values[:, 0], "b-", linewidth=0.5)
    ax.set_xlabel("Time t")
    ax.set_ylabel("x(t)")
    ax.set_title(f"Lorenz System: x vs t (Explicit Euler, Δt = {time_step})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "lorenz_x_vs_t.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lorenz_x_vs_y(
    solution_values: npt.NDArray[np.float64],
    time_step: float,
) -> None:
    """Plot x vs y phase portrait for the Lorenz system.

    Args:
        solution_values: Solution array with shape (n, 3) for [x, y, z].
        time_step: Integration time step used.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(
        solution_values[:, 0], solution_values[:, 1], "b-", linewidth=0.3, alpha=0.7
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Lorenz System: x vs y (Explicit Euler, Δt = {time_step})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "lorenz_x_vs_y.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_explicit_euler_stability_bound(stiff_coefficient: float) -> float:
    """Compute the maximum stable step-size for the explicit Euler method.

    For dy/dt = -a*y + forcing(t), the explicit Euler method is stable
    when a*Δt <= 2.

    Args:
        stiff_coefficient: The coefficient a from the -a*y term.

    Returns:
        Maximum stable step-size Δt_max = 2/a.
    """
    return 2.0 / stiff_coefficient


def main() -> None:
    """Entry point for Session 4 solutions."""
    # Question 1: Stability bound for explicit Euler
    stiff_coefficient = 200000.0
    max_step_size = compute_explicit_euler_stability_bound(stiff_coefficient)
    print(f"Q1: Δt_max = 2/a = 2/{stiff_coefficient:.0f} = {max_step_size:.5f}")

    # Question 2: Lorenz system
    sigma = 10.0
    beta = 2.666667
    rho = 28.0
    initial_conditions = np.array([5.1, 5.1, 5.1])
    time_step = 0.004

    def lorenz(
        time: float, state: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return lorenz_derivatives(time, state, sigma, beta, rho)

    time_values, solution_values = explicit_euler_system(
        lorenz, initial_conditions, 0.0, 20.0, time_step
    )

    print(f"Q2: y(20) = {solution_values[-1, 1]:.7g}")

    plot_lorenz_x_vs_t(time_values, solution_values, time_step)
    plot_lorenz_x_vs_y(solution_values, time_step)
    print("    Plots saved to figures/")


if __name__ == "__main__":
    main()
