"""Solutions for Session 5 - Higher Order ODEs."""

import sys
from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "exam" / "helper files"))

from numerical_methods_b import explicit_euler_system  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"


def rk4_system(
    derivative_function: Callable[
        [float, npt.NDArray[np.float64]], npt.NDArray[np.float64]
    ],
    initial_values: npt.NDArray[np.float64],
    time_start: float,
    time_end: float,
    time_step: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve a system of ODEs using the classical fourth-order Runge-Kutta method.

    The RK4 method for systems uses:
    k1 = f(t_n, y_n)
    k2 = f(t_n + h/2, y_n + h/2 * k1)
    k3 = f(t_n + h/2, y_n + h/2 * k2)
    k4 = f(t_n + h, y_n + h * k3)
    y_{n+1} = y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

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

        k1 = derivative_function(current_time, current_values)
        k2 = derivative_function(
            current_time + time_step / 2, current_values + time_step / 2 * k1
        )
        k3 = derivative_function(
            current_time + time_step / 2, current_values + time_step / 2 * k2
        )
        k4 = derivative_function(
            current_time + time_step, current_values + time_step * k3
        )

        solution_values[step_index + 1] = current_values + (time_step / 6) * (
            k1 + 2 * k2 + 2 * k3 + k4
        )

    return time_values, solution_values


def system_derivatives(
    _time: float,
    state: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute derivatives for the system form of y'' + 6y = 0.

    With y1 = y and y2 = y', the system is:
        dy1/dt = y2
        dy2/dt = -6 * y1

    Args:
        _time: Current time (unused, system is autonomous).
        state: Current state vector [y1, y2].

    Returns:
        Derivative vector [dy1/dt, dy2/dt].
    """
    y1, y2 = state
    return np.array([y2, -6.0 * y1])


def exact_solution(time_values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the exact solution y(t) = -(sqrt(6)/2)*sin(sqrt(6)*t) + 2*cos(sqrt(6)*t).

    Args:
        time_values: Array of time values.

    Returns:
        Array of exact solution values.
    """
    sqrt6 = np.sqrt(6.0)
    return -(sqrt6 / 2) * np.sin(sqrt6 * time_values) + 2 * np.cos(sqrt6 * time_values)


def plot_euler_vs_exact(
    time_values: npt.NDArray[np.float64],
    euler_y: npt.NDArray[np.float64],
    exact_y: npt.NDArray[np.float64],
    time_step: float,
) -> None:
    """Plot forward Euler approximation against exact solution.

    Args:
        time_values: Array of time values.
        euler_y: Forward Euler y(t) values.
        exact_y: Exact y(t) values.
        time_step: Step size used.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_values, exact_y, "k-", linewidth=2, label="Exact")
    ax.plot(time_values, euler_y, "b--", linewidth=1.5, label="Forward Euler")
    ax.set_xlabel("Time t")
    ax.set_ylabel("y(t)")
    ax.set_title(f"Forward Euler vs Exact Solution (Δt = {time_step})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "euler_vs_exact.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_methods(
    time_values: npt.NDArray[np.float64],
    euler_y: npt.NDArray[np.float64],
    rk4_y: npt.NDArray[np.float64],
    exact_y: npt.NDArray[np.float64],
    time_step: float,
) -> None:
    """Plot forward Euler, RK4, and exact solution on the same axes.

    Args:
        time_values: Array of time values.
        euler_y: Forward Euler y(t) values.
        rk4_y: RK4 y(t) values.
        exact_y: Exact y(t) values.
        time_step: Step size used.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_values, exact_y, "k-", linewidth=2, label="Exact")
    ax.plot(time_values, euler_y, "b--", linewidth=1.5, label="Forward Euler")
    ax.plot(time_values, rk4_y, "r-.", linewidth=1.5, label="RK4")
    ax.set_xlabel("Time t")
    ax.set_ylabel("y(t)")
    ax.set_title(f"Forward Euler vs RK4 vs Exact Solution (Δt = {time_step})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "all_methods_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Entry point for Session 5 solutions."""
    # Problem parameters
    initial_values = np.array([2.0, -3.0])  # y(0) = 2, y'(0) = -3
    time_start = 0.0
    time_end = 5.0
    time_step = 0.1

    # -------------------------------------------------------------------------
    # Question 5.1
    # -------------------------------------------------------------------------

    # Q5.1.1: System formulation
    print("Q5.1.1: System of first-order ODEs")
    print("  Let y1(t) = y(t) and y2(t) = y'(t).")
    print("  dy1/dt = y2")
    print("  dy2/dt = -6*y1")
    print("  y1(0) = 2, y2(0) = -3")
    print()

    # Q5.1.2 & Q5.1.3: Forward Euler method
    time_euler, solution_euler = explicit_euler_system(
        system_derivatives, initial_values, time_start, time_end, time_step
    )
    euler_y = solution_euler[:, 0]
    euler_y_at_5 = euler_y[-1]

    print(f"Q5.1.3: Forward Euler y(5) = {euler_y_at_5:.10f}")

    # Exact solution
    exact_y = exact_solution(time_euler)
    exact_y_at_5 = exact_y[-1]
    euler_error_at_5 = abs(euler_y_at_5 - exact_y_at_5)

    print(f"        Exact y(5)          = {exact_y_at_5:.10f}")
    print(f"        Euler error at t=5  = {euler_error_at_5:.10f}")
    print()

    # Q5.1.4: Plot
    plot_euler_vs_exact(time_euler, euler_y, exact_y, time_step)
    print("Q5.1.4: Plot saved to figures/euler_vs_exact.png")
    print()

    # Q5.1.5: Commentary
    print("Q5.1.5: The forward Euler method is only first-order accurate, so")
    print("        errors accumulate rapidly. For this oscillatory problem")
    print("        (y'' + 6y = 0 has purely imaginary eigenvalues ±i*sqrt(6)),")
    print("        the Euler method is unconditionally unstable: the amplitude")
    print("        of the numerical solution grows over time rather than")
    print("        remaining bounded. This is visible in the plot as the Euler")
    print("        solution spirals outward from the true oscillation.")
    print()

    # -------------------------------------------------------------------------
    # Question 5.2
    # -------------------------------------------------------------------------

    # Q5.2.1 & Q5.2.2: RK4 method
    time_rk4, solution_rk4 = rk4_system(
        system_derivatives, initial_values, time_start, time_end, time_step
    )
    rk4_y = solution_rk4[:, 0]
    rk4_y_at_5 = rk4_y[-1]
    rk4_error_at_5 = abs(rk4_y_at_5 - exact_y_at_5)

    print(f"Q5.2.2: RK4 y(5)           = {rk4_y_at_5:.10f}")
    print(f"        RK4 error at t=5   = {rk4_error_at_5:.10f}")
    print()

    # Q5.2.3: Combined plot
    plot_all_methods(time_euler, euler_y, rk4_y, exact_y, time_step)
    print("Q5.2.3: Plot saved to figures/all_methods_comparison.png")
    print()

    # Q5.2.4: Error comparison
    print("Q5.2.4: Error comparison at t = 5:")
    print(f"        Forward Euler error = {euler_error_at_5:.10f}")
    print(f"        RK4 error           = {rk4_error_at_5:.10f}")
    if rk4_error_at_5 > 0:
        print(f"        Ratio (Euler/RK4)   = {euler_error_at_5 / rk4_error_at_5:.2f}")
    print()
    print("        The forward Euler method is first-order (O(h)), so halving h")
    print("        roughly halves the error. RK4 is fourth-order (O(h^4)), so")
    print("        halving h reduces the error by a factor of ~16. The RK4")
    print("        error is orders of magnitude smaller than Euler's, reflecting")
    print("        its higher order. Moreover, RK4 preserves the oscillatory")
    print("        character of the solution far better than Euler, which")
    print("        exhibits artificial amplitude growth due to instability.")


if __name__ == "__main__":
    main()
