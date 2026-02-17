"""Solutions for Session 1 - Introduction to Numerical Methods for ODEs.

This module contains implementations of explicit and implicit Euler methods
for solving ordinary differential equations, along with error analysis
and visualization tools.
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"


def explicit_euler_method(
    derivative_function: Callable[[float, float], float],
    initial_value: float,
    time_start: float,
    time_end: float,
    time_step: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve an ODE using the explicit Euler method.

    The explicit Euler method uses the approximation:
    y_{n+1} = y_n + h * f(t_n, y_n)

    Args:
        derivative_function: Function f(t, y) that computes dy/dt.
        initial_value: Initial condition y(t_0).
        time_start: Starting time t_0.
        time_end: Ending time t_max.
        time_step: Time step size h (Delta t).

    Returns:
        Tuple of (time_values, solution_values) arrays.
    """
    number_of_steps = int((time_end - time_start) / time_step)
    time_values = np.linspace(time_start, time_end, number_of_steps + 1)
    solution_values = np.zeros(number_of_steps + 1)
    solution_values[0] = initial_value

    for step_index in range(number_of_steps):
        current_time = time_values[step_index]
        current_value = solution_values[step_index]
        derivative = derivative_function(current_time, current_value)
        solution_values[step_index + 1] = current_value + time_step * derivative

    return time_values, solution_values


def implicit_euler_method(
    derivative_function: Callable[[float, float], float],
    initial_value: float,
    time_start: float,
    time_end: float,
    time_step: float,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve an ODE using the implicit Euler method.

    The implicit Euler method uses the approximation:
    y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})

    This requires solving a nonlinear equation at each step using
    fixed-point iteration.

    Args:
        derivative_function: Function f(t, y) that computes dy/dt.
        initial_value: Initial condition y(t_0).
        time_start: Starting time t_0.
        time_end: Ending time t_max.
        time_step: Time step size h (Delta t).
        max_iterations: Maximum iterations for fixed-point solver.
        tolerance: Convergence tolerance for fixed-point solver.

    Returns:
        Tuple of (time_values, solution_values) arrays.
    """
    number_of_steps = int((time_end - time_start) / time_step)
    time_values = np.linspace(time_start, time_end, number_of_steps + 1)
    solution_values = np.zeros(number_of_steps + 1)
    solution_values[0] = initial_value

    for step_index in range(number_of_steps):
        current_time = time_values[step_index + 1]
        previous_value = solution_values[step_index]

        next_value_guess = previous_value
        for _ in range(max_iterations):
            residual = (
                next_value_guess
                - previous_value
                - time_step * derivative_function(current_time, next_value_guess)
            )
            perturbation = 1e-8 * max(abs(next_value_guess), 1.0)
            derivative_approx = (
                1.0
                - time_step
                * (
                    derivative_function(
                        current_time, next_value_guess + perturbation
                    )
                    - derivative_function(current_time, next_value_guess)
                )
                / perturbation
            )
            correction = residual / derivative_approx
            next_value_guess -= correction
            if abs(correction) < tolerance:
                break

        solution_values[step_index + 1] = next_value_guess

    return time_values, solution_values


def ode_derivative_function(
    time: float,
    solution_value: float,
    coefficient_a: float,
    coefficient_b: float,
) -> float:
    """Compute the derivative dy/dt = b*t - a*y.

    Args:
        time: Current time t.
        solution_value: Current solution value y(t).
        coefficient_a: Coefficient a in the ODE.
        coefficient_b: Coefficient b in the ODE.

    Returns:
        Value of dy/dt at the given time and solution value.
    """
    return coefficient_b * time - coefficient_a * solution_value


def analytical_solution(
    time: float | npt.NDArray[np.float64],
    initial_value: float,
    coefficient_a: float,
    coefficient_b: float,
) -> npt.NDArray[np.float64]:
    """Compute the analytical solution to dy/dt = b*t - a*y.

    The analytical solution is:
    y(t) = exp(-a*t) * (y_0 + b/a^2) + b*t/a - b/a^2

    Args:
        time: Time value(s) at which to evaluate the solution.
        initial_value: Initial condition y(0).
        coefficient_a: Coefficient a in the ODE.
        coefficient_b: Coefficient b in the ODE.

    Returns:
        Analytical solution value(s) at the given time(s).
    """
    exponential_term = np.exp(-coefficient_a * time)
    constant_term = initial_value + coefficient_b / (coefficient_a**2)
    linear_term = coefficient_b * time / coefficient_a
    offset_term = coefficient_b / (coefficient_a**2)

    return exponential_term * constant_term + linear_term - offset_term


def compute_error(
    numerical_solution: npt.NDArray[np.float64],
    analytical_solution_values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Compute the error between numerical and analytical solutions.

    Args:
        numerical_solution: Numerical solution values.
        analytical_solution_values: Analytical solution values.

    Returns:
        Tuple of (absolute_error, max_absolute_error, root_mean_square_error).
    """
    absolute_error = np.abs(numerical_solution - analytical_solution_values)
    max_absolute_error = np.max(absolute_error)
    root_mean_square_error = np.sqrt(np.mean(absolute_error**2))

    return absolute_error, max_absolute_error, root_mean_square_error


def plot_solutions(
    time_values_small: npt.NDArray[np.float64],
    explicit_solution_small: npt.NDArray[np.float64],
    implicit_solution_small: npt.NDArray[np.float64],
    analytical_values_small: npt.NDArray[np.float64],
    time_values_large: npt.NDArray[np.float64],
    explicit_solution_large: npt.NDArray[np.float64],
    implicit_solution_large: npt.NDArray[np.float64],
    analytical_values_large: npt.NDArray[np.float64],
    time_step_small: float,
    time_step_large: float,
) -> None:
    """Plot and compare numerical and analytical solutions.

    Args:
        time_values_small: Time values for small time step.
        explicit_solution_small: Explicit Euler solution for small time step.
        implicit_solution_small: Implicit Euler solution for small time step.
        analytical_values_small: Analytical solution for small time step.
        time_values_large: Time values for large time step.
        explicit_solution_large: Explicit Euler solution for large time step.
        implicit_solution_large: Implicit Euler solution for large time step.
        analytical_values_large: Analytical solution for large time step.
        time_step_small: Small time step size.
        time_step_large: Large time step size.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(
        time_values_small,
        analytical_values_small,
        "k-",
        linewidth=2,
        label="Analytical",
    )
    axes[0].plot(
        time_values_small,
        explicit_solution_small,
        "b--",
        linewidth=1.5,
        label="Explicit Euler",
    )
    axes[0].plot(
        time_values_small,
        implicit_solution_small,
        "r:",
        linewidth=1.5,
        label="Implicit Euler",
    )
    axes[0].set_xlabel("Time t")
    axes[0].set_ylabel("Solution y(t)")
    axes[0].set_title(f"Solution with Δt = {time_step_small}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        time_values_large,
        analytical_values_large,
        "k-",
        linewidth=2,
        label="Analytical",
    )
    axes[1].plot(
        time_values_large,
        explicit_solution_large,
        "b--",
        linewidth=1.5,
        label="Explicit Euler",
    )
    axes[1].plot(
        time_values_large,
        implicit_solution_large,
        "r:",
        linewidth=1.5,
        label="Implicit Euler",
    )
    axes[1].set_xlabel("Time t")
    axes[1].set_ylabel("Solution y(t)")
    axes[1].set_title(f"Solution with Δt = {time_step_large}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "solution_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_errors(
    time_values_small: npt.NDArray[np.float64],
    error_explicit_small: npt.NDArray[np.float64],
    error_implicit_small: npt.NDArray[np.float64],
    time_values_large: npt.NDArray[np.float64],
    error_explicit_large: npt.NDArray[np.float64],
    error_implicit_large: npt.NDArray[np.float64],
    time_step_small: float,
    time_step_large: float,
) -> None:
    """Plot error comparison for different methods and time steps.

    Args:
        time_values_small: Time values for small time step.
        error_explicit_small: Absolute error for explicit Euler with small step.
        error_implicit_small: Absolute error for implicit Euler with small step.
        time_values_large: Time values for large time step.
        error_explicit_large: Absolute error for explicit Euler with large step.
        error_implicit_large: Absolute error for implicit Euler with large step.
        time_step_small: Small time step size.
        time_step_large: Large time step size.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].semilogy(
        time_values_small,
        error_explicit_small,
        "b-",
        linewidth=1.5,
        label="Explicit Euler",
    )
    axes[0].semilogy(
        time_values_small,
        error_implicit_small,
        "r-",
        linewidth=1.5,
        label="Implicit Euler",
    )
    axes[0].set_xlabel("Time t")
    axes[0].set_ylabel("Absolute Error")
    axes[0].set_title(f"Error with Δt = {time_step_small}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(
        time_values_large,
        error_explicit_large,
        "b-",
        linewidth=1.5,
        label="Explicit Euler",
    )
    axes[1].semilogy(
        time_values_large,
        error_implicit_large,
        "r-",
        linewidth=1.5,
        label="Implicit Euler",
    )
    axes[1].set_xlabel("Time t")
    axes[1].set_ylabel("Absolute Error")
    axes[1].set_title(f"Error with Δt = {time_step_large}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURES_DIR / "error_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Solve the ODE and generate plots."""
    coefficient_a = 22.0
    coefficient_b = 1.0
    initial_value = 1.0
    time_start = 0.0
    time_end = 1.0
    time_step_small = 0.01
    time_step_large = 0.1

    def derivative_function(
        time: float, solution_value: float
    ) -> float:
        return ode_derivative_function(
            time, solution_value, coefficient_a, coefficient_b
        )

    print("=" * 70)
    print("SOLVING ODE: dy/dt = bt - ay(t)")
    print(f"Parameters: a = {coefficient_a}, b = {coefficient_b}, y(0) = {initial_value}")
    print(f"Time interval: [{time_start}, {time_end}]")
    print("=" * 70)

    print(f"\nSolving with Δt = {time_step_small}...")
    time_values_small, explicit_solution_small = explicit_euler_method(
        derivative_function, initial_value, time_start, time_end, time_step_small
    )
    _, implicit_solution_small = implicit_euler_method(
        derivative_function, initial_value, time_start, time_end, time_step_small
    )
    analytical_values_small = analytical_solution(
        time_values_small, initial_value, coefficient_a, coefficient_b
    )

    print(f"Solving with Δt = {time_step_large}...")
    time_values_large, explicit_solution_large = explicit_euler_method(
        derivative_function, initial_value, time_start, time_end, time_step_large
    )
    _, implicit_solution_large = implicit_euler_method(
        derivative_function, initial_value, time_start, time_end, time_step_large
    )
    analytical_values_large = analytical_solution(
        time_values_large, initial_value, coefficient_a, coefficient_b
    )

    (
        error_explicit_small,
        max_error_explicit_small,
        rmse_explicit_small,
    ) = compute_error(explicit_solution_small, analytical_values_small)

    (
        error_implicit_small,
        max_error_implicit_small,
        rmse_implicit_small,
    ) = compute_error(implicit_solution_small, analytical_values_small)

    (
        error_explicit_large,
        max_error_explicit_large,
        rmse_explicit_large,
    ) = compute_error(explicit_solution_large, analytical_values_large)

    (
        error_implicit_large,
        max_error_implicit_large,
        rmse_implicit_large,
    ) = compute_error(implicit_solution_large, analytical_values_large)

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    print(f"\nWith Δt = {time_step_small}:")
    print("  Explicit Euler:")
    print(f"    Max Absolute Error: {max_error_explicit_small:.6e}")
    print(f"    RMSE: {rmse_explicit_small:.6e}")
    print("  Implicit Euler:")
    print(f"    Max Absolute Error: {max_error_implicit_small:.6e}")
    print(f"    RMSE: {rmse_implicit_small:.6e}")

    print(f"\nWith Δt = {time_step_large}:")
    print("  Explicit Euler:")
    print(f"    Max Absolute Error: {max_error_explicit_large:.6e}")
    print(f"    RMSE: {rmse_explicit_large:.6e}")
    print("  Implicit Euler:")
    print(f"    Max Absolute Error: {max_error_implicit_large:.6e}")
    print(f"    RMSE: {rmse_implicit_large:.6e}")

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    print("\nCreating solution comparison plots...")
    plot_solutions(
        time_values_small,
        explicit_solution_small,
        implicit_solution_small,
        analytical_values_small,
        time_values_large,
        explicit_solution_large,
        implicit_solution_large,
        analytical_values_large,
        time_step_small,
        time_step_large,
    )

    print("Creating error comparison plots...")
    plot_errors(
        time_values_small,
        error_explicit_small,
        error_implicit_small,
        time_values_large,
        error_explicit_large,
        error_implicit_large,
        time_step_small,
        time_step_large,
    )

    print("\nAll plots saved to figures/ directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()
