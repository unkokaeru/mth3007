"""Library of numerical methods for solving ordinary differential equations.

This module contains implementations of explicit and implicit Euler methods,
error analysis utilities, and visualization tools for ODE solutions.
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

matplotlib.use("Agg")


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
    output_dir: Path | None = None,
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
        output_dir: Directory to save figures. Defaults to current directory.
    """
    if output_dir is None:
        output_dir = Path.cwd()

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
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "solution_comparison.png", dpi=300, bbox_inches="tight")
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
    output_dir: Path | None = None,
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
        output_dir: Directory to save figures. Defaults to current directory.
    """
    if output_dir is None:
        output_dir = Path.cwd()

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
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "error_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def ralston_method(
    derivative_function: Callable[[float, float], float],
    initial_value: float,
    time_start: float,
    time_end: float,
    time_step: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve an ODE using the Ralston method (2nd-order Runge-Kutta).

    The Ralston method is given by:
    first_stage = f(t_n, y_n)
    second_stage = f(t_n + 2h/3, y_n + 2h/3*first_stage)
    y_{n+1} = y_n + h*(1/4*first_stage + 3/4*second_stage)

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

        first_stage = derivative_function(current_time, current_value)
        second_stage = derivative_function(
            current_time + 2 * time_step / 3,
            current_value + 2 * time_step / 3 * first_stage,
        )

        solution_values[step_index + 1] = (
            current_value + time_step * (first_stage / 4 + 3 * second_stage / 4)
        )

    return time_values, solution_values


def implicit_trapezoid_method(
    derivative_function: Callable[[float, float], float],
    initial_value: float,
    time_start: float,
    time_end: float,
    time_step: float,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve an ODE using the implicit trapezoid method.

    The implicit trapezoid method (trapezoidal rule) uses the approximation:
    y_{n+1} = y_n + (h/2) * [f(t_n, y_n) + f(t_{n+1}, y_{n+1})]

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
        current_time = time_values[step_index]
        next_time = time_values[step_index + 1]
        previous_value = solution_values[step_index]

        # Compute the derivative at the current step
        current_derivative = derivative_function(current_time, previous_value)

        # Fixed-point iteration to solve:
        # y_{n+1} = y_n + (h/2) * [f(t_n, y_n) + f(t_{n+1}, y_{n+1})]
        next_value_guess = previous_value + time_step * current_derivative
        for _ in range(max_iterations):
            next_derivative = derivative_function(next_time, next_value_guess)
            next_value_new = (
                previous_value + (time_step / 2) * (current_derivative + next_derivative)
            )
            correction = abs(next_value_new - next_value_guess)
            next_value_guess = next_value_new
            if correction < tolerance:
                break

        solution_values[step_index + 1] = next_value_guess

    return time_values, solution_values
