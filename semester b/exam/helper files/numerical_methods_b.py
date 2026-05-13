"""Library of numerical methods for MTH3007b Numerical Methods (Semester B).

Provides implementations of all methods covered in the module:
- ODE solvers (scalar): explicit Euler, implicit Euler, Ralston, implicit trapezoid
- ODE solvers (systems): explicit Euler system, RK4 system
- PDE solvers: explicit heat equation (FTCS), implicit heat equation (BTCS)
- Elliptic PDE: Liebmann's method (2D Laplace equation)
- Linear algebra: Gaussian elimination (augmented matrix form)
- Stochastic: Euler-Maruyama, Ornstein-Uhlenbeck process
- Probability: Monte Carlo integration
- Error analysis and visualisation utilities
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

        stage_1 = derivative_function(current_time, current_values)
        stage_2 = derivative_function(
            current_time + time_step / 2, current_values + time_step / 2 * stage_1
        )
        stage_3 = derivative_function(
            current_time + time_step / 2, current_values + time_step / 2 * stage_2
        )
        stage_4 = derivative_function(
            current_time + time_step, current_values + time_step * stage_3
        )

        solution_values[step_index + 1] = current_values + (time_step / 6) * (
            stage_1 + 2 * stage_2 + 2 * stage_3 + stage_4
        )

    return time_values, solution_values


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


def explicit_heat_equation_1d(
    diffusion_coefficient: float,
    rod_length: float,
    spatial_step: float,
    time_step: float,
    time_end: float,
    initial_condition: npt.NDArray[np.float64],
    boundary_left: float,
    boundary_right: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve the 1D heat equation using the explicit finite-difference scheme.

    Uses forward Euler in time and central difference in space:
    u_j^{n+1} = u_j^n + r * (u_{j-1}^n - 2*u_j^n + u_{j+1}^n)
    where r = alpha * dt / dx^2.

    Args:
        diffusion_coefficient: Heat diffusion coefficient alpha (cm^2/s).
        rod_length: Length of the rod L (cm).
        spatial_step: Spatial step size Delta x (cm).
        time_step: Time step size Delta t (s).
        time_end: Final time to solve until (s).
        initial_condition: Initial temperature profile u(0, x) at all grid points.
        boundary_left: Fixed temperature at x=0 for all t.
        boundary_right: Fixed temperature at x=L for all t.

    Returns:
        Tuple of (spatial_values, time_values, solution_matrix) where
        solution_matrix has shape (number_of_time_steps + 1, number_of_spatial_points).
    """
    number_of_spatial_points = int(rod_length / spatial_step) + 1
    number_of_time_steps = int(time_end / time_step)
    stability_ratio = diffusion_coefficient * time_step / spatial_step**2

    spatial_values = np.linspace(0, rod_length, number_of_spatial_points)
    time_values = np.linspace(0, time_end, number_of_time_steps + 1)

    solution_matrix = np.zeros((number_of_time_steps + 1, number_of_spatial_points))
    solution_matrix[0, :] = initial_condition

    for time_index in range(number_of_time_steps):
        for spatial_index in range(1, number_of_spatial_points - 1):
            solution_matrix[time_index + 1, spatial_index] = (
                solution_matrix[time_index, spatial_index]
                + stability_ratio * (
                    solution_matrix[time_index, spatial_index - 1]
                    - 2 * solution_matrix[time_index, spatial_index]
                    + solution_matrix[time_index, spatial_index + 1]
                )
            )
        solution_matrix[time_index + 1, 0] = boundary_left
        solution_matrix[time_index + 1, -1] = boundary_right

    return spatial_values, time_values, solution_matrix




def gaussian_elimination(
    augmented_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solve the matrix equation Ax = b using Gaussian elimination.

    Implements the algorithm from lecture notes section 42.4. The input is
    an augmented matrix M where the last column contains b and the remaining
    columns contain A. The matrix is triangularised in-place.

    Args:
        augmented_matrix: Augmented matrix [A | b] of shape (n, n+1).
            This array is modified in-place during triangularisation.

    Returns:
        Solution vector x of length n.
    """
    matrix = augmented_matrix.astype(float)
    number_of_rows = matrix.shape[0]
    number_of_cols = matrix.shape[1]

    # Triangularisation
    for pivot_row in range(number_of_rows - 1):
        for lower_row in range(pivot_row + 1, number_of_rows):
            elimination_coefficient = matrix[lower_row, pivot_row] / matrix[pivot_row, pivot_row]
            for col_index in range(pivot_row, number_of_cols):
                matrix[lower_row, col_index] -= matrix[pivot_row, col_index] * elimination_coefficient

    # Back substitution
    solution_vector = np.zeros(number_of_rows)
    for row_index in range(number_of_rows - 1, -1, -1):
        solution_vector[row_index] = matrix[row_index, number_of_cols - 1]
        for col_index in range(row_index + 1, number_of_cols - 1):
            solution_vector[row_index] -= matrix[row_index, col_index] * solution_vector[col_index]
        solution_vector[row_index] /= matrix[row_index, row_index]

    return solution_vector

def implicit_heat_equation_1d(
    diffusion_coefficient: float,
    rod_length: float,
    spatial_step: float,
    time_step: float,
    time_end: float,
    initial_condition: npt.NDArray[np.float64],
    boundary_left: float,
    boundary_right: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve the 1D heat equation using the implicit (BTCS) finite-difference scheme.

    Solves -r*u_{j-1}^{n+1} + (1+2r)*u_j^{n+1} - r*u_{j+1}^{n+1} = u_j^n at each
    time step, where r = alpha * dt / dx^2. Unconditionally stable.

    Args:
        diffusion_coefficient: Heat diffusion coefficient alpha (cm^2/s).
        rod_length: Length of the rod L (cm).
        spatial_step: Spatial step size Delta x (cm).
        time_step: Time step size Delta t (s).
        time_end: Final time to solve until (s).
        initial_condition: Initial temperature profile u(0, x) at all grid points.
        boundary_left: Fixed temperature at x=0 for all t (Dirichlet).
        boundary_right: Fixed temperature at x=L for all t (Dirichlet).

    Returns:
        Tuple of (spatial_values, time_values, solution_matrix) where
        solution_matrix has shape (number_of_time_steps + 1, number_of_spatial_points).
    """
    number_of_spatial_points = int(rod_length / spatial_step) + 1
    number_of_time_steps = int(time_end / time_step)
    diffusion_number = diffusion_coefficient * time_step / spatial_step**2

    spatial_values = np.linspace(0, rod_length, number_of_spatial_points)
    time_values = np.linspace(0, time_end, number_of_time_steps + 1)

    solution_matrix = np.zeros((number_of_time_steps + 1, number_of_spatial_points))
    solution_matrix[0, :] = initial_condition

    number_of_interior_nodes = number_of_spatial_points - 2
    diagonal_value = 1.0 + 2.0 * diffusion_number
    off_diagonal_value = -diffusion_number

    coefficient_matrix = np.zeros((number_of_interior_nodes, number_of_interior_nodes))
    for node_index in range(number_of_interior_nodes):
        coefficient_matrix[node_index, node_index] = diagonal_value
        if node_index > 0:
            coefficient_matrix[node_index, node_index - 1] = off_diagonal_value
        if node_index < number_of_interior_nodes - 1:
            coefficient_matrix[node_index, node_index + 1] = off_diagonal_value

    for time_index in range(number_of_time_steps):
        rhs_vector = solution_matrix[time_index, 1:-1].copy()
        rhs_vector[0] += diffusion_number * boundary_left
        rhs_vector[-1] += diffusion_number * boundary_right

        inverse_matrix = np.linalg.inv(coefficient_matrix)
        interior_solution = np.matmul(inverse_matrix, rhs_vector)

        solution_matrix[time_index + 1, 1:-1] = interior_solution
        solution_matrix[time_index + 1, 0] = boundary_left
        solution_matrix[time_index + 1, -1] = boundary_right

    return spatial_values, time_values, solution_matrix


def liebmann_method(
    plate_width: float,
    plate_height: float,
    spatial_step: float,
    boundary_top: float,
    boundary_bottom: float,
    boundary_left: float,
    boundary_right: float,
    tolerance: float = 1e-4,
    max_iterations: int = 10000,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve the 2D Laplace equation on a rectangular plate using Liebmann's method.

    Liebmann's method (Gauss-Seidel iteration for the Laplacian) maintains two arrays:
    the current iterate (temperature_grid) and the updated iterate (updated_grid).
    Each sweep computes all updated values from the current values, then replaces the
    current array. This matches the two-array algorithm in the lecture notes (section 46.4).

    The update rule for each interior node is:
    unew_{i,j} = (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1}) / 4

    Iteration stops when max |unew_{i,j} - u_{i,j}| < tolerance for all interior nodes.

    Args:
        plate_width: Width of the plate in x-direction.
        plate_height: Height of the plate in y-direction.
        spatial_step: Grid spacing Delta x = Delta y (must be equal).
        boundary_top: Temperature along the top edge (y = plate_height).
        boundary_bottom: Temperature along the bottom edge (y = 0).
        boundary_left: Temperature along the left edge (x = 0).
        boundary_right: Temperature along the right edge (x = plate_width).
        tolerance: Convergence criterion: max nodal change < tolerance (e.g. 1e-4).
        max_iterations: Maximum number of iterations.

    Returns:
        Tuple of (x_values, y_values, temperature_grid) where temperature_grid
        has shape (number_of_y_nodes, number_of_x_nodes).
    """
    number_of_x_nodes = int(plate_width / spatial_step) + 1
    number_of_y_nodes = int(plate_height / spatial_step) + 1

    x_values = np.linspace(0, plate_width, number_of_x_nodes)
    y_values = np.linspace(0, plate_height, number_of_y_nodes)

    # Step 1-3: Initialise u_xy and set boundary conditions
    temperature_grid = np.zeros((number_of_y_nodes, number_of_x_nodes))
    temperature_grid[-1, :] = boundary_top       # top row (y = plate_height)
    temperature_grid[0, :] = boundary_bottom     # bottom row (y = 0)
    temperature_grid[:, 0] = boundary_left       # left column
    temperature_grid[:, -1] = boundary_right     # right column

    for _ in range(max_iterations):
        # Step 4: Compute unew_xy from u_xy (do NOT use updated values within this sweep)
        updated_grid = temperature_grid.copy()

        for row_index in range(1, number_of_y_nodes - 1):
            for col_index in range(1, number_of_x_nodes - 1):
                updated_grid[row_index, col_index] = (
                    temperature_grid[row_index - 1, col_index]
                    + temperature_grid[row_index + 1, col_index]
                    + temperature_grid[row_index, col_index - 1]
                    + temperature_grid[row_index, col_index + 1]
                ) / 4.0

        # Step 5: Maximum difference between u_xy and unew_xy
        max_change = 0.0
        for row_index in range(1, number_of_y_nodes - 1):
            for col_index in range(1, number_of_x_nodes - 1):
                change = abs(updated_grid[row_index, col_index] - temperature_grid[row_index, col_index])
                if change > max_change:
                    max_change = change

        # Step 6: Replace u_xy with unew_xy
        temperature_grid = updated_grid

        # Step 7: Check convergence
        if max_change < tolerance:
            break

    return x_values, y_values, temperature_grid

def monte_carlo_integrate(
    integrand: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    lower_bound: float,
    upper_bound: float,
    number_of_samples: int,
    random_seed: int | None = None,
) -> float:
    """Estimate a definite integral using Monte Carlo sampling.

    Estimator: (b - a) * mean(f(x_i)), x_i ~ Uniform[a, b].

    Args:
        integrand: Function f(x) to integrate.
        lower_bound: Lower limit a.
        upper_bound: Upper limit b.
        number_of_samples: Number of random sample points N.
        random_seed: Optional seed for reproducibility.

    Returns:
        Monte Carlo estimate of the integral.
    """
    rng = np.random.default_rng(random_seed)
    sample_points = rng.uniform(lower_bound, upper_bound, number_of_samples)
    return float((upper_bound - lower_bound) * np.mean(integrand(sample_points)))


def euler_maruyama(
    drift_coefficient: float,
    diffusion_coefficient: float,
    initial_value: float,
    time_start: float,
    time_end: float,
    time_step: float,
    random_seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve a linear SDE dX = -drift_coefficient * X dt + diffusion_coefficient * dW.

    Uses the Euler-Maruyama scheme:
    X_{n+1} = X_n - drift_coefficient * X_n * dt + diffusion_coefficient * sqrt(dt) * Z_n

    Args:
        drift_coefficient: Mean-reversion rate.
        diffusion_coefficient: Noise amplitude sigma.
        initial_value: Initial condition X(t_0).
        time_start: Starting time t_0.
        time_end: Ending time t_max.
        time_step: Time step size Delta t.
        random_seed: Optional seed for reproducibility.

    Returns:
        Tuple of (time_values, path_values) arrays.
    """
    rng = np.random.default_rng(random_seed)
    number_of_steps = int((time_end - time_start) / time_step)
    time_values = np.linspace(time_start, time_end, number_of_steps + 1)
    path_values = np.zeros(number_of_steps + 1)
    path_values[0] = initial_value

    noise_increments = rng.standard_normal(number_of_steps) * np.sqrt(time_step)

    for step_index in range(number_of_steps):
        current_value = path_values[step_index]
        path_values[step_index + 1] = (
            current_value
            - drift_coefficient * current_value * time_step
            + diffusion_coefficient * noise_increments[step_index]
        )

    return time_values, path_values


def ornstein_uhlenbeck_process(
    mean_reversion_rate: float,
    diffusion_coefficient: float,
    initial_value: float,
    time_start: float,
    time_end: float,
    time_step: float,
    random_seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simulate an Ornstein-Uhlenbeck process using the Euler-Maruyama scheme.

    Solves: dV = -mean_reversion_rate * V dt + dW

    Numerical scheme (lecture section 38.6.1):
    V(t + dt) = (1 - mean_reversion_rate * dt) * V(t) + sqrt(dt) * Z(t)

    Args:
        mean_reversion_rate: Rate k controlling reversion towards zero.
        diffusion_coefficient: Noise amplitude (coefficient of dW).
        initial_value: Initial condition V(t_0).
        time_start: Starting time t_0.
        time_end: Ending time t_max.
        time_step: Time step size Delta t.
        random_seed: Optional seed for reproducibility.

    Returns:
        Tuple of (time_values, path_values) arrays.
    """
    rng = np.random.default_rng(random_seed)
    number_of_steps = int((time_end - time_start) / time_step)
    time_values = np.linspace(time_start, time_end, number_of_steps + 1)
    path_values = np.zeros(number_of_steps + 1)
    path_values[0] = initial_value

    noise_increments = rng.standard_normal(number_of_steps) * np.sqrt(time_step)

    for step_index in range(number_of_steps):
        current_value = path_values[step_index]
        path_values[step_index + 1] = (
            (1.0 - mean_reversion_rate * time_step) * current_value
            + diffusion_coefficient * noise_increments[step_index]
        )

    return time_values, path_values

