"""Solutions for Session 1 - Introduction to Numerical Methods for ODEs.

This module contains implementations of explicit and implicit Euler methods
for solving ordinary differential equations, along with error analysis
and visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt


def explicit_euler_method(
    derivative_function,
    initial_value,
    time_start,
    time_end,
    time_step,
):
    """Solve an ODE using the explicit Euler method.

    The explicit Euler method uses the approximation:
    y_{n+1} = y_n + h * f(t_n, y_n)

    Parameters
    ----------
    derivative_function : callable
        Function f(t, y) that computes dy/dt given t and y.
    initial_value : float
        Initial condition y(t_0).
    time_start : float
        Starting time t_0.
    time_end : float
        Ending time t_max.
    time_step : float
        Time step size h (Delta t).

    Returns
    -------
    time_values : numpy.ndarray
        Array of time points.
    solution_values : numpy.ndarray
        Array of solution values y(t) at each time point.
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
    derivative_function,
    initial_value,
    time_start,
    time_end,
    time_step,
    max_iterations=100,
    tolerance=1e-10,
):
    """Solve an ODE using the implicit Euler method.

    The implicit Euler method uses the approximation:
    y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})

    This requires solving a nonlinear equation at each step using
    fixed-point iteration.

    Parameters
    ----------
    derivative_function : callable
        Function f(t, y) that computes dy/dt given t and y.
    initial_value : float
        Initial condition y(t_0).
    time_start : float
        Starting time t_0.
    time_end : float
        Ending time t_max.
    time_step : float
        Time step size h (Delta t).
    max_iterations : int, optional
        Maximum number of iterations for fixed-point solver (default: 100).
    tolerance : float, optional
        Convergence tolerance for fixed-point solver (default: 1e-10).

    Returns
    -------
    time_values : numpy.ndarray
        Array of time points.
    solution_values : numpy.ndarray
        Array of solution values y(t) at each time point.
    """
    number_of_steps = int((time_end - time_start) / time_step)
    time_values = np.linspace(time_start, time_end, number_of_steps + 1)
    solution_values = np.zeros(number_of_steps + 1)
    solution_values[0] = initial_value

    for step_index in range(number_of_steps):
        current_time = time_values[step_index + 1]
        previous_value = solution_values[step_index]
        
        # Use fixed-point iteration to solve for y_{n+1}
        next_value_guess = previous_value
        for iteration in range(max_iterations):
            next_value_new = (
                previous_value
                + time_step * derivative_function(current_time, next_value_guess)
            )
            if abs(next_value_new - next_value_guess) < tolerance:
                next_value_guess = next_value_new
                break
            next_value_guess = next_value_new
        
        solution_values[step_index + 1] = next_value_guess

    return time_values, solution_values


def ode_derivative_function(time, solution_value, coefficient_a, coefficient_b):
    """Compute the derivative dy/dt = b*t - a*y.

    Parameters
    ----------
    time : float
        Current time t.
    solution_value : float
        Current solution value y(t).
    coefficient_a : float
        Coefficient a in the ODE.
    coefficient_b : float
        Coefficient b in the ODE.

    Returns
    -------
    float
        Value of dy/dt at the given time and solution value.
    """
    return coefficient_b * time - coefficient_a * solution_value


def analytical_solution(time, initial_value, coefficient_a, coefficient_b):
    """Compute the analytical solution to dy/dt = b*t - a*y.

    The analytical solution is:
    y(t) = exp(-a*t) * (y_0 + b/a^2) + b*t/a - b/a^2

    Parameters
    ----------
    time : float or numpy.ndarray
        Time value(s) at which to evaluate the solution.
    initial_value : float
        Initial condition y(0).
    coefficient_a : float
        Coefficient a in the ODE.
    coefficient_b : float
        Coefficient b in the ODE.

    Returns
    -------
    float or numpy.ndarray
        Analytical solution value(s) at the given time(s).
    """
    exponential_term = np.exp(-coefficient_a * time)
    constant_term = initial_value + coefficient_b / (coefficient_a**2)
    linear_term = coefficient_b * time / coefficient_a
    offset_term = coefficient_b / (coefficient_a**2)
    
    return exponential_term * constant_term + linear_term - offset_term


def compute_error(numerical_solution, analytical_solution_values):
    """Compute the error between numerical and analytical solutions.

    Parameters
    ----------
    numerical_solution : numpy.ndarray
        Numerical solution values.
    analytical_solution_values : numpy.ndarray
        Analytical solution values.

    Returns
    -------
    absolute_error : numpy.ndarray
        Absolute error at each point.
    max_absolute_error : float
        Maximum absolute error.
    root_mean_square_error : float
        Root mean square error.
    """
    absolute_error = np.abs(numerical_solution - analytical_solution_values)
    max_absolute_error = np.max(absolute_error)
    root_mean_square_error = np.sqrt(np.mean(absolute_error**2))
    
    return absolute_error, max_absolute_error, root_mean_square_error


def plot_solutions(
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
):
    """Plot and compare numerical and analytical solutions.

    Parameters
    ----------
    time_values_small : numpy.ndarray
        Time values for small time step.
    explicit_solution_small : numpy.ndarray
        Explicit Euler solution for small time step.
    implicit_solution_small : numpy.ndarray
        Implicit Euler solution for small time step.
    analytical_values_small : numpy.ndarray
        Analytical solution for small time step.
    time_values_large : numpy.ndarray
        Time values for large time step.
    explicit_solution_large : numpy.ndarray
        Explicit Euler solution for large time step.
    implicit_solution_large : numpy.ndarray
        Implicit Euler solution for large time step.
    analytical_values_large : numpy.ndarray
        Analytical solution for large time step.
    time_step_small : float
        Small time step size.
    time_step_large : float
        Large time step size.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot for small time step
    axes[0].plot(
        time_values_small,
        analytical_values_small,
        'k-',
        linewidth=2,
        label='Analytical'
    )
    axes[0].plot(
        time_values_small,
        explicit_solution_small,
        'b--',
        linewidth=1.5,
        label='Explicit Euler'
    )
    axes[0].plot(
        time_values_small,
        implicit_solution_small,
        'r:',
        linewidth=1.5,
        label='Implicit Euler'
    )
    axes[0].set_xlabel('Time t')
    axes[0].set_ylabel('Solution y(t)')
    axes[0].set_title(f'Solution with Δt = {time_step_small}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot for large time step
    axes[1].plot(
        time_values_large,
        analytical_values_large,
        'k-',
        linewidth=2,
        label='Analytical'
    )
    axes[1].plot(
        time_values_large,
        explicit_solution_large,
        'b--',
        linewidth=1.5,
        label='Explicit Euler'
    )
    axes[1].plot(
        time_values_large,
        implicit_solution_large,
        'r:',
        linewidth=1.5,
        label='Implicit Euler'
    )
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('Solution y(t)')
    axes[1].set_title(f'Solution with Δt = {time_step_large}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/solution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_errors(
    time_values_small,
    error_explicit_small,
    error_implicit_small,
    time_values_large,
    error_explicit_large,
    error_implicit_large,
    time_step_small,
    time_step_large,
):
    """Plot error comparison for different methods and time steps.

    Parameters
    ----------
    time_values_small : numpy.ndarray
        Time values for small time step.
    error_explicit_small : numpy.ndarray
        Absolute error for explicit Euler with small time step.
    error_implicit_small : numpy.ndarray
        Absolute error for implicit Euler with small time step.
    time_values_large : numpy.ndarray
        Time values for large time step.
    error_explicit_large : numpy.ndarray
        Absolute error for explicit Euler with large time step.
    error_implicit_large : numpy.ndarray
        Absolute error for implicit Euler with large time step.
    time_step_small : float
        Small time step size.
    time_step_large : float
        Large time step size.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot for small time step
    axes[0].semilogy(
        time_values_small,
        error_explicit_small,
        'b-',
        linewidth=1.5,
        label='Explicit Euler'
    )
    axes[0].semilogy(
        time_values_small,
        error_implicit_small,
        'r-',
        linewidth=1.5,
        label='Implicit Euler'
    )
    axes[0].set_xlabel('Time t')
    axes[0].set_ylabel('Absolute Error')
    axes[0].set_title(f'Error with Δt = {time_step_small}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot for large time step
    axes[1].semilogy(
        time_values_large,
        error_explicit_large,
        'b-',
        linewidth=1.5,
        label='Explicit Euler'
    )
    axes[1].semilogy(
        time_values_large,
        error_implicit_large,
        'r-',
        linewidth=1.5,
        label='Implicit Euler'
    )
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title(f'Error with Δt = {time_step_large}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/error_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to solve the ODE and generate plots."""
    # Problem parameters
    coefficient_a = 22.0
    coefficient_b = 1.0
    initial_value = 1.0
    time_start = 0.0
    time_end = 1.0
    time_step_small = 0.01
    time_step_large = 0.1
    
    # Create derivative function with fixed coefficients
    def derivative_function(time, solution_value):
        return ode_derivative_function(
            time, solution_value, coefficient_a, coefficient_b
        )
    
    print("=" * 70)
    print("SOLVING ODE: dy/dt = bt - ay(t)")
    print(f"Parameters: a = {coefficient_a}, b = {coefficient_b}, y(0) = {initial_value}")
    print(f"Time interval: [{time_start}, {time_end}]")
    print("=" * 70)
    
    # Solve with small time step
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
    
    # Solve with large time step
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
    
    # Compute errors for small time step
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
    
    # Compute errors for large time step
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
    
    # Print error analysis
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    
    print(f"\nWith Δt = {time_step_small}:")
    print(f"  Explicit Euler:")
    print(f"    Max Absolute Error: {max_error_explicit_small:.6e}")
    print(f"    RMSE: {rmse_explicit_small:.6e}")
    print(f"  Implicit Euler:")
    print(f"    Max Absolute Error: {max_error_implicit_small:.6e}")
    print(f"    RMSE: {rmse_implicit_small:.6e}")
    
    print(f"\nWith Δt = {time_step_large}:")
    print(f"  Explicit Euler:")
    print(f"    Max Absolute Error: {max_error_explicit_large:.6e}")
    print(f"    RMSE: {rmse_explicit_large:.6e}")
    print(f"  Implicit Euler:")
    print(f"    Max Absolute Error: {max_error_implicit_large:.6e}")
    print(f"    RMSE: {rmse_implicit_large:.6e}")
    
    # Generate plots
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