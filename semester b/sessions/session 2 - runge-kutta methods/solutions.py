"""Solutions for Session 2 - Runge-Kutta Methods."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


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


def analytical_solution(time: float, decay_coefficient: float, forcing_coefficient: float, initial_condition: float) -> float:
    """Compute the analytical solution to dy/dt = bt - ay(t).

    y(t) = exp(-at)(y(0) + b/a^2) + bt/a - b/a^2

    Args:
        time: Time value.
        decay_coefficient: Parameter a.
        forcing_coefficient: Parameter b.
        initial_condition: Initial condition y(0).

    Returns:
        Analytical solution value y(t).
    """
    return (
        np.exp(-decay_coefficient * time) * (initial_condition + forcing_coefficient / decay_coefficient**2)
        + forcing_coefficient * time / decay_coefficient
        - forcing_coefficient / decay_coefficient**2
    )


def main() -> None:
    """Entry point for Session 2 solutions."""
    # Problem parameters
    decay_coefficient = 8
    forcing_coefficient = 1
    initial_condition = 4
    maximum_time = 1.0
    time_step = 0.01

    # Define the derivative function
    def derivative(time: float, solution_value: float) -> float:
        return forcing_coefficient * time - decay_coefficient * solution_value

    # Solve using Explicit Euler and Ralston methods
    time_values, explicit_solution = explicit_euler_method(
        derivative, initial_condition, 0, maximum_time, time_step
    )
    _, ralston_solution = ralston_method(derivative, initial_condition, 0, maximum_time, time_step)

    # Compute analytical solution at final time
    analytical_final_solution = analytical_solution(
        maximum_time, decay_coefficient, forcing_coefficient, initial_condition
    )

    # Compute differences from analytical solution
    explicit_difference = explicit_solution[-1] - analytical_final_solution
    ralston_difference = ralston_solution[-1] - analytical_final_solution

    # Print results
    print("=" * 70)
    print("ODE: dy/dt = bt - ay(t), a=8, b=1, y(0)=4")
    print("=" * 70)
    print(f"\nTime step Δt = {time_step}, Integration: t=0 to t={maximum_time}\n")

    print("Analytical solution at maximum_time:")
    print(f"  y({maximum_time}) = {analytical_final_solution:.10f}\n")

    print("Explicit Euler Method:")
    print(f"  y({maximum_time}) = {explicit_solution[-1]:.10f}")
    print(f"  Difference: {explicit_difference:.10f}\n")

    print("Ralston Method (2nd-order Runge-Kutta):")
    print(f"  y({maximum_time}) = {ralston_solution[-1]:.10f}")
    print(f"  Difference: {ralston_difference:.10f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
