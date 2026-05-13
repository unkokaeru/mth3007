"""Solutions for Session 11 - Revision.

This session covers worked examples from the Chapra & Canale exercise list
given in the lecture revision session.

Chapra & Canale exercise categories (from lecture notes):
  - 1st order ODEs: 25.1, 25.3, 25.4, 25.5, 25.6, 25.17, 25.19, 25.21
  - Systems of 1st order ODEs: 25.7, 25.25, 28.10
  - 2nd order ODEs reduced to systems: 25.16, 25.18, 25.22
  - 1D Monte Carlo integration: 22.1, 22.2, 22.3
  - PDE (heat equation, Dirichlet BCs): 30.7
  - PDE (heat equation, Neumann BCs): 30.16

Stochastic ODEs are listed as possibly appearing in the exam but are not in
the Chapra & Canale list above.

Each worked example below demonstrates the correct method from the module.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# ODE methods (scalar)
# ---------------------------------------------------------------------------


def explicit_euler_method(
    derivative_function: Callable[[float, float], float],
    initial_value: float,
    time_start: float,
    time_end: float,
    time_step: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve a scalar ODE using the explicit Euler method.

    y_{n+1} = y_n + h * f(t_n, y_n)

    Args:
        derivative_function: f(t, y) giving dy/dt.
        initial_value: y(t_0).
        time_start: Starting time t_0.
        time_end: End time t_max.
        time_step: Step size h (Delta t).

    Returns:
        Tuple of (time_values, solution_values).
    """
    number_of_steps = int((time_end - time_start) / time_step)
    time_values = np.linspace(time_start, time_end, number_of_steps + 1)
    solution_values = np.zeros(number_of_steps + 1)
    solution_values[0] = initial_value

    for step_index in range(number_of_steps):
        current_time = time_values[step_index]
        current_value = solution_values[step_index]
        solution_values[step_index + 1] = (
            current_value + time_step * derivative_function(current_time, current_value)
        )

    return time_values, solution_values


def ralston_method(
    derivative_function: Callable[[float, float], float],
    initial_value: float,
    time_start: float,
    time_end: float,
    time_step: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve a scalar ODE using the Ralston method (2nd-order RK).

    first_stage  = f(t_n, y_n)
    second_stage = f(t_n + 2h/3, y_n + 2h/3 * first_stage)
    y_{n+1}      = y_n + h * (first_stage/4 + 3*second_stage/4)

    Args:
        derivative_function: f(t, y) giving dy/dt.
        initial_value: y(t_0).
        time_start: Starting time t_0.
        time_end: End time t_max.
        time_step: Step size h (Delta t).

    Returns:
        Tuple of (time_values, solution_values).
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


def rk4_method(
    derivative_function: Callable[[float, float], float],
    initial_value: float,
    time_start: float,
    time_end: float,
    time_step: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve a scalar ODE using the classical 4th-order Runge-Kutta method.

    stage_1 = f(t_n,       y_n)
    stage_2 = f(t_n + h/2, y_n + h/2 * stage_1)
    stage_3 = f(t_n + h/2, y_n + h/2 * stage_2)
    stage_4 = f(t_n + h,   y_n + h   * stage_3)
    y_{n+1} = y_n + (h/6) * (stage_1 + 2*stage_2 + 2*stage_3 + stage_4)

    Args:
        derivative_function: f(t, y) giving dy/dt.
        initial_value: y(t_0).
        time_start: Starting time t_0.
        time_end: End time t_max.
        time_step: Step size h (Delta t).

    Returns:
        Tuple of (time_values, solution_values).
    """
    number_of_steps = int((time_end - time_start) / time_step)
    time_values = np.linspace(time_start, time_end, number_of_steps + 1)
    solution_values = np.zeros(number_of_steps + 1)
    solution_values[0] = initial_value

    for step_index in range(number_of_steps):
        current_time = time_values[step_index]
        current_value = solution_values[step_index]
        stage_1 = derivative_function(current_time, current_value)
        stage_2 = derivative_function(current_time + time_step / 2, current_value + time_step / 2 * stage_1)
        stage_3 = derivative_function(current_time + time_step / 2, current_value + time_step / 2 * stage_2)
        stage_4 = derivative_function(current_time + time_step, current_value + time_step * stage_3)
        solution_values[step_index + 1] = current_value + (time_step / 6) * (
            stage_1 + 2 * stage_2 + 2 * stage_3 + stage_4
        )

    return time_values, solution_values


# ---------------------------------------------------------------------------
# ODE methods (systems)
# ---------------------------------------------------------------------------


def explicit_euler_system(
    derivative_function: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_values: npt.NDArray[np.float64],
    time_start: float,
    time_end: float,
    time_step: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve a system of ODEs using the explicit Euler method.

    Args:
        derivative_function: g(t, y_vec) giving the vector dy/dt.
        initial_values: Initial condition vector y(t_0).
        time_start: Starting time t_0.
        time_end: End time t_max.
        time_step: Step size h (Delta t).

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
        solution_values[step_index + 1] = (
            current_values + time_step * derivative_function(current_time, current_values)
        )

    return time_values, solution_values


# ---------------------------------------------------------------------------
# Monte Carlo integration
# ---------------------------------------------------------------------------


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
        integrand: f(x) to integrate.
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


# ---------------------------------------------------------------------------
# Worked revision examples
# ---------------------------------------------------------------------------


def example_first_order_ode() -> None:
    """Chapra 25.1 style: dy/dt = -2y + 4t + 1, y(0) = 4, solve to t=1."""
    def derivative(time: float, solution_value: float) -> float:
        return -2.0 * solution_value + 4.0 * time + 1.0

    initial_value = 4.0
    time_step = 0.1

    time_values, euler_solution = explicit_euler_method(derivative, initial_value, 0.0, 1.0, time_step)
    _, ralston_solution = ralston_method(derivative, initial_value, 0.0, 1.0, time_step)
    _, rk4_solution = rk4_method(derivative, initial_value, 0.0, 1.0, time_step)

    # Analytical: y(t) = 2t + e^(-2t) * (y(0) - 0) + ... (solve by integrating factor)
    # For quick comparison, just print the numerical values at t=1
    print("Example: dy/dt = -2y + 4t + 1, y(0) = 4")
    print(f"  Explicit Euler y(1)  = {euler_solution[-1]:.6f}")
    print(f"  Ralston        y(1)  = {ralston_solution[-1]:.6f}")
    print(f"  RK4            y(1)  = {rk4_solution[-1]:.6f}")


def example_lotka_volterra() -> None:
    """Chapra 25.7 style: Lotka-Volterra predator-prey system."""
    # Parameters: prey growth rate, predation rate, predator death rate, predator growth
    prey_growth_rate = 1.2
    predation_rate = 0.6
    predator_death_rate = 0.8
    predator_growth_rate = 0.3

    def derivatives(
        _time: float,
        state: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        prey_population, predator_population = state
        dprey = (prey_growth_rate - predation_rate * predator_population) * prey_population
        dpredator = (-predator_death_rate + predator_growth_rate * prey_population) * predator_population
        return np.array([dprey, dpredator])

    initial_values = np.array([4.0, 2.0])
    time_step = 0.01

    time_values, solution_values = explicit_euler_system(
        derivatives, initial_values, 0.0, 20.0, time_step
    )

    final_prey = solution_values[-1, 0]
    final_predator = solution_values[-1, 1]
    print(f"Lotka-Volterra at t=20: prey={final_prey:.4f}, predator={final_predator:.4f}")


def example_monte_carlo() -> None:
    """Chapra 22.1 style: Monte Carlo estimate of integral of x^2 from 0 to 1 (exact = 1/3)."""
    def integrand(sample_points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return sample_points**2

    exact_value = 1.0 / 3.0
    for number_of_samples in [100, 1000, 10000]:
        estimate = monte_carlo_integrate(integrand, 0.0, 1.0, number_of_samples, random_seed=42)
        print(f"  N={number_of_samples:6d}: estimate={estimate:.6f}, error={abs(estimate - exact_value):.4e}")


def main() -> None:
    """Run all revision examples."""
    print("=" * 60)
    print("REVISION EXAMPLES")
    print("=" * 60)

    print("\n--- 1st-order ODE ---")
    example_first_order_ode()

    print("\n--- Lotka-Volterra system ---")
    example_lotka_volterra()

    print("\n--- Monte Carlo integration of x^2 ---")
    example_monte_carlo()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
