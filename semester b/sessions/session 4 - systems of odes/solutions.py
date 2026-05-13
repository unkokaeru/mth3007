"""Solutions for Session 4 - Systems of ODEs.

Covers:
  - Explicit Euler method for systems of ODEs
  - Predator-prey (Lotka-Volterra) equations
  - Stability of explicit Euler for stiff ODEs
"""

import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def lotka_volterra_derivatives(
    _time: float,
    state: npt.NDArray[np.float64],
    prey_growth_rate: float,
    predation_rate: float,
    predator_death_rate: float,
    predator_growth_rate: float,
) -> npt.NDArray[np.float64]:
    """Compute derivatives for the Lotka-Volterra predator-prey equations.

    dx/dt =  prey_growth_rate * x - predation_rate * x * y
    dy/dt = -predator_death_rate * y + predator_growth_rate * x * y

    where x is the prey population and y is the predator population.

    Args:
        _time: Current time (unused; system is autonomous).
        state: Current state vector [prey_count, predator_count].
        prey_growth_rate: Rate a at which prey grow.
        predation_rate: Rate b at which predators consume prey.
        predator_death_rate: Rate c at which predators die.
        predator_growth_rate: Rate d at which predators grow from consuming prey.

    Returns:
        Derivative vector [dx/dt, dy/dt].
    """
    prey_count, predator_count = state
    dprey = prey_growth_rate * prey_count - predation_rate * prey_count * predator_count
    dpredator = (
        -predator_death_rate * predator_count
        + predator_growth_rate * prey_count * predator_count
    )
    return np.array([dprey, dpredator])


def plot_phase_portrait(
    solution_values: npt.NDArray[np.float64],
    initial_prey: float,
    initial_predator: float,
    time_step: float,
) -> None:
    """Plot the predator-prey phase portrait (prey vs predator).

    Args:
        solution_values: Solution array with shape (n, 2) for [prey, predator].
        initial_prey: Initial prey population.
        initial_predator: Initial predator population.
        time_step: Integration time step used.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(8, 6))
    axes.plot(
        solution_values[:, 0],
        solution_values[:, 1],
        "b-",
        linewidth=0.8,
        label="Orbit",
    )
    axes.plot(initial_prey, initial_predator, ".r", markersize=8, label="Start")
    axes.plot(
        solution_values[-1, 0],
        solution_values[-1, 1],
        "xr",
        markersize=8,
        label="End",
    )
    axes.set_xlabel("Prey population x")
    axes.set_ylabel("Predator population y")
    axes.set_title(f"Predator-Prey Phase Portrait (Explicit Euler, Δt = {time_step})")
    axes.legend()
    axes.grid(True, alpha=0.3)

    figure.savefig(FIGURES_DIR / "predator_prey_phase.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figures/predator_prey_phase.png")


def plot_populations_vs_time(
    time_values: npt.NDArray[np.float64],
    solution_values: npt.NDArray[np.float64],
    time_step: float,
) -> None:
    """Plot prey and predator populations as functions of time.

    Args:
        time_values: Array of time values.
        solution_values: Solution array with shape (n, 2) for [prey, predator].
        time_step: Integration time step used.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(10, 5))
    axes.plot(time_values, solution_values[:, 0], "b-", linewidth=1.2, label="Prey x(t)")
    axes.plot(time_values, solution_values[:, 1], "r--", linewidth=1.2, label="Predator y(t)")
    axes.set_xlabel("Time t")
    axes.set_ylabel("Population")
    axes.set_title(f"Predator-Prey Populations vs Time (Explicit Euler, Δt = {time_step})")
    axes.legend()
    axes.grid(True, alpha=0.3)

    figure.savefig(FIGURES_DIR / "predator_prey_time.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figures/predator_prey_time.png")


def compute_explicit_euler_stability_bound(stiff_coefficient: float) -> float:
    """Compute the maximum stable step-size for the explicit Euler method.

    For dy/dt = -a*y, the explicit Euler method is stable when a*Delta_t <= 2.

    Args:
        stiff_coefficient: The coefficient a from the -a*y term.

    Returns:
        Maximum stable step-size Delta_t_max = 2/a.
    """
    return 2.0 / stiff_coefficient


def main() -> None:
    """Entry point for Session 4 solutions."""
    # --- Exercise 4.4: Predator-prey (Lotka-Volterra) ---
    # Parameters from lecture section 23.3
    prey_growth_rate = 1.2
    predation_rate = 0.6
    predator_death_rate = 0.8
    predator_growth_rate = 0.3

    initial_prey = 2.0
    initial_predator = 1.0
    initial_values = np.array([initial_prey, initial_predator])
    time_step = 0.01
    time_end = 30.0

    def predator_prey(
        time: float, state: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return lotka_volterra_derivatives(
            time,
            state,
            prey_growth_rate,
            predation_rate,
            predator_death_rate,
            predator_growth_rate,
        )

    time_values, solution_values = explicit_euler_system(
        predator_prey, initial_values, 0.0, time_end, time_step
    )

    print(f"Predator-prey: final prey={solution_values[-1, 0]:.4f}, predator={solution_values[-1, 1]:.4f}")

    plot_phase_portrait(solution_values, initial_prey, initial_predator, time_step)
    plot_populations_vs_time(time_values, solution_values, time_step)

    # --- Exercise 4.5: Stability of explicit Euler ---
    # For dy/dt = -a*y, explicit Euler is stable when a*dt <= 2 (i.e., dt <= 2/a)
    stiff_coefficient = 200000.0
    max_step_size = compute_explicit_euler_stability_bound(stiff_coefficient)
    print(f"\nStability: Delta_t_max = 2/a = 2/{stiff_coefficient:.0f} = {max_step_size:.5f}")


if __name__ == "__main__":
    main()
