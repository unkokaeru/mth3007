"""Solutions for Session 6 - Heat Equation (Explicit Finite Difference)."""

import logging
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


def plot_temperature_profiles(
    spatial_values: npt.NDArray[np.float64],
    solution_matrix: npt.NDArray[np.float64],
    time_values: npt.NDArray[np.float64],
    times_to_plot: list[float],
) -> None:
    """Plot temperature profiles at specified times.

    Args:
        spatial_values: Array of spatial grid points (cm).
        solution_matrix: Solution array of shape (time_steps+1, spatial_points).
        time_values: Array of time values (s).
        times_to_plot: List of times at which to plot profiles.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    line_styles = ["k-", "b--", "r-."]
    line_widths = [2.0, 1.5, 1.5]

    figure, axes = plt.subplots(figsize=(10, 6))

    for plot_index, target_time in enumerate(times_to_plot):
        time_index = int(np.argmin(np.abs(time_values - target_time)))
        axes.plot(
            spatial_values,
            solution_matrix[time_index, :],
            line_styles[plot_index],
            linewidth=line_widths[plot_index],
            label=f"t = {int(target_time)} s",
        )

    axes.set_xlabel("Position x (cm)")
    axes.set_ylabel("Temperature u (°C)")
    axes.set_title("Temperature Distribution in Rod (Explicit FD Scheme)")
    axes.legend()
    axes.grid(True, alpha=0.3)

    figure.savefig(FIGURES_DIR / "temperature_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figures/temperature_profiles.png")


def main() -> None:
    """Entry point for Session 6 solutions."""
    # Problem parameters
    rod_length = 10.0  # cm
    diffusion_coefficient = 0.735  # cm^2/s
    spatial_step = 0.2  # cm
    time_step = 0.01  # s
    time_end = 12.0  # s
    boundary_left = 100.0  # °C
    boundary_right = 25.0  # °C

    # Build initial condition: 0°C interior, boundaries fixed
    number_of_spatial_points = int(rod_length / spatial_step) + 1
    initial_condition = np.zeros(number_of_spatial_points)
    initial_condition[0] = boundary_left
    initial_condition[-1] = boundary_right

    # Stability check
    stability_ratio = diffusion_coefficient * time_step / spatial_step**2
    print(f"Stability parameter r = α·Δt/Δx² = {stability_ratio:.6f} (must be < 0.5)")

    # Solve
    spatial_values, time_values, solution_matrix = explicit_heat_equation_1d(
        diffusion_coefficient,
        rod_length,
        spatial_step,
        time_step,
        time_end,
        initial_condition,
        boundary_left,
        boundary_right,
    )

    # Q6.2: Plot temperature profiles
    plot_temperature_profiles(
        spatial_values, solution_matrix, time_values, [0.0, 6.0, 12.0]
    )

    # Q6.3: Temperature at t=12s, x=4cm
    time_index_12 = int(12.0 / time_step)
    spatial_index_4 = int(4.0 / spatial_step)
    temperature_at_12_4 = solution_matrix[time_index_12, spatial_index_4]
    print(f"Q6.3: u(12, 4) = {temperature_at_12_4:.6f} °C")


if __name__ == "__main__":
    main()
