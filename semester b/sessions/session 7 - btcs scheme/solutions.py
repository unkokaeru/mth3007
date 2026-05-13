"""Solutions for Session 7 - BTCS Scheme (Implicit Heat Equation)."""

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

    Uses backward Euler in time and central difference in space. At each time
    step this requires solving the tridiagonal linear system:
    -r*u_{j-1}^{n+1} + (1+2r)*u_j^{n+1} - r*u_{j+1}^{n+1} = u_j^n
    where r = alpha * dt / dx^2.

    The tridiagonal system is assembled as a full matrix, inverted with
    numpy.linalg.inv, and applied via matrix multiplication (numpy.matmul),
    matching the library approach shown in the lecture notes (section 42.5).
    For large grids the Thomas algorithm (O(n)) would be more efficient, but
    the lecture notes explicitly state it is not required.

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

    # Number of interior nodes
    number_of_interior_nodes = number_of_spatial_points - 2

    # Build the tridiagonal coefficient matrix (interior nodes only)
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
        # Right-hand side: current interior values
        rhs_vector = solution_matrix[time_index, 1:-1].copy()

        # Apply boundary conditions to rhs
        rhs_vector[0] += diffusion_number * boundary_left
        rhs_vector[-1] += diffusion_number * boundary_right

        # Solve the tridiagonal system
        inverse_matrix = np.linalg.inv(coefficient_matrix)
        interior_solution = np.matmul(inverse_matrix, rhs_vector)

        solution_matrix[time_index + 1, 1:-1] = interior_solution
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
    axes.set_title("Temperature Distribution in Rod (BTCS Implicit Scheme)")
    axes.legend()
    axes.grid(True, alpha=0.3)

    figure.savefig(FIGURES_DIR / "btcs_temperature_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figures/btcs_temperature_profiles.png")


def main() -> None:
    """Entry point for Session 7 solutions."""
    # Problem parameters (same rod as Session 6 for comparison)
    rod_length = 10.0        # cm
    diffusion_coefficient = 0.735  # cm^2/s
    spatial_step = 0.2       # cm
    time_end = 12.0          # s
    boundary_left = 100.0    # °C
    boundary_right = 25.0    # °C

    # BTCS is unconditionally stable, so we can use a much larger time step
    time_step = 0.5          # s (would violate FTCS stability r <= 0.5)

    diffusion_number = diffusion_coefficient * time_step / spatial_step**2
    print(f"Diffusion number r = alpha*dt/dx^2 = {diffusion_number:.4f}")
    print(f"(BTCS is unconditionally stable; FTCS would require r <= 0.5)")

    number_of_spatial_points = int(rod_length / spatial_step) + 1
    initial_condition = np.zeros(number_of_spatial_points)
    initial_condition[0] = boundary_left
    initial_condition[-1] = boundary_right

    spatial_values, time_values, solution_matrix = implicit_heat_equation_1d(
        diffusion_coefficient,
        rod_length,
        spatial_step,
        time_step,
        time_end,
        initial_condition,
        boundary_left,
        boundary_right,
    )

    plot_temperature_profiles(
        spatial_values, solution_matrix, time_values, [0.0, 6.0, 12.0]
    )

    # Temperature at t=12s, x=4cm
    time_index_12 = int(np.argmin(np.abs(time_values - 12.0)))
    spatial_index_4 = int(4.0 / spatial_step)
    temperature_at_12_4 = solution_matrix[time_index_12, spatial_index_4]
    print(f"u(12, 4) = {temperature_at_12_4:.6f} °C")


if __name__ == "__main__":
    main()
