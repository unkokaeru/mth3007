"""Solutions for Session 8 - Laplace Equation (2D, Liebmann's Method)."""

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

def plot_temperature_contour(
    x_values: npt.NDArray[np.float64],
    y_values: npt.NDArray[np.float64],
    temperature_grid: npt.NDArray[np.float64],
) -> None:
    """Plot a filled contour map of the 2D temperature distribution.

    Args:
        x_values: Array of x grid coordinates.
        y_values: Array of y grid coordinates.
        temperature_grid: 2D temperature array of shape (ny, nx).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    x_mesh, y_mesh = np.meshgrid(x_values, y_values)

    figure, axes = plt.subplots(figsize=(8, 7))
    contour_fill = axes.contourf(x_mesh, y_mesh, temperature_grid, levels=20, cmap="hot")
    contour_lines = axes.contour(x_mesh, y_mesh, temperature_grid, levels=10, colors="k", linewidths=0.5)
    figure.colorbar(contour_fill, ax=axes, label="Temperature (°C)")
    axes.clabel(contour_lines, inline=True, fontsize=8)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title("2D Temperature Distribution - Laplace Equation (Liebmann's Method)")

    figure.savefig(FIGURES_DIR / "laplace_temperature.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figures/laplace_temperature.png")


def main() -> None:
    """Entry point for Session 8 solutions."""
    # Standard textbook problem: heated plate with mixed boundary conditions
    plate_width = 4.0
    plate_height = 4.0
    spatial_step = 1.0
    boundary_top = 100.0
    boundary_bottom = 0.0
    boundary_left = 75.0
    boundary_right = 50.0

    x_values, y_values, temperature_grid = liebmann_method(
        plate_width,
        plate_height,
        spatial_step,
        boundary_top,
        boundary_bottom,
        boundary_left,
        boundary_right,
    )

    print("Temperature grid (rows = y from bottom to top, cols = x left to right):")
    print(np.flipud(temperature_grid))

    plot_temperature_contour(x_values, y_values, temperature_grid)


if __name__ == "__main__":
    main()
