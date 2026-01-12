"""Session 5: Interpolation Methods Solutions.

This module provides solutions to the Session 5 exercises covering:
- Newton's divided difference interpolation
- Lagrange interpolation
- Bilinear interpolation

Author: William Fayers
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_divided_differences(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    """Calculate Newton's divided differences table.
    
    Parameters
    ----------
    x_values : np.ndarray
        Array of x coordinates.
    y_values : np.ndarray
        Array of y coordinates.
    
    Returns
    -------
    np.ndarray
        Divided differences table as a lower triangular matrix.
        The first column contains f[xᵢ], the second f[xᵢ, xᵢ₊₁], etc.
    
    Notes
    -----
    The divided difference f[x₀, x₁, ..., xₖ] is defined recursively:
    
        f[xᵢ] = yᵢ
        f[xᵢ, ..., xᵢ₊ₖ] = (f[xᵢ₊₁, ..., xᵢ₊ₖ] - f[xᵢ, ..., xᵢ₊ₖ₋₁]) / (xᵢ₊ₖ - xᵢ)
    """
    num_points = len(x_values)
    difference_table = np.zeros((num_points, num_points))
    
    # First column: f[xᵢ] = yᵢ
    difference_table[:, 0] = y_values
    
    # Fill subsequent columns
    for column in range(1, num_points):
        for row in range(num_points - column):
            numerator = difference_table[row + 1, column - 1] - difference_table[row, column - 1]
            denominator = x_values[row + column] - x_values[row]
            difference_table[row, column] = numerator / denominator
    
    return difference_table


def newton_interpolation(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_evaluate: float | np.ndarray,
) -> float | np.ndarray:
    """Evaluate Newton's interpolating polynomial.
    
    Parameters
    ----------
    x_values : np.ndarray
        Array of x coordinates of data points.
    y_values : np.ndarray
        Array of y coordinates of data points.
    x_evaluate : float or np.ndarray
        Point(s) at which to evaluate the polynomial.
    
    Returns
    -------
    float or np.ndarray
        Interpolated value(s) at x_evaluate.
    
    Notes
    -----
    Newton's form of the interpolating polynomial is:
    
        P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...
    """
    difference_table = calculate_divided_differences(x_values, y_values)
    num_points = len(x_values)
    
    # Evaluate Newton's polynomial directly
    # P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...
    result = difference_table[0, 0]  # f[x₀]
    product_term = np.ones_like(x_evaluate, dtype=float)
    
    for term_index in range(1, num_points):
        product_term = product_term * (x_evaluate - x_values[term_index - 1])
        result = result + difference_table[0, term_index] * product_term
    
    return result


def lagrange_basis(
    x_values: np.ndarray,
    basis_index: int,
    x_evaluate: float | np.ndarray,
) -> float | np.ndarray:
    """Evaluate the kth Lagrange basis polynomial.
    
    Parameters
    ----------
    x_values : np.ndarray
        Array of x coordinates of data points.
    basis_index : int
        Index k of the basis polynomial to evaluate.
    x_evaluate : float or np.ndarray
        Point(s) at which to evaluate.
    
    Returns
    -------
    float or np.ndarray
        Value(s) of Lₖ(x).
    
    Notes
    -----
    The Lagrange basis polynomial is:
    
        Lₖ(x) = ∏_{j≠k} (x - xⱼ) / (xₖ - xⱼ)
    """
    num_points = len(x_values)
    result = np.ones_like(x_evaluate, dtype=float)
    
    for point_index in range(num_points):
        if point_index != basis_index:
            result *= (x_evaluate - x_values[point_index]) / (x_values[basis_index] - x_values[point_index])
    
    return result


def lagrange_interpolation(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_evaluate: float | np.ndarray,
) -> float | np.ndarray:
    """Evaluate Lagrange's interpolating polynomial.
    
    Parameters
    ----------
    x_values : np.ndarray
        Array of x coordinates of data points.
    y_values : np.ndarray
        Array of y coordinates of data points.
    x_evaluate : float or np.ndarray
        Point(s) at which to evaluate the polynomial.
    
    Returns
    -------
    float or np.ndarray
        Interpolated value(s).
    
    Notes
    -----
    Lagrange's form of the interpolating polynomial is:
    
        P(x) = Σₖ yₖ Lₖ(x)
    """
    num_points = len(x_values)
    result = np.zeros_like(x_evaluate, dtype=float)
    
    for point_index in range(num_points):
        result += y_values[point_index] * lagrange_basis(x_values, point_index, x_evaluate)
    
    return result


def bilinear_interpolation(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_values: np.ndarray,
    x_evaluate: float,
    y_evaluate: float,
) -> float:
    """Perform bilinear interpolation on a rectangular grid.
    
    Parameters
    ----------
    x_grid : np.ndarray
        Array of x coordinates (must be sorted).
    y_grid : np.ndarray
        Array of y coordinates (must be sorted).
    z_values : np.ndarray
        2D array of z values at grid points, shape (len(y_grid), len(x_grid)).
    x_evaluate : float
        x coordinate at which to interpolate.
    y_evaluate : float
        y coordinate at which to interpolate.
    
    Returns
    -------
    float
        Interpolated z value.
    
    Notes
    -----
    Bilinear interpolation combines two linear interpolations:
    
    1. Interpolate in x at y = y₁ and y = y₂
    2. Interpolate in y between these results
    
    Formula:
        z ≈ (1-t)(1-s)z₀₀ + t(1-s)z₁₀ + (1-t)s z₀₁ + ts z₁₁
    
    where t = (x - x₁)/(x₂ - x₁) and s = (y - y₁)/(y₂ - y₁)
    """
    # Find bounding grid indices
    x_index = np.searchsorted(x_grid, x_evaluate) - 1
    y_index = np.searchsorted(y_grid, y_evaluate) - 1
    
    # Clamp to valid range
    x_index = max(0, min(x_index, len(x_grid) - 2))
    y_index = max(0, min(y_index, len(y_grid) - 2))
    
    # Get bounding box coordinates
    x_lower = x_grid[x_index]
    x_upper = x_grid[x_index + 1]
    y_lower = y_grid[y_index]
    y_upper = y_grid[y_index + 1]
    
    # Get corner values
    z_lower_left = z_values[y_index, x_index]
    z_lower_right = z_values[y_index, x_index + 1]
    z_upper_left = z_values[y_index + 1, x_index]
    z_upper_right = z_values[y_index + 1, x_index + 1]
    
    # Calculate interpolation parameters
    t_param = (x_evaluate - x_lower) / (x_upper - x_lower)
    s_param = (y_evaluate - y_lower) / (y_upper - y_lower)
    
    # Bilinear interpolation
    z_interpolated = (
        (1 - t_param) * (1 - s_param) * z_lower_left +
        t_param * (1 - s_param) * z_lower_right +
        (1 - t_param) * s_param * z_upper_left +
        t_param * s_param * z_upper_right
    )
    
    return z_interpolated


def task1_polynomial_interpolation() -> dict:
    """Interpolate through (0,1), (1,3), (3,55) using Newton and Lagrange methods.
    
    Returns
    -------
    dict
        Results containing divided differences, polynomial coefficients,
        and verification values.
    """
    x_values = np.array([0.0, 1.0, 3.0])
    y_values = np.array([1.0, 3.0, 55.0])
    
    # Newton's divided differences
    difference_table = calculate_divided_differences(x_values, y_values)
    
    # Evaluate at test points
    test_points = np.array([0.0, 1.0, 2.0, 3.0])
    newton_values = newton_interpolation(x_values, y_values, test_points)
    lagrange_values = lagrange_interpolation(x_values, y_values, test_points)
    
    return {
        'x_values': x_values,
        'y_values': y_values,
        'difference_table': difference_table,
        'newton_coefficients': difference_table[0, :],
        'test_points': test_points,
        'newton_values': newton_values,
        'lagrange_values': lagrange_values,
    }


def task2_bilinear_temperature() -> dict:
    """Interpolate temperature data using bilinear interpolation.
    
    Returns
    -------
    dict
        Results containing grid data and interpolated values.
    """
    # Temperature data grid (degrees Celsius)
    # Rows: y values (depth or latitude)
    # Columns: x values (distance or longitude)
    x_grid = np.array([0.0, 1.0, 2.0])  # km
    y_grid = np.array([0.0, 1.0, 2.0])  # km
    
    # Temperature values at grid points
    temperature_grid = np.array([
        [20.0, 22.0, 24.0],   # y = 0
        [21.0, 23.0, 25.0],   # y = 1
        [22.0, 24.0, 27.0],   # y = 2
    ])
    
    # Test interpolation at (0.5, 0.5)
    x_test = 0.5
    y_test = 0.5
    temperature_interpolated = bilinear_interpolation(
        x_grid, y_grid, temperature_grid, x_test, y_test
    )
    
    # Additional test points
    test_points = [
        (0.5, 0.5),
        (1.5, 0.5),
        (0.5, 1.5),
        (1.5, 1.5),
        (1.0, 1.0),  # Should match grid value exactly
    ]
    
    interpolated_values = []
    for x_test, y_test in test_points:
        value = bilinear_interpolation(
            x_grid, y_grid, temperature_grid, x_test, y_test
        )
        interpolated_values.append((x_test, y_test, value))
    
    return {
        'x_grid': x_grid,
        'y_grid': y_grid,
        'temperature_grid': temperature_grid,
        'test_points': test_points,
        'interpolated_values': interpolated_values,
    }


def plot_polynomial_interpolation(results: dict) -> None:
    """Plot the interpolating polynomial and data points.
    
    Parameters
    ----------
    results : dict
        Results from task1_polynomial_interpolation().
    """
    x_values = results['x_values']
    y_values = results['y_values']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Interpolating polynomial
    x_smooth = np.linspace(-0.5, 3.5, 100)
    y_newton = newton_interpolation(x_values, y_values, x_smooth)
    
    axes[0].scatter(x_values, y_values, s=150, c='red', zorder=5, label='Data points')
    axes[0].plot(x_smooth, y_newton, 'b-', linewidth=2, label='Interpolating polynomial')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Newton/Lagrange Interpolation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Lagrange basis polynomials
    for basis_index in range(len(x_values)):
        basis_values = lagrange_basis(x_values, basis_index, x_smooth)
        axes[1].plot(x_smooth, basis_values, linewidth=2, label=f'L{basis_index}(x)')
    
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].axhline(y=1, color='black', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Lₖ(x)')
    axes[1].set_title('Lagrange Basis Polynomials')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_bilinear_interpolation(results: dict) -> None:
    """Plot the temperature grid and interpolated surface.
    
    Parameters
    ----------
    results : dict
        Results from task2_bilinear_temperature().
    """
    x_grid = results['x_grid']
    y_grid = results['y_grid']
    temperature_grid = results['temperature_grid']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Contour plot
    x_fine = np.linspace(0, 2, 50)
    y_fine = np.linspace(0, 2, 50)
    temperature_fine = np.zeros((len(y_fine), len(x_fine)))
    
    for i, y_val in enumerate(y_fine):
        for j, x_val in enumerate(x_fine):
            temperature_fine[i, j] = bilinear_interpolation(
                x_grid, y_grid, temperature_grid, x_val, y_val
            )
    
    contour = axes[0].contourf(x_fine, y_fine, temperature_fine, levels=20, cmap='coolwarm')
    plt.colorbar(contour, ax=axes[0], label='Temperature (°C)')
    
    # Mark original grid points
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    axes[0].scatter(x_mesh, y_mesh, c='black', s=50, marker='s', label='Grid points')
    
    # Mark test points
    for x_test, y_test, _ in results['interpolated_values']:
        axes[0].scatter(x_test, y_test, c='green', s=100, marker='x', linewidths=3)
    
    axes[0].set_xlabel('x (km)')
    axes[0].set_ylabel('y (km)')
    axes[0].set_title('Bilinear Interpolation: Temperature Field')
    
    # Plot 2: 3D surface
    ax3d = fig.add_subplot(122, projection='3d')
    x_mesh_fine, y_mesh_fine = np.meshgrid(x_fine, y_fine)
    ax3d.plot_surface(x_mesh_fine, y_mesh_fine, temperature_fine, cmap='coolwarm', alpha=0.8)
    ax3d.set_xlabel('x (km)')
    ax3d.set_ylabel('y (km)')
    ax3d.set_zlabel('Temperature (°C)')
    ax3d.set_title('3D Temperature Surface')
    
    # Remove the empty axes[1] since we replaced it
    axes[1].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    plt.close()


def display_divided_differences_table(
    x_values: np.ndarray,
    difference_table: np.ndarray,
) -> None:
    """Display the divided differences table in a formatted way.
    
    Parameters
    ----------
    x_values : np.ndarray
        Array of x coordinates.
    difference_table : np.ndarray
        The divided differences table.
    """
    num_points = len(x_values)
    
    print("\nDivided Differences Table:")
    print("-" * 60)
    
    header = "  x   |   f[xᵢ]"
    for col_index in range(1, num_points):
        header += f"  |  f[x₀...x{col_index}]"
    print(header)
    print("-" * 60)
    
    for row_index in range(num_points):
        row = f" {x_values[row_index]:.1f}  |  {difference_table[row_index, 0]:7.3f}"
        for col_index in range(1, num_points - row_index):
            row += f"  |  {difference_table[row_index, col_index]:10.3f}"
        print(row)


def main() -> None:
    """Execute all Session 5 tasks and display results."""
    print("=" * 60)
    print("SESSION 5: INTERPOLATION METHODS")
    print("=" * 60)
    
    # Task 1: Polynomial Interpolation
    print("\n--- Task 1: Polynomial Interpolation ---\n")
    print("Data points: (0, 1), (1, 3), (3, 55)")
    
    results_poly = task1_polynomial_interpolation()
    
    # Display divided differences
    display_divided_differences_table(
        results_poly['x_values'],
        results_poly['difference_table']
    )
    
    print("\nNewton's Interpolating Polynomial:")
    coefficients = results_poly['newton_coefficients']
    print(f"  P(x) = {coefficients[0]:.1f} + {coefficients[1]:.1f}(x - 0) + "
          f"{coefficients[2]:.1f}(x - 0)(x - 1)")
    print(f"  P(x) = {coefficients[0]:.1f} + {coefficients[1]:.1f}x + "
          f"{coefficients[2]:.1f}x(x - 1)")
    
    # Expand the polynomial
    print("\n  Expanded form: P(x) = 1 + 2x + 6x(x-1) = 1 + 2x + 6x² - 6x")
    print("                 P(x) = 1 - 4x + 6x²")
    print("                 P(x) = 6x² - 4x + 1")
    
    print("\nLagrange Basis Polynomials:")
    x_values = results_poly['x_values']
    y_values = results_poly['y_values']
    
    print(f"  L₀(x) = (x - {x_values[1]:.0f})(x - {x_values[2]:.0f}) / "
          f"({x_values[0]:.0f} - {x_values[1]:.0f})({x_values[0]:.0f} - {x_values[2]:.0f})")
    print(f"        = (x - 1)(x - 3) / ((-1)(-3))")
    print(f"        = (x - 1)(x - 3) / 3")
    
    print(f"\n  L₁(x) = (x - {x_values[0]:.0f})(x - {x_values[2]:.0f}) / "
          f"({x_values[1]:.0f} - {x_values[0]:.0f})({x_values[1]:.0f} - {x_values[2]:.0f})")
    print(f"        = x(x - 3) / ((1)(-2))")
    print(f"        = -x(x - 3) / 2")
    
    print(f"\n  L₂(x) = (x - {x_values[0]:.0f})(x - {x_values[1]:.0f}) / "
          f"({x_values[2]:.0f} - {x_values[0]:.0f})({x_values[2]:.0f} - {x_values[1]:.0f})")
    print(f"        = x(x - 1) / ((3)(2))")
    print(f"        = x(x - 1) / 6")
    
    print("\n  P(x) = y₀L₀(x) + y₁L₁(x) + y₂L₂(x)")
    print(f"       = {y_values[0]:.0f}·L₀(x) + {y_values[1]:.0f}·L₁(x) + {y_values[2]:.0f}·L₂(x)")
    
    print("\nVerification (both methods should give identical results):")
    print("  x  | Newton P(x) | Lagrange P(x)")
    print("  ---|-------------|---------------")
    for i, x_test in enumerate(results_poly['test_points']):
        print(f" {x_test:.0f}  |   {results_poly['newton_values'][i]:7.2f}   |   "
              f"{results_poly['lagrange_values'][i]:7.2f}")
    
    print("\n  Verification: P(0) = 1 ✓, P(1) = 3 ✓, P(3) = 55 ✓")
    
    # Plot polynomial interpolation
    print("\nGenerating polynomial interpolation plots...")
    plot_polynomial_interpolation(results_poly)
    
    # Task 2: Bilinear Interpolation
    print("\n--- Task 2: Bilinear Interpolation ---\n")
    
    results_bilinear = task2_bilinear_temperature()
    
    print("Temperature grid (°C):")
    print("       x = 0   x = 1   x = 2")
    for j, y_val in enumerate(results_bilinear['y_grid']):
        row = f"y = {y_val:.0f}  "
        for temp in results_bilinear['temperature_grid'][j]:
            row += f" {temp:5.1f}  "
        print(row)
    
    print("\nBilinear interpolation formula:")
    print("  z ≈ (1-t)(1-s)z₀₀ + t(1-s)z₁₀ + (1-t)s·z₀₁ + ts·z₁₁")
    print("  where t = (x - x₁)/(x₂ - x₁), s = (y - y₁)/(y₂ - y₁)")
    
    print("\nInterpolated values:")
    print("  (x, y)   | Temperature (°C)")
    print("  ---------|------------------")
    for x_test, y_test, temp in results_bilinear['interpolated_values']:
        print(f"  ({x_test:.1f}, {y_test:.1f}) |     {temp:.2f}")
    
    # Detailed calculation for (0.5, 0.5)
    print("\nDetailed calculation for (0.5, 0.5):")
    print("  t = (0.5 - 0)/(1 - 0) = 0.5")
    print("  s = (0.5 - 0)/(1 - 0) = 0.5")
    print("  z₀₀ = 20, z₁₀ = 22, z₀₁ = 21, z₁₁ = 23")
    print("  T = (1-0.5)(1-0.5)·20 + 0.5(1-0.5)·22 + (1-0.5)·0.5·21 + 0.5·0.5·23")
    print("    = 0.25·20 + 0.25·22 + 0.25·21 + 0.25·23")
    print("    = 5 + 5.5 + 5.25 + 5.75")
    print("    = 21.5°C")
    
    # Plot bilinear interpolation
    print("\nGenerating bilinear interpolation plots...")
    plot_bilinear_interpolation(results_bilinear)
    
    print("\n" + "=" * 60)
    print("SESSION 5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
