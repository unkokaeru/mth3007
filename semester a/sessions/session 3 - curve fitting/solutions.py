"""Session 3: Polynomial and Multiple Regression Solutions.

This module provides solutions to the Session 3 exercises covering:
- Extended sum calculations for linear regression
- Multiple linear regression
- Gaussian elimination for solving systems

Author: William Fayers
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gauss_eliminate(
    coefficient_matrix: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Solve a system of linear equations using Gaussian elimination.
    
    Solves the system Ax = b using Gaussian elimination followed by
    back substitution.
    
    Parameters
    ----------
    coefficient_matrix : np.ndarray
        The coefficient matrix A of shape (n, n).
    right_hand_side : np.ndarray
        The right-hand side vector b of shape (n,).
    
    Returns
    -------
    np.ndarray
        The solution vector x of shape (n,).
    
    Notes
    -----
    Uses forward elimination to reduce to row echelon form,
    then back substitution to find the solution.
    """
    num_equations = len(right_hand_side)
    
    # Create augmented matrix
    augmented = np.column_stack([
        coefficient_matrix.astype(float),
        right_hand_side.astype(float)
    ])
    
    # Forward elimination
    for pivot_row in range(num_equations):
        # Eliminate entries below pivot
        for row_index in range(pivot_row + 1, num_equations):
            if augmented[pivot_row, pivot_row] != 0:
                factor = augmented[row_index, pivot_row] / augmented[pivot_row, pivot_row]
                augmented[row_index, pivot_row:] -= factor * augmented[pivot_row, pivot_row:]
    
    # Back substitution
    solution = np.zeros(num_equations)
    for row_index in range(num_equations - 1, -1, -1):
        solution[row_index] = (
            augmented[row_index, -1] -
            np.dot(augmented[row_index, row_index + 1:num_equations],
                   solution[row_index + 1:])
        ) / augmented[row_index, row_index]
    
    return solution


def task1_extended_sums() -> tuple[float, float, float, float, float, float, float, float]:
    """Compute extended sums for linear regression data.
    
    Uses x_i = i and y_i = 2i + 0.3 for i = 1, 2, ..., 10.
    
    Returns
    -------
    tuple[float, ...]
        Contains sum_x, sum_y, sum_xy, mean_x, mean_y, a0, a1.
    """
    indices = np.arange(1, 11)
    x_values = indices.astype(float)
    y_values = 2 * indices + 0.3
    
    num_points = len(x_values)
    
    sum_x = np.sum(x_values)
    sum_y = np.sum(y_values)
    sum_xy = np.sum(x_values * y_values)
    sum_x_squared = np.sum(x_values**2)
    
    mean_x = sum_x / num_points
    mean_y = sum_y / num_points
    
    # Calculate regression parameters
    numerator = num_points * sum_xy - sum_x * sum_y
    denominator = num_points * sum_x_squared - sum_x**2
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    
    return sum_x, sum_y, sum_xy, mean_x, mean_y, intercept, slope, sum_x_squared


def task2_quadratic_regression(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    """Fit a quadratic model y = a0 + a1*x + a2*x² to data.
    
    Uses the normal equations derived from minimising the sum of
    squared errors for polynomial regression.
    
    Parameters
    ----------
    x_values : np.ndarray
        Independent variable values.
    y_values : np.ndarray
        Dependent variable values.
    
    Returns
    -------
    np.ndarray
        Array of coefficients [a0, a1, a2].
    
    Notes
    -----
    The normal equations for quadratic regression are:
    
    [n      Σx     Σx²  ] [a0]   [Σy    ]
    [Σx     Σx²    Σx³  ] [a1] = [Σxy   ]
    [Σx²    Σx³    Σx⁴  ] [a2]   [Σx²y  ]
    """
    num_points = len(x_values)
    
    # Calculate required sums
    sum_x = np.sum(x_values)
    sum_x2 = np.sum(x_values**2)
    sum_x3 = np.sum(x_values**3)
    sum_x4 = np.sum(x_values**4)
    sum_y = np.sum(y_values)
    sum_xy = np.sum(x_values * y_values)
    sum_x2y = np.sum(x_values**2 * y_values)
    
    # Build coefficient matrix
    coefficient_matrix = np.array([
        [num_points, sum_x, sum_x2],
        [sum_x, sum_x2, sum_x3],
        [sum_x2, sum_x3, sum_x4],
    ])
    
    # Build right-hand side
    right_hand_side = np.array([sum_y, sum_xy, sum_x2y])
    
    # Solve using Gaussian elimination
    parameters = gauss_eliminate(coefficient_matrix, right_hand_side)
    
    return parameters


def task3_multiple_linear_regression(
    x1_values: np.ndarray,
    x2_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    """Fit a multiple linear regression model y = a0 + a1*x1 + a2*x2.
    
    Parameters
    ----------
    x1_values : np.ndarray
        First independent variable values.
    x2_values : np.ndarray
        Second independent variable values.
    y_values : np.ndarray
        Dependent variable values.
    
    Returns
    -------
    np.ndarray
        Array of coefficients [a0, a1, a2].
    
    Notes
    -----
    The normal equations for multiple linear regression with two
    independent variables are:
    
    [n      Σx₁     Σx₂   ] [a0]   [Σy     ]
    [Σx₁    Σx₁²    Σx₁x₂ ] [a1] = [Σx₁y   ]
    [Σx₂    Σx₁x₂   Σx₂²  ] [a2]   [Σx₂y   ]
    """
    num_points = len(y_values)
    
    # Calculate required sums
    sum_x1 = np.sum(x1_values)
    sum_x2 = np.sum(x2_values)
    sum_x1_squared = np.sum(x1_values**2)
    sum_x2_squared = np.sum(x2_values**2)
    sum_x1_x2 = np.sum(x1_values * x2_values)
    sum_y = np.sum(y_values)
    sum_x1_y = np.sum(x1_values * y_values)
    sum_x2_y = np.sum(x2_values * y_values)
    
    # Build coefficient matrix
    coefficient_matrix = np.array([
        [num_points, sum_x1, sum_x2],
        [sum_x1, sum_x1_squared, sum_x1_x2],
        [sum_x2, sum_x1_x2, sum_x2_squared],
    ])
    
    # Build right-hand side
    right_hand_side = np.array([sum_y, sum_x1_y, sum_x2_y])
    
    # Solve using Gaussian elimination
    parameters = gauss_eliminate(coefficient_matrix, right_hand_side)
    
    return parameters


def plot_multiple_regression_3d(
    x1_values: np.ndarray,
    x2_values: np.ndarray,
    y_values: np.ndarray,
    parameters: np.ndarray,
    title: str = "Multiple Linear Regression",
) -> None:
    """Plot data points and fitted plane for multiple regression.
    
    Parameters
    ----------
    x1_values : np.ndarray
        First independent variable values.
    x2_values : np.ndarray
        Second independent variable values.
    y_values : np.ndarray
        Dependent variable values.
    parameters : np.ndarray
        Fitted parameters [a0, a1, a2].
    title : str, optional
        Plot title.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot data points
    ax.scatter(x1_values, x2_values, y_values, color='blue', s=100, label='Data')
    
    # Create grid for surface
    x1_grid = np.linspace(np.min(x1_values) - 1, np.max(x1_values) + 1, 20)
    x2_grid = np.linspace(np.min(x2_values) - 1, np.max(x2_values) + 1, 20)
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
    
    # Calculate fitted surface
    y_fitted = parameters[0] + parameters[1] * x1_mesh + parameters[2] * x2_mesh
    
    # Plot surface
    ax.plot_surface(x1_mesh, x2_mesh, y_fitted, alpha=0.5, color='red')
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('y')
    ax.set_title(title)
    
    plt.show()
    plt.close()


def task4_linearisation_explanation() -> str:
    """Explain how to linearise y = a0 * x1^a1 * x2^a2 * ... * xn^an.
    
    Returns
    -------
    str
        Explanation of the linearisation process.
    """
    explanation = """
    LINEARISATION OF POWER-LAW MODELS
    ==================================
    
    For a model of the form:
        y = a₀ · x₁^a₁ · x₂^a₂ · ... · xₙ^aₙ
    
    Take the natural logarithm of both sides:
        ln(y) = ln(a₀) + a₁·ln(x₁) + a₂·ln(x₂) + ... + aₙ·ln(xₙ)
    
    Define new variables:
        Y = ln(y)
        A₀ = ln(a₀)
        X₁ = ln(x₁)
        X₂ = ln(x₂)
        ...
        Xₙ = ln(xₙ)
    
    The transformed equation is linear:
        Y = A₀ + a₁·X₁ + a₂·X₂ + ... + aₙ·Xₙ
    
    This can now be solved using standard multiple linear regression.
    After finding A₀, recover a₀ = exp(A₀).
    
    NOTE: The other coefficients a₁, a₂, ..., aₙ are found directly
    from the linear regression and do not need transformation.
    """
    return explanation


def main() -> None:
    """Execute all Session 3 tasks and display results."""
    print("=" * 60)
    print("SESSION 3: POLYNOMIAL AND MULTIPLE REGRESSION")
    print("=" * 60)
    
    # Task 1: Extended Sum Calculations
    print("\n--- Task 1: Extended Sum Calculations ---\n")
    
    (sum_x, sum_y, sum_xy, mean_x, mean_y,
     intercept, slope, sum_x_squared) = task1_extended_sums()
    
    print("For xᵢ = i and yᵢ = 2i + 0.3, where i = 1, 2, ..., 10:")
    print(f"  1. Σxᵢ = {sum_x:.1f}")
    print(f"  2. Σyᵢ = {sum_y:.1f}")
    print(f"  3. Σxᵢyᵢ = {sum_xy:.1f}")
    print(f"  4. x̄ = {mean_x:.1f}, ȳ = {mean_y:.1f}")
    print(f"  5. a₀ = {intercept:.4f}, a₁ = {slope:.4f}")
    print(f"\n  Fitted model: y = {intercept:.4f} + {slope:.4f}x")
    print(f"  Expected (exact): y = 0.3 + 2.0x")
    
    # Plot Task 1 verification
    print("\n  Generating verification plot...")
    indices = np.arange(1, 11)
    x_data = indices.astype(float)
    y_data = 2 * indices + 0.3
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Data points', zorder=5)
    x_line = np.linspace(0, 11, 100)
    y_line = intercept + slope * x_line
    plt.plot(x_line, y_line, 'r-', label=f'Fit: y = {intercept:.2f} + {slope:.2f}x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Task 1: Linear Regression Verification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.close()
    
    # Task 2: Quadratic Regression Derivation
    print("\n--- Task 2: Quadratic Regression Derivation ---\n")
    print("For the quadratic model: y = a₀ + a₁x + a₂x²")
    print("\n(i) Partial derivatives:")
    print("    ∂fᵢ/∂a₀ = 1")
    print("    ∂fᵢ/∂a₁ = xᵢ")
    print("    ∂fᵢ/∂a₂ = xᵢ²")
    
    print("\n(ii) Setting ∂S/∂aₖ = 0 gives the normal equations.")
    
    print("\n(iii) The three normal equations are:")
    print("    n·a₀ + (Σxᵢ)·a₁ + (Σxᵢ²)·a₂ = Σyᵢ")
    print("    (Σxᵢ)·a₀ + (Σxᵢ²)·a₁ + (Σxᵢ³)·a₂ = Σxᵢyᵢ")
    print("    (Σxᵢ²)·a₀ + (Σxᵢ³)·a₁ + (Σxᵢ⁴)·a₂ = Σxᵢ²yᵢ")
    
    print("\n(iv) Matrix form: Ax = b where")
    print("    A = [n      Σx     Σx² ]")
    print("        [Σx     Σx²    Σx³ ]")
    print("        [Σx²    Σx³    Σx⁴ ]")
    
    # Example quadratic fit
    print("\n(v) Example quadratic fit:")
    x_quad = np.array([0, 1, 2, 3, 4, 5])
    y_quad = np.array([2.1, 7.7, 13.6, 27.2, 40.9, 61.1])
    params_quad = task2_quadratic_regression(x_quad, y_quad)
    print(f"    a₀ = {params_quad[0]:.4f}")
    print(f"    a₁ = {params_quad[1]:.4f}")
    print(f"    a₂ = {params_quad[2]:.4f}")
    print(f"    Model: y = {params_quad[0]:.2f} + {params_quad[1]:.2f}x + {params_quad[2]:.2f}x²")
    
    # Task 3: Multiple Linear Regression
    print("\n--- Task 3: Multiple Linear Regression ---\n")
    
    # Given data
    x1_data = np.array([0, 2, 2.5, 1, 4, 7])
    x2_data = np.array([0, 1, 2, 3, 6, 2])
    y_data = np.array([5, 10, 9, 0, 3, 27])
    
    parameters = task3_multiple_linear_regression(x1_data, x2_data, y_data)
    
    print("Given data:")
    print("  x₁ | x₂ | y")
    print("  ---|----|-")
    for data_index in range(len(y_data)):
        print(f"  {x1_data[data_index]:.1f} | {x2_data[data_index]:.1f} | {y_data[data_index]:.0f}")
    
    print(f"\nFitted parameters:")
    print(f"  a₀ = {parameters[0]:.4f}")
    print(f"  a₁ = {parameters[1]:.4f}")
    print(f"  a₂ = {parameters[2]:.4f}")
    print(f"\n  Model: y = {parameters[0]:.2f} + {parameters[1]:.2f}x₁ + {parameters[2]:.2f}x₂")
    
    # Plot 3D surface
    print("\n  Generating 3D plot...")
    plot_multiple_regression_3d(x1_data, x2_data, y_data, parameters)
    
    # Task 4: Linearisation
    print("\n--- Task 4: Linearisation ---")
    print(task4_linearisation_explanation())
    
    print("\n" + "=" * 60)
    print("SESSION 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
