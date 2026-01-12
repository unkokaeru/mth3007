"""Session 4: General Least Squares Solutions.

This module provides solutions to the Session 4 exercises covering:
- General Least Squares (GLS) with arbitrary basis functions
- Confidence intervals for fitted parameters
- Exponential basis function fitting

Author: William Fayers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def gauss_eliminate(
    coefficient_matrix: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Solve a system of linear equations using Gaussian elimination.
    
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
    """
    num_equations = len(right_hand_side)
    augmented = np.column_stack([
        coefficient_matrix.astype(float),
        right_hand_side.astype(float)
    ])
    
    # Forward elimination with partial pivoting
    for pivot_row in range(num_equations):
        max_row_index = pivot_row + np.argmax(
            np.abs(augmented[pivot_row:, pivot_row])
        )
        augmented[[pivot_row, max_row_index]] = augmented[[max_row_index, pivot_row]]
        
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


def general_least_squares(
    x_values: np.ndarray,
    y_values: np.ndarray,
    basis_functions: list,
) -> dict:
    """Fit data using General Least Squares with arbitrary basis functions.
    
    Fits the model: y = a₀f₀(x) + a₁f₁(x) + ... + aₘfₘ(x)
    
    Parameters
    ----------
    x_values : np.ndarray
        Independent variable values.
    y_values : np.ndarray
        Dependent variable values.
    basis_functions : list
        List of basis functions fₖ(x).
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'parameters': fitted coefficients
        - 'design_matrix': the Z matrix
        - 'covariance_matrix': (ZᵀZ)⁻¹
        - 'residual_variance': s²
        - 'r_squared': coefficient of determination
    
    Notes
    -----
    The normal equations are: ZᵀZ a = Zᵀy
    where Z[i,k] = fₖ(xᵢ)
    """
    num_points = len(x_values)
    num_params = len(basis_functions)
    
    # Build design matrix Z
    design_matrix = np.zeros((num_points, num_params))
    for param_index, basis_func in enumerate(basis_functions):
        design_matrix[:, param_index] = basis_func(x_values)
    
    # Form normal equations: ZᵀZ a = Zᵀy
    z_transpose_z = design_matrix.T @ design_matrix
    z_transpose_y = design_matrix.T @ y_values
    
    # Solve for parameters
    parameters = gauss_eliminate(z_transpose_z, z_transpose_y)
    
    # Calculate fitted values and residuals
    y_fitted = design_matrix @ parameters
    residuals = y_values - y_fitted
    
    # Calculate statistics
    sum_squared_residuals = np.sum(residuals**2)
    degrees_of_freedom = num_points - num_params
    residual_variance = sum_squared_residuals / degrees_of_freedom
    
    # Calculate R²
    y_mean = np.mean(y_values)
    total_sum_squares = np.sum((y_values - y_mean)**2)
    r_squared = 1 - sum_squared_residuals / total_sum_squares
    
    # Calculate covariance matrix
    covariance_matrix = np.linalg.inv(z_transpose_z) * residual_variance
    
    return {
        'parameters': parameters,
        'design_matrix': design_matrix,
        'covariance_matrix': covariance_matrix,
        'residual_variance': residual_variance,
        'r_squared': r_squared,
        'y_fitted': y_fitted,
        'residuals': residuals,
    }


def calculate_confidence_intervals(
    parameters: np.ndarray,
    covariance_matrix: np.ndarray,
    num_points: int,
    num_params: int,
    confidence_level: float = 0.95,
) -> dict:
    """Calculate confidence intervals for fitted parameters.
    
    Parameters
    ----------
    parameters : np.ndarray
        Fitted parameter values.
    covariance_matrix : np.ndarray
        Covariance matrix of the parameters.
    num_points : int
        Number of data points.
    num_params : int
        Number of fitted parameters.
    confidence_level : float, optional
        Confidence level (default 0.95 for 95%).
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'standard_errors': standard error for each parameter
        - 't_critical': critical t-value
        - 'lower_bounds': lower confidence bounds
        - 'upper_bounds': upper confidence bounds
    """
    degrees_of_freedom = num_points - num_params
    
    # Calculate standard errors from diagonal of covariance matrix
    standard_errors = np.sqrt(np.diag(covariance_matrix))
    
    # Calculate critical t-value
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)
    
    # Calculate confidence intervals
    margin_of_error = t_critical * standard_errors
    lower_bounds = parameters - margin_of_error
    upper_bounds = parameters + margin_of_error
    
    return {
        'standard_errors': standard_errors,
        't_critical': t_critical,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'degrees_of_freedom': degrees_of_freedom,
    }


def task1_exponential_basis_fit() -> dict:
    """Fit y = a₀ + a₁e⁻ˣ + a₂e⁻²ˣ to given data.
    
    Returns
    -------
    dict
        Results containing parameters and statistics.
    
    Notes
    -----
    Uses the basis functions:
    f₀(x) = 1
    f₁(x) = e⁻ˣ
    f₂(x) = e⁻²ˣ
    """
    # Given data
    x_values = np.array([0, 1, 2, 3, 4, 5])
    y_values = np.array([2.1, 7.7, 13.6, 27.2, 40.9, 61.1])
    
    # Define basis functions
    def basis_constant(x):
        return np.ones_like(x)
    
    def basis_exp_neg_x(x):
        return np.exp(-x)
    
    def basis_exp_neg_2x(x):
        return np.exp(-2 * x)
    
    basis_functions = [basis_constant, basis_exp_neg_x, basis_exp_neg_2x]
    
    # Fit using GLS
    results = general_least_squares(x_values, y_values, basis_functions)
    results['x_values'] = x_values
    results['y_values'] = y_values
    results['basis_functions'] = basis_functions
    
    return results


def task2_confidence_intervals(results: dict) -> dict:
    """Calculate 95% confidence intervals for the parameters.
    
    Parameters
    ----------
    results : dict
        Results from task1_exponential_basis_fit().
    
    Returns
    -------
    dict
        Confidence interval results.
    """
    num_points = len(results['y_values'])
    num_params = len(results['parameters'])
    
    confidence_results = calculate_confidence_intervals(
        results['parameters'],
        results['covariance_matrix'],
        num_points,
        num_params,
        confidence_level=0.95,
    )
    
    return confidence_results


def plot_exponential_fit(results: dict) -> None:
    """Plot the data and fitted exponential model.
    
    Parameters
    ----------
    results : dict
        Results from task1_exponential_basis_fit().
    """
    x_values = results['x_values']
    y_values = results['y_values']
    parameters = results['parameters']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data and fit
    axes[0].scatter(x_values, y_values, s=100, zorder=5, label='Data')
    
    x_smooth = np.linspace(0, 5, 100)
    y_smooth = (
        parameters[0] +
        parameters[1] * np.exp(-x_smooth) +
        parameters[2] * np.exp(-2 * x_smooth)
    )
    axes[0].plot(x_smooth, y_smooth, 'r-', linewidth=2, label='GLS Fit')
    
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Exponential Basis Function Fit')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = results['residuals']
    axes[1].scatter(x_values, residuals, s=100, zorder=5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()


def demonstrate_design_matrix_construction() -> None:
    """Show step-by-step construction of the design matrix Z."""
    x_values = np.array([0, 1, 2, 3, 4, 5])
    
    print("Design Matrix Z Construction")
    print("=" * 50)
    print("\nBasis functions:")
    print("  f₀(x) = 1")
    print("  f₁(x) = e⁻ˣ")
    print("  f₂(x) = e⁻²ˣ")
    
    print("\nFor each data point xᵢ, row i of Z is [f₀(xᵢ), f₁(xᵢ), f₂(xᵢ)]:")
    print()
    
    design_matrix = np.zeros((len(x_values), 3))
    design_matrix[:, 0] = 1
    design_matrix[:, 1] = np.exp(-x_values)
    design_matrix[:, 2] = np.exp(-2 * x_values)
    
    print("     x |   f₀(x)  |   f₁(x)   |   f₂(x)")
    print("  -----+----------+-----------+-----------")
    for i, x in enumerate(x_values):
        print(f"   {x:.0f}  |  {design_matrix[i, 0]:.4f}  |  {design_matrix[i, 1]:.6f}  |  {design_matrix[i, 2]:.6f}")
    
    print("\nFull design matrix Z:")
    print(design_matrix)


def main() -> None:
    """Execute all Session 4 tasks and display results."""
    print("=" * 60)
    print("SESSION 4: GENERAL LEAST SQUARES")
    print("=" * 60)
    
    # Demonstrate design matrix construction
    print("\n")
    demonstrate_design_matrix_construction()
    
    # Task 1: Exponential Basis Function Fit
    print("\n" + "=" * 60)
    print("Task 1: Fit y = a₀ + a₁e⁻ˣ + a₂e⁻²ˣ")
    print("=" * 60)
    
    results = task1_exponential_basis_fit()
    
    print("\nGiven data:")
    print("  x:", results['x_values'])
    print("  y:", results['y_values'])
    
    print(f"\nFitted parameters:")
    print(f"  a₀ = {results['parameters'][0]:.6f}")
    print(f"  a₁ = {results['parameters'][1]:.6f}")
    print(f"  a₂ = {results['parameters'][2]:.6f}")
    
    print(f"\nModel: y = {results['parameters'][0]:.4f} + "
          f"{results['parameters'][1]:.4f}e⁻ˣ + "
          f"{results['parameters'][2]:.4f}e⁻²ˣ")
    
    print(f"\nGoodness of fit:")
    print(f"  R² = {results['r_squared']:.6f}")
    print(f"  Residual variance s² = {results['residual_variance']:.6f}")
    
    print("\nCovariance matrix (ZᵀZ)⁻¹ · s²:")
    print(results['covariance_matrix'])
    
    # Task 2: Confidence Intervals
    print("\n" + "=" * 60)
    print("Task 2: 95% Confidence Intervals")
    print("=" * 60)
    
    confidence_results = task2_confidence_intervals(results)
    
    print(f"\nDegrees of freedom: n - m = 6 - 3 = {confidence_results['degrees_of_freedom']}")
    print(f"Critical t-value (α = 0.05, two-tailed): {confidence_results['t_critical']:.4f}")
    
    print("\nParameter confidence intervals:")
    for i, (param, std_err, lower, upper) in enumerate(zip(
        results['parameters'],
        confidence_results['standard_errors'],
        confidence_results['lower_bounds'],
        confidence_results['upper_bounds']
    )):
        print(f"\n  a{i}:")
        print(f"    Estimate: {param:.6f}")
        print(f"    Standard error: {std_err:.6f}")
        print(f"    95% CI: [{lower:.6f}, {upper:.6f}]")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_exponential_fit(results)
    
    # Summary of variance calculation
    print("\n" + "=" * 60)
    print("VARIANCE CALCULATION SUMMARY")
    print("=" * 60)
    print("""
    For GLS, the variance of the parameters is given by:
    
        Var(a) = (ZᵀZ)⁻¹ · s²
    
    where s² is the residual variance:
    
        s² = Σ(yᵢ - ŷᵢ)² / (n - m)
    
    The standard error of parameter aₖ is:
    
        SE(aₖ) = √(Var(aₖ)) = √((ZᵀZ)⁻¹ₖₖ · s²)
    
    The 95% confidence interval is:
    
        aₖ ± t_{α/2, n-m} · SE(aₖ)
    
    where t_{α/2, n-m} is the critical value from the t-distribution
    with (n - m) degrees of freedom.
    """)
    
    print("\n" + "=" * 60)
    print("SESSION 4 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
