"""Session 1: Computing Sums and Linear Regression Solutions.

This module provides solutions to the Session 1 exercises covering:
- Computing various mathematical sums
- Linear regression using the least squares method

Author: MTH3007 Numerical Methods
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_sum_2n_squared(upper_limit: int = 99) -> float:
    """Compute the sum of 2n² from n=0 to upper_limit.
    
    Parameters
    ----------
    upper_limit : int, optional
        The upper limit of the summation (inclusive), by default 99.
    
    Returns
    -------
    float
        The computed sum: Σ(2n²) for n from 0 to upper_limit.
    
    Examples
    --------
    >>> compute_sum_2n_squared(99)
    656700.0
    """
    indices = np.arange(0, upper_limit + 1)
    return float(np.sum(2 * indices**2))


def compute_sum_n(start: int = 1, end: int = 100) -> float:
    """Compute the sum of n from start to end.
    
    Parameters
    ----------
    start : int, optional
        The starting value of n, by default 1.
    end : int, optional
        The ending value of n (inclusive), by default 100.
    
    Returns
    -------
    float
        The computed sum: Σn for n from start to end.
    
    Examples
    --------
    >>> compute_sum_n(1, 100)
    5050.0
    """
    indices = np.arange(start, end + 1)
    return float(np.sum(indices))


def compute_sum_2n(start: int = 2, end: int = 200) -> float:
    """Compute the sum of 2n from start to end.
    
    Parameters
    ----------
    start : int, optional
        The starting value of n, by default 2.
    end : int, optional
        The ending value of n (inclusive), by default 200.
    
    Returns
    -------
    float
        The computed sum: Σ(2n) for n from start to end.
    
    Examples
    --------
    >>> compute_sum_2n(2, 200)
    40200.0
    """
    indices = np.arange(start, end + 1)
    return float(np.sum(2 * indices))


def compute_sum_2n_squared_range(start: int = 1, end: int = 200) -> float:
    """Compute the sum of 2n² from start to end.
    
    Parameters
    ----------
    start : int, optional
        The starting value of n, by default 1.
    end : int, optional
        The ending value of n (inclusive), by default 200.
    
    Returns
    -------
    float
        The computed sum: Σ(2n²) for n from start to end.
    
    Examples
    --------
    >>> compute_sum_2n_squared_range(1, 200)
    5373400.0
    """
    indices = np.arange(start, end + 1)
    return float(np.sum(2 * indices**2))


def calculate_linear_regression(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[float, float, dict]:
    """Calculate linear regression coefficients using least squares method.
    
    Fits the model y = a₀ + a₁x to the given data points using the
    normal equations derived from minimising the sum of squared errors.
    
    Parameters
    ----------
    x_values : np.ndarray
        Array of independent variable values.
    y_values : np.ndarray
        Array of dependent variable values.
    
    Returns
    -------
    tuple[float, float, dict]
        A tuple containing:
        - a0 : float - The y-intercept of the fitted line.
        - a1 : float - The slope of the fitted line.
        - statistics : dict - Dictionary containing intermediate calculations:
            - 'sum_x': Sum of x values
            - 'sum_y': Sum of y values
            - 'sum_xy': Sum of x*y products
            - 'sum_x_squared': Sum of x² values
            - 'mean_x': Mean of x values
            - 'mean_y': Mean of y values
            - 'n': Number of data points
    
    Notes
    -----
    The slope and intercept are calculated using:
    
    a₁ = (n·Σxᵢyᵢ - Σxᵢ·Σyᵢ) / (n·Σxᵢ² - (Σxᵢ)²)
    a₀ = ȳ - a₁·x̄
    
    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2.1, 4.0, 5.9, 8.1, 9.9])
    >>> a0, a1, stats = calculate_linear_regression(x, y)
    """
    num_points = len(x_values)
    
    # Calculate required sums
    sum_x = np.sum(x_values)
    sum_y = np.sum(y_values)
    sum_xy = np.sum(x_values * y_values)
    sum_x_squared = np.sum(x_values**2)
    
    # Calculate means
    mean_x = sum_x / num_points
    mean_y = sum_y / num_points
    
    # Calculate slope (a1) using the normal equation
    numerator = num_points * sum_xy - sum_x * sum_y
    denominator = num_points * sum_x_squared - sum_x**2
    slope = numerator / denominator
    
    # Calculate intercept (a0)
    intercept = mean_y - slope * mean_x
    
    # Collect statistics
    statistics = {
        'sum_x': sum_x,
        'sum_y': sum_y,
        'sum_xy': sum_xy,
        'sum_x_squared': sum_x_squared,
        'mean_x': mean_x,
        'mean_y': mean_y,
        'n': num_points,
    }
    
    return intercept, slope, statistics


def plot_linear_regression(
    x_values: np.ndarray,
    y_values: np.ndarray,
    intercept: float,
    slope: float,
    title: str = "Linear Regression",
    save_path: str | None = None,
) -> None:
    """Plot data points and fitted regression line.
    
    Parameters
    ----------
    x_values : np.ndarray
        Array of independent variable values.
    y_values : np.ndarray
        Array of dependent variable values.
    intercept : float
        The y-intercept (a₀) of the fitted line.
    slope : float
        The slope (a₁) of the fitted line.
    title : str, optional
        Title for the plot, by default "Linear Regression".
    save_path : str or None, optional
        Path to save the figure, by default None (displays instead).
    """
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(x_values, y_values, color='blue', label='Data points', zorder=5)
    
    # Plot regression line
    x_line = np.linspace(np.min(x_values) - 0.1, np.max(x_values) + 0.1, 100)
    y_line = intercept + slope * x_line
    plt.plot(
        x_line, y_line, color='red', linewidth=2,
        label=f'Fit: y = {intercept:.4f} + {slope:.4f}x'
    )
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def main() -> None:
    """Execute all Session 1 tasks and display results."""
    print("=" * 60)
    print("SESSION 1: COMPUTING SUMS AND LINEAR REGRESSION")
    print("=" * 60)
    
    # Task 1: Computing Sums
    print("\n--- Task 1: Computing Sums ---\n")
    
    result_1 = compute_sum_2n_squared(99)
    print(f"1. Σ(2n²) for n=0 to 99:  {result_1:,.0f}")
    
    result_2 = compute_sum_n(1, 100)
    print(f"2. Σn for n=1 to 100:     {result_2:,.0f}")
    
    result_3 = compute_sum_2n(2, 200)
    print(f"3. Σ(2n) for n=2 to 200:  {result_3:,.0f}")
    
    result_4 = compute_sum_2n_squared_range(1, 200)
    print(f"4. Σ(2n²) for n=1 to 200: {result_4:,.0f}")
    
    # Task 2: Linear Regression
    print("\n--- Task 2: Linear Regression ---\n")
    
    # Given data
    x_data = np.array([
        0.526993994, 0.691126852, 0.745407955, 0.669344512, 0.518168748,
        0.291558862, 0.010870453, 0.718185730, 0.897190954, 0.476789102,
    ])
    
    y_data = np.array([
        3.477982975, 4.197925374, 4.127080815, 3.365719179, 3.387060084,
        1.829099436, 0.658137249, 4.023164612, 5.074088869, 2.752890033,
    ])
    
    # Calculate regression
    intercept, slope, stats = calculate_linear_regression(x_data, y_data)
    
    print("Intermediate calculations:")
    print(f"  Σxᵢ = {stats['sum_x']:.6f}")
    print(f"  Σyᵢ = {stats['sum_y']:.6f}")
    print(f"  Σxᵢyᵢ = {stats['sum_xy']:.6f}")
    print(f"  Σxᵢ² = {stats['sum_x_squared']:.6f}")
    print(f"  x̄ = {stats['mean_x']:.6f}")
    print(f"  ȳ = {stats['mean_y']:.6f}")
    print(f"  n = {stats['n']}")
    
    print("\nRegression results:")
    print(f"  Intercept (a₀) = {intercept:.6f}")
    print(f"  Slope (a₁) = {slope:.6f}")
    print(f"\n  Fitted model: y = {intercept:.4f} + {slope:.4f}x")
    
    # Verify with numpy's polyfit
    numpy_coefficients = np.polyfit(x_data, y_data, 1)
    print("\nVerification with numpy.polyfit:")
    print(f"  Slope = {numpy_coefficients[0]:.6f}")
    print(f"  Intercept = {numpy_coefficients[1]:.6f}")
    
    # Plot results
    print("\nGenerating plot...")
    plot_linear_regression(
        x_data, y_data, intercept, slope,
        title="Session 1: Linear Regression"
    )
    
    print("\n" + "=" * 60)
    print("SESSION 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
