"""Session 12: Practice Test Solutions.

This module provides solutions to the Practice Test questions covering:
- Linear regression with correlation coefficient (Q1)
- Linear regression assumptions discussion (Q2, Q3)
- Gaussian elimination for systems of equations (Q4)
- Root finding with Newton and Secant methods (Q5)
- ODE solving with finite differences (Q6)
- Fourier series coefficient calculation (Q7)

Author: William Fayers
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Create figures directory in the same folder as this script
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# =============================================================================
# Common Utilities
# =============================================================================

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
    
    # Forward elimination
    for pivot_row in range(num_equations):
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


# =============================================================================
# Question 1: Linear Regression with Correlation Coefficient
# =============================================================================

def question1_linear_regression() -> dict:
    """Perform linear least squares regression and compute correlation coefficient.
    
    Returns
    -------
    dict
        Results including regression parameters, correlation coefficient, and data.
    """
    # Given data
    x_data = np.array([0.000000, 1.428571, 2.857143, 4.285714,
                       5.714286, 7.142857, 8.571429, 10.000000])
    y_data = np.array([85.358681, 14.165804, -26.825715, -4.997952,
                       -94.922415, -164.033960, -180.876888, -206.688112])
    
    num_points = len(x_data)
    
    # Calculate sums
    sum_x = np.sum(x_data)
    sum_y = np.sum(y_data)
    sum_xy = np.sum(x_data * y_data)
    sum_x_squared = np.sum(x_data**2)
    sum_y_squared = np.sum(y_data**2)
    
    # Calculate means
    mean_x = sum_x / num_points
    mean_y = sum_y / num_points
    
    # Calculate regression coefficients
    # a1 = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    numerator_slope = num_points * sum_xy - sum_x * sum_y
    denominator_slope = num_points * sum_x_squared - sum_x**2
    slope = numerator_slope / denominator_slope
    
    # a0 = ȳ - a1*x̄
    intercept = mean_y - slope * mean_x
    
    # Calculate correlation coefficient r
    # r = (n*Σxy - Σx*Σy) / √[(n*Σx² - (Σx)²)(n*Σy² - (Σy)²)]
    numerator_r = num_points * sum_xy - sum_x * sum_y
    denominator_r = np.sqrt(
        (num_points * sum_x_squared - sum_x**2) *
        (num_points * sum_y_squared - sum_y**2)
    )
    correlation_coefficient = numerator_r / denominator_r
    
    # Calculate fitted values and residuals
    y_fitted = intercept + slope * x_data
    residuals = y_data - y_fitted
    
    # Calculate R² (coefficient of determination)
    ss_residual = np.sum(residuals**2)
    ss_total = np.sum((y_data - mean_y)**2)
    r_squared = 1 - ss_residual / ss_total
    
    return {
        'x_data': x_data,
        'y_data': y_data,
        'intercept': intercept,
        'slope': slope,
        'correlation_coefficient': correlation_coefficient,
        'r_squared': r_squared,
        'y_fitted': y_fitted,
        'residuals': residuals,
    }


def plot_question1(results: dict) -> None:
    """Plot the regression results for Question 1."""
    x_data = results['x_data']
    y_data = results['y_data']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Data and regression line
    axes[0].scatter(x_data, y_data, s=100, c='blue', zorder=5, label='Data')
    
    x_line = np.linspace(min(x_data) - 0.5, max(x_data) + 0.5, 100)
    y_line = results['intercept'] + results['slope'] * x_line
    axes[0].plot(x_line, y_line, 'r-', linewidth=2,
                 label=f"y = {results['intercept']:.2f} + {results['slope']:.2f}x")
    
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f"Linear Regression (r = {results['correlation_coefficient']:.3f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    axes[1].scatter(x_data, results['residuals'], s=100, c='green', zorder=5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'question1_regression.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {fig_path}")
    plt.close()


# =============================================================================
# Question 2 & 3: Regression Assumptions
# =============================================================================

def question2_false_statement() -> str:
    """Identify the FALSE statement about linear regression assumptions.
    
    Returns
    -------
    str
        The answer and explanation.
    """
    answer = """
    QUESTION 2: Which statement is FALSE?
    
    A) The y values are randomly distributed with the same variance.
       TRUE - This is the homoscedasticity assumption.
    
    B) The y values of the data are normally distributed.
       TRUE - This assumption allows for confidence intervals and hypothesis tests.
    
    C) All the data points are measured exactly.
       FALSE - This is the INCORRECT statement. We do NOT assume all data points
       are exact. Linear regression assumes the x values are known exactly, but
       the y values contain random error (noise). If all points were exact,
       there would be no need for regression.
    
    D) Each x value is precisely measured.
       TRUE - The independent variable x is assumed to be measured without error.
    
    ANSWER: C
    """
    return answer


def question3_model_checks() -> str:
    """Describe checks to confirm a model has been fitted correctly.
    
    Returns
    -------
    str
        Description of model validation checks.
    """
    checks = """
    QUESTION 3: Model Validation Checks
    
    After fitting a model to data, perform the following checks:
    
    1. VISUAL INSPECTION:
       - Plot the fitted model against the original data
       - Ensure the model captures the general trend
    
    2. RESIDUAL ANALYSIS:
       - Plot residuals vs fitted values (should show no pattern)
       - Plot residuals vs x values (should be randomly scattered)
       - Check for heteroscedasticity (unequal variance)
    
    3. NORMALITY OF RESIDUALS:
       - Create a histogram of residuals (should be approximately normal)
       - Use a Q-Q plot to check normality
    
    4. STATISTICAL MEASURES:
       - Check R² (coefficient of determination): how much variance is explained
       - Check correlation coefficient r: strength and direction of relationship
       - Calculate standard error of the regression
    
    5. OUTLIER DETECTION:
       - Identify points with unusually large residuals
       - Consider Cook's distance for influential points
    
    6. PARAMETER SIGNIFICANCE:
       - Check confidence intervals for parameters
       - Perform t-tests on regression coefficients
    
    7. CROSS-VALIDATION:
       - Test model on held-out data if available
       - Check prediction accuracy
    """
    return checks


# =============================================================================
# Question 4: System of Linear Equations
# =============================================================================

def question4_system_of_equations() -> dict:
    """Solve the system of linear equations and find sum of solutions.
    
    Returns
    -------
    dict
        Results including solution vector and sum.
    """
    # Given coefficient matrix
    coefficient_matrix = np.array([
        [9.88147746930662, 9.446294330943026, 6.867343600899789],
        [3.3013023006376, 8.866849207559063, 2.5855821434029],
        [7.897613140999123, 2.5810421630855136, 9.843121360475344]
    ])
    
    # Given right-hand side
    right_hand_side = np.array([
        1.7147395700538763,
        3.82744793002092,
        8.883242030765913
    ])
    
    # Solve using Gaussian elimination
    solution = gauss_eliminate(coefficient_matrix, right_hand_side)
    
    # Calculate sum of solutions
    sum_of_solutions = np.sum(solution)
    
    # Verify solution
    verification = coefficient_matrix @ solution
    
    return {
        'coefficient_matrix': coefficient_matrix,
        'right_hand_side': right_hand_side,
        'solution': solution,
        'sum_of_solutions': sum_of_solutions,
        'verification': verification,
    }


# =============================================================================
# Question 5: Root Finding
# =============================================================================

def newtons_method(
    func: callable,
    derivative: callable,
    initial_guess: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> dict:
    """Find a root using Newton's method.
    
    Parameters
    ----------
    func : callable
        Function f(x) to find root of.
    derivative : callable
        Derivative f'(x).
    initial_guess : float
        Starting point.
    tolerance : float
        Convergence tolerance.
    max_iterations : int
        Maximum iterations.
    
    Returns
    -------
    dict
        Results including root and iteration history.
    """
    current_x = initial_guess
    history = []
    
    for iteration in range(max_iterations):
        f_value = func(current_x)
        f_prime = derivative(current_x)
        
        history.append((current_x, f_value))
        
        if abs(f_value) < tolerance:
            return {
                'root': current_x,
                'iterations': iteration + 1,
                'history': history,
                'converged': True,
            }
        
        if abs(f_prime) < 1e-15:
            return {
                'root': current_x,
                'iterations': iteration + 1,
                'history': history,
                'converged': False,
            }
        
        current_x = current_x - f_value / f_prime
    
    return {
        'root': current_x,
        'iterations': max_iterations,
        'history': history,
        'converged': False,
    }


def secant_method(
    func: callable,
    x_prev: float,
    x_curr: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> dict:
    """Find a root using the Secant method.
    
    Parameters
    ----------
    func : callable
        Function f(x) to find root of.
    x_prev : float
        First starting point.
    x_curr : float
        Second starting point.
    tolerance : float
        Convergence tolerance.
    max_iterations : int
        Maximum iterations.
    
    Returns
    -------
    dict
        Results including root and iteration history.
    """
    f_prev = func(x_prev)
    f_curr = func(x_curr)
    history = [(x_prev, f_prev), (x_curr, f_curr)]
    
    for iteration in range(max_iterations):
        if abs(f_curr) < tolerance:
            return {
                'root': x_curr,
                'iterations': iteration + 2,
                'history': history,
                'converged': True,
            }
        
        denominator = f_curr - f_prev
        if abs(denominator) < 1e-15:
            return {
                'root': x_curr,
                'iterations': iteration + 2,
                'history': history,
                'converged': False,
            }
        
        x_next = x_curr - f_curr * (x_curr - x_prev) / denominator
        
        x_prev, f_prev = x_curr, f_curr
        x_curr = x_next
        f_curr = func(x_curr)
        history.append((x_curr, f_curr))
    
    return {
        'root': x_curr,
        'iterations': max_iterations + 2,
        'history': history,
        'converged': False,
    }


def question5_root_finding() -> dict:
    """Find roots of x² + 2.5x = 4.6875.
    
    Rearranged: x² + 2.5x - 4.6875 = 0
    
    Returns
    -------
    dict
        Results from Newton and Secant methods.
    """
    def equation(x):
        return x**2 + 2.5 * x - 4.6875
    
    def derivative(x):
        return 2 * x + 2.5
    
    # Using quadratic formula for exact solutions:
    # x = (-2.5 ± √(6.25 + 18.75)) / 2 = (-2.5 ± 5) / 2
    # x = 1.25 or x = -3.75
    
    results = {
        'newton_positive': newtons_method(equation, derivative, 2.0),
        'newton_negative': newtons_method(equation, derivative, -5.0),
        'secant_positive': secant_method(equation, 0.0, 2.0),
        'secant_negative': secant_method(equation, -5.0, -3.0),
        'exact_roots': [1.25, -3.75],
    }
    
    return results


def secant_method_explanation() -> str:
    """Explain the Secant method and its relationship to Newton's method.
    
    Returns
    -------
    str
        Explanation of the Secant method.
    """
    explanation = """
    SECANT METHOD EXPLANATION
    =========================
    
    Newton's method uses the iteration:
        xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)
    
    This requires knowing the derivative f'(x), which may be:
    - Difficult to compute analytically
    - Expensive to evaluate
    - Unknown (e.g., from experimental data)
    
    The Secant method APPROXIMATES the derivative using a finite difference:
        f'(xₙ) ≈ [f(xₙ) - f(xₙ₋₁)] / [xₙ - xₙ₋₁]
    
    Substituting this into Newton's formula gives the Secant iteration:
        xₙ₊₁ = xₙ - f(xₙ) × (xₙ - xₙ₋₁) / [f(xₙ) - f(xₙ₋₁)]
    
    KEY DIFFERENCES:
    
    1. INITIAL VALUES:
       - Newton: needs 1 starting point (x₀)
       - Secant: needs 2 starting points (x₀ and x₁)
    
    2. DERIVATIVE:
       - Newton: requires f'(x) explicitly
       - Secant: approximates f'(x) from function values
    
    3. CONVERGENCE RATE:
       - Newton: quadratic (order 2)
       - Secant: superlinear (order φ ≈ 1.618, the golden ratio)
    
    4. FUNCTION EVALUATIONS:
       - Newton: 2 per iteration (f and f')
       - Secant: 1 per iteration (just f)
    
    The Secant method is slower per iteration but may be faster overall
    when derivative evaluation is expensive.
    """
    return explanation


# =============================================================================
# Question 6: ODE with Finite Differences
# =============================================================================

def solve_second_order_ode_general(
    q_coefficient: float,
    k_coefficient: float,
    b_source: float,
    g_start: float,
    g_end: float,
    h_left: float,
    h_right: float,
    num_intervals: int,
) -> dict:
    """Solve q*d²H/dg² + k*H = b using finite differences.
    
    Parameters
    ----------
    q_coefficient : float
        Coefficient of d²H/dg².
    k_coefficient : float
        Coefficient of H.
    b_source : float
        Source term (constant).
    g_start : float
        Left boundary.
    g_end : float
        Right boundary.
    h_left : float
        Boundary condition H(g_start).
    h_right : float
        Boundary condition H(g_end).
    num_intervals : int
        Number of intervals for discretisation.
    
    Returns
    -------
    dict
        Results including g values, H values, and maximum H.
    
    Notes
    -----
    Discretisation:
        q*(Hᵢ₋₁ - 2Hᵢ + Hᵢ₊₁)/Δg² + k*Hᵢ = b
    
    Rearranging:
        q*Hᵢ₋₁ + (-2q + k*Δg²)*Hᵢ + q*Hᵢ₊₁ = b*Δg²
    """
    step_size = (g_end - g_start) / num_intervals
    num_interior = num_intervals - 1
    
    # Grid points
    g_values = np.linspace(g_start, g_end, num_intervals + 1)
    g_interior = g_values[1:-1]
    
    # Build coefficient matrix
    # Diagonal: -2q + k*Δg²
    # Off-diagonal: q
    diagonal_value = -2 * q_coefficient + k_coefficient * step_size**2
    off_diagonal = q_coefficient
    
    coefficient_matrix = np.zeros((num_interior, num_interior))
    np.fill_diagonal(coefficient_matrix, diagonal_value)
    
    for row_index in range(num_interior - 1):
        coefficient_matrix[row_index, row_index + 1] = off_diagonal
        coefficient_matrix[row_index + 1, row_index] = off_diagonal
    
    # Build right-hand side
    right_hand_side = b_source * step_size**2 * np.ones(num_interior)
    
    # Apply boundary conditions
    right_hand_side[0] -= q_coefficient * h_left
    right_hand_side[-1] -= q_coefficient * h_right
    
    # Solve
    h_interior = gauss_eliminate(coefficient_matrix, right_hand_side)
    
    # Assemble full solution
    h_values = np.zeros(num_intervals + 1)
    h_values[0] = h_left
    h_values[-1] = h_right
    h_values[1:-1] = h_interior
    
    return {
        'g_values': g_values,
        'h_values': h_values,
        'max_h': np.max(h_values),
        'step_size': step_size,
        'num_intervals': num_intervals,
    }


def question6_ode_solution() -> dict:
    """Solve the ODE from Question 6.
    
    q*d²H/dg² + k*H = b
    where g ∈ [0, 5], H_L = 5, H_R = 5, q = -3.0, k = 0, b = 6
    
    Returns
    -------
    dict
        Results for different grid sizes and k values.
    """
    # Parameters (from ODE: q*d²H/dg² + k*H = b)
    q_coefficient = -3.0
    k_coefficient = 0.0
    b_source = 6.0
    g_start = 0.0
    g_end = 5.0
    h_left = 5.0
    h_right = 5.0
    
    results = {'k_zero': {}}
    
    # Solve with different grid sizes
    for num_intervals in [10, 50, 100, 200]:
        results['k_zero'][num_intervals] = solve_second_order_ode_general(
            q_coefficient, k_coefficient, b_source, g_start, g_end, h_left, h_right, num_intervals
        )
    
    # Also solve with k = 5.538013
    k_coefficient_nonzero = 5.538013
    results['k_nonzero'] = {}
    for num_intervals in [10, 50, 100, 200]:
        results['k_nonzero'][num_intervals] = solve_second_order_ode_general(
            q_coefficient, k_coefficient_nonzero, b_source, g_start, g_end, h_left, h_right, num_intervals
        )
    
    return results


def plot_question6(results: dict) -> None:
    """Plot the ODE solutions for Question 6."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: k = 0
    ax1 = axes[0]
    colors = ['blue', 'green', 'orange', 'red']
    for i, num_intervals in enumerate([10, 50, 100, 200]):
        data = results['k_zero'][num_intervals]
        ax1.plot(data['g_values'], data['h_values'],
                 color=colors[i], linewidth=2, label=f'n = {num_intervals}')
    
    ax1.set_xlabel('g')
    ax1.set_ylabel('H(g)')
    ax1.set_title('Question 6: k = 0 (Parabolic Solution)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: k = 5.538013
    ax2 = axes[1]
    for i, num_intervals in enumerate([10, 50, 100, 200]):
        data = results['k_nonzero'][num_intervals]
        ax2.plot(data['g_values'], data['h_values'],
                 color=colors[i], linewidth=2, label=f'n = {num_intervals}')
    
    ax2.set_xlabel('g')
    ax2.set_ylabel('H(g)')
    ax2.set_title('Question 6: k = 5.538013 (Oscillatory Solution)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'question6_ode.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {fig_path}")
    plt.close()


def question6_nonzero_k_explanation() -> str:
    """Explain how to handle the case k ≠ 0.
    
    Returns
    -------
    str
        Explanation of the k ≠ 0 case.
    """
    explanation = """
    QUESTION 6: Handling k ≠ 0
    ==========================
    
    When k = 0, the equation is:
        q*d²H/dg² = b
    
    This is a simple Poisson equation with a parabolic solution.
    
    When k ≠ 0, the equation becomes:
        q*d²H/dg² + k*H = b
    
    This is a Helmholtz-type equation. The approach is similar:
    
    1. DISCRETISATION:
       Same finite difference scheme, but the coefficient matrix changes:
       
       q*(Hᵢ₋₁ - 2Hᵢ + Hᵢ₊₁)/Δg² + k*Hᵢ = b
       
       Rearranging:
       q*Hᵢ₋₁ + (-2q + k*Δg²)*Hᵢ + q*Hᵢ₊₁ = b*Δg²
    
    2. COEFFICIENT MATRIX:
       - Diagonal elements change from -2q to (-2q + k*Δg²)
       - Off-diagonal elements remain q
    
    3. SOLUTION BEHAVIOUR:
       - For q < 0 and k > 0: The solution may oscillate
       - The eigenvalues of the differential operator determine behaviour
       - With k = 5.538013, we may see more complex behaviour
    
    4. STABILITY CONSIDERATIONS:
       - The matrix must remain diagonally dominant for stability
       - Very large k may require finer grids
    
    The implementation remains the same; only the diagonal value changes.
    """
    return explanation


# =============================================================================
# Question 7: Fourier Coefficient Calculation
# =============================================================================

def question7_fourier_coefficient() -> dict:
    """Calculate a₁ for f(x) = 4(x - 1/2)² on [0, 1).
    
    Returns
    -------
    dict
        Results including a₁ and convergence analysis.
    """
    def target_function(x):
        return 4 * (x - 0.5)**2
    
    def phi_m(x, m, half_period):
        return np.cos(m * np.pi * x / half_period)
    
    half_period = 0.5  # Half-period for [0, 1)
    
    results = {}
    
    # Calculate a₁ with different numbers of grid points
    for num_points in [10, 50, 100, 500, 1000]:
        x_grid = np.linspace(0, 1, num_points, endpoint=False)
        dx = 1 / num_points
        
        # a_m = (1/l) * integral of phi_m(x) * f(x) dx
        # For m = 1: phi_1(x) = cos(πx/l) = cos(2πx) since l = 0.5
        
        integrand = phi_m(x_grid, 1, half_period) * target_function(x_grid)
        a1_numerical = (1 / half_period) * np.sum(integrand) * dx
        
        results[num_points] = a1_numerical
    
    # Analytical value for comparison
    # f(x) = 4(x - 1/2)² = 4x² - 4x + 1
    # a₁ = (2/1) ∫₀¹ cos(2πx) * 4(x-1/2)² dx
    # This can be computed analytically using integration by parts
    
    # Numerical integration with high accuracy
    x_fine = np.linspace(0, 1, 10000, endpoint=False)
    dx_fine = 1 / 10000
    integrand_fine = phi_m(x_fine, 1, half_period) * target_function(x_fine)
    a1_high_accuracy = (1 / half_period) * np.sum(integrand_fine) * dx_fine
    
    results['high_accuracy'] = a1_high_accuracy
    
    return results


def fourier_numerical_approximation_explanation() -> str:
    """Explain the numerical approximation for Fourier coefficients.
    
    Returns
    -------
    str
        Explanation of the numerical method.
    """
    explanation = """
    QUESTION 7: Numerical Approximation of Fourier Coefficients
    ============================================================
    
    The exact Fourier coefficient is defined as:
        aₘ = (1/l) ∫₋ₗˡ φₘ(x)f(x) dx
    
    where φₘ(x) = cos(mπx/l).
    
    NUMERICAL APPROXIMATION:
    
    We approximate the integral using a Riemann sum:
    
        ∫ₐᵇ g(x) dx ≈ Σᵢ g(xᵢ) Δx
    
    where:
    - xᵢ are regularly spaced points: xᵢ = a + i·Δx
    - Δx = (b - a)/N is the grid spacing
    - N is the number of grid points
    
    Therefore:
        aₘ ≈ (1/l) Σᵢ₌₀ᴺ⁻¹ φₘ(xᵢ)f(xᵢ) Δx
    
    IMPLEMENTATION DETAILS:
    
    1. Create a regular grid of N points: x₀, x₁, ..., xₙ₋₁
    2. Evaluate φₘ(xᵢ) = cos(mπxᵢ/l) at each point
    3. Evaluate f(xᵢ) at each point
    4. Compute the product φₘ(xᵢ)·f(xᵢ)
    5. Sum and multiply by Δx/l
    
    ACCURACY:
    
    - The Riemann sum has O(Δx) = O(1/N) error
    - Trapezoidal rule gives O(Δx²) = O(1/N²) error
    - Simpson's rule gives O(Δx⁴) = O(1/N⁴) error
    
    For better accuracy, use more sophisticated quadrature rules
    or increase the number of grid points.
    
    NOTE: The formula given uses endpoints=False to avoid double-counting
    at the boundary for periodic functions.
    """
    return explanation


# =============================================================================
# Main Function
# =============================================================================

def main() -> None:
    """Execute all Practice Test solutions and display results."""
    print("=" * 70)
    print("SESSION 12: PRACTICE TEST SOLUTIONS")
    print("=" * 70)
    
    # Question 1: Linear Regression
    print("\n" + "=" * 70)
    print("QUESTION 1: Linear Regression")
    print("=" * 70)
    
    q1_results = question1_linear_regression()
    
    print("\nGiven data:")
    print("      x      |      y")
    print("  -----------|-------------")
    for x, y in zip(q1_results['x_data'], q1_results['y_data']):
        print(f"  {x:10.6f} | {y:12.6f}")
    
    print(f"\nRegression Results:")
    print(f"  Intercept (a₀): {q1_results['intercept']:.6f}")
    print(f"  Slope (a₁): {q1_results['slope']:.6f}")
    print(f"  Model: y = {q1_results['intercept']:.4f} + {q1_results['slope']:.4f}x")
    
    print(f"\n  Correlation coefficient r = {q1_results['correlation_coefficient']:.3f}")
    print(f"  R² = {q1_results['r_squared']:.6f}")
    
    print("\nGenerating plot...")
    plot_question1(q1_results)
    
    # Question 2: False Statement
    print("\n" + "=" * 70)
    print("QUESTION 2: Linear Regression Assumptions")
    print("=" * 70)
    print(question2_false_statement())
    
    # Question 3: Model Checks
    print("\n" + "=" * 70)
    print("QUESTION 3: Model Validation")
    print("=" * 70)
    print(question3_model_checks())
    
    # Question 4: System of Equations
    print("\n" + "=" * 70)
    print("QUESTION 4: System of Linear Equations")
    print("=" * 70)
    
    q4_results = question4_system_of_equations()
    
    print("\nCoefficient matrix A:")
    print(q4_results['coefficient_matrix'])
    
    print("\nRight-hand side b:")
    print(q4_results['right_hand_side'])
    
    print("\nSolution vector x:")
    for i, x in enumerate(q4_results['solution']):
        print(f"  x{i} = {x:.10f}")
    
    print(f"\n  Sum of solutions: x₀ + x₁ + x₂ = {q4_results['sum_of_solutions']:.10f}")
    
    print("\nVerification (Ax should equal b):")
    print(f"  Ax = {q4_results['verification']}")
    print(f"  b  = {q4_results['right_hand_side']}")
    
    # Question 5: Root Finding
    print("\n" + "=" * 70)
    print("QUESTION 5: Root Finding")
    print("=" * 70)
    
    q5_results = question5_root_finding()
    
    print("\nEquation: x² + 2.5x = 4.6875")
    print("Rearranged: x² + 2.5x - 4.6875 = 0")
    
    print(f"\nExact roots (quadratic formula): {q5_results['exact_roots']}")
    
    print("\nNewton's Method Results:")
    print(f"  Positive root: {q5_results['newton_positive']['root']:.10f}")
    print(f"    Iterations: {q5_results['newton_positive']['iterations']}")
    print(f"  Negative root: {q5_results['newton_negative']['root']:.10f}")
    print(f"    Iterations: {q5_results['newton_negative']['iterations']}")
    
    print("\nSecant Method Results:")
    print(f"  Positive root: {q5_results['secant_positive']['root']:.10f}")
    print(f"    Iterations: {q5_results['secant_positive']['iterations']}")
    print(f"  Negative root: {q5_results['secant_negative']['root']:.10f}")
    print(f"    Iterations: {q5_results['secant_negative']['iterations']}")
    
    print(secant_method_explanation())
    
    # Question 6: ODE
    print("\n" + "=" * 70)
    print("QUESTION 6: ODE with Finite Differences")
    print("=" * 70)
    
    q6_results = question6_ode_solution()
    
    print("\nEquation: q·d²H/dg² + k·H = b")
    print("Parameters: q = -3.0, k = 0, b = 6, g ∈ [0, 5], H(0) = H(5) = 5")
    
    print("\nResults for k = 0:")
    print("  n intervals |  Max H(g)")
    print("  ------------|----------")
    for num_intervals, data in q6_results['k_zero'].items():
        print(f"      {num_intervals:4d}    | {data['max_h']:.6f}")
    
    print(f"\n  Answer: Maximum H(g) = {q6_results['k_zero'][200]['max_h']:.6f}")
    
    print("\nResults for k = 5.538013:")
    print("  n intervals |  Max H(g)")
    print("  ------------|----------")
    for num_intervals, data in q6_results['k_nonzero'].items():
        print(f"      {num_intervals:4d}    | {data['max_h']:.6f}")
    
    print("\nGenerating ODE solution plots...")
    plot_question6(q6_results)
    
    print(question6_nonzero_k_explanation())
    
    # Question 7: Fourier Coefficient
    print("\n" + "=" * 70)
    print("QUESTION 7: Fourier Coefficient Calculation")
    print("=" * 70)
    
    q7_results = question7_fourier_coefficient()
    
    print("\nFunction: f(x) = 4(x - 1/2)² on [0, 1)")
    print("Calculating a₁ (coefficient of cos(2πx)):")
    print()
    print("  N points |       a₁")
    print("  ---------|------------")
    for num_points, a1_value in q7_results.items():
        if num_points != 'high_accuracy':
            print(f"    {num_points:5d}  | {a1_value:.10f}")
    
    print(f"\n  High accuracy (10000 points): a₁ = {q7_results['high_accuracy']:.10f}")
    
    print(fourier_numerical_approximation_explanation())
    
    print("\n" + "=" * 70)
    print("PRACTICE TEST SOLUTIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
