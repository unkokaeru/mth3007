"""MTH3007 Numerical Methods - Exam Reference Module.

A comprehensive collection of numerical methods covered in Semester A,
with detailed NumPy-style docstrings explaining the mathematical theory.

Topics covered:
1. Linear Algebra (Gaussian elimination, QR decomposition)
2. Regression & Curve Fitting (linear, polynomial, multiple, general least squares)
3. Interpolation (Newton, Lagrange, bilinear)
4. Root Finding & Optimisation (Newton's method, secant method)
5. Ordinary Differential Equations (finite difference methods)
6. Fourier Analysis (coefficients, Parseval's theorem)

Author: William Fayers
"""

import numpy as np
from scipy import stats


# =============================================================================
# 1. LINEAR ALGEBRA
# =============================================================================

def gauss_eliminate(
    coefficient_matrix: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Solve a system of linear equations Ax = b using Gaussian elimination.
    
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
    **Gaussian Elimination** converts a system Ax = b into upper triangular
    form through row operations, then solves by back substitution.
    
    **Algorithm:**
    
    1. **Forward Elimination:** For each pivot row k = 0, 1, ..., n-1:
       - For each row i > k below the pivot:
       - Compute multiplier: m = A[i,k] / A[k,k]
       - Subtract: Row_i ← Row_i - m × Row_k
    
    2. **Back Substitution:** Starting from the last equation:
       
       x[n-1] = b[n-1] / A[n-1, n-1]
       x[i] = (b[i] - Σ_{j>i} A[i,j]x[j]) / A[i,i]
    
    **Complexity:** O(n³) for forward elimination, O(n²) for back substitution.
    
    Examples
    --------
    >>> A = np.array([[2, 1], [1, 3]])
    >>> b = np.array([4, 5])
    >>> x = gauss_eliminate(A, b)
    >>> x  # Solution: x = [1, 2]
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


def qr_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute QR decomposition using Gram-Schmidt orthogonalisation.
    
    Parameters
    ----------
    matrix : np.ndarray
        Square matrix A to decompose, shape (n, n).
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Q : Orthogonal matrix (QᵀQ = I)
        R : Upper triangular matrix
        Such that A = QR.
    
    Notes
    -----
    **QR Decomposition** factors a matrix A into the product of an orthogonal
    matrix Q and an upper triangular matrix R.
    
    **Gram-Schmidt Process:**
    
    Given column vectors a₁, a₂, ..., aₙ of A:
    
    1. u₁ = a₁, q₁ = u₁/‖u₁‖
    2. For k = 2, ..., n:
       - uₖ = aₖ - Σⱼ₌₁ᵏ⁻¹ (aₖ·qⱼ)qⱼ  (subtract projections)
       - qₖ = uₖ/‖uₖ‖  (normalise)
    
    **Matrix R entries:**
    
    - R[j,k] = aₖ·qⱼ for j < k (projection coefficients)
    - R[k,k] = ‖uₖ‖ (norms)
    
    **Applications:**
    
    - Solving linear systems (more stable than Gaussian elimination)
    - Eigenvalue computation via QR algorithm
    - Least squares problems
    
    Examples
    --------
    >>> A = np.array([[1, 1], [0, 1], [1, 0]])
    >>> Q, R = qr_decomposition(A)
    >>> np.allclose(A, Q @ R)
    True
    """
    num_rows, num_cols = matrix.shape
    orthogonal_matrix = np.zeros((num_rows, num_cols))
    upper_triangular = np.zeros((num_cols, num_cols))
    
    for col_index in range(num_cols):
        # Start with original column
        orthogonal_vector = matrix[:, col_index].astype(float)
        
        # Subtract projections onto previous orthogonal vectors
        for prev_index in range(col_index):
            projection_coefficient = np.dot(
                matrix[:, col_index],
                orthogonal_matrix[:, prev_index]
            )
            upper_triangular[prev_index, col_index] = projection_coefficient
            orthogonal_vector -= projection_coefficient * orthogonal_matrix[:, prev_index]
        
        # Normalise
        vector_norm = np.linalg.norm(orthogonal_vector)
        upper_triangular[col_index, col_index] = vector_norm
        
        if vector_norm > 1e-10:
            orthogonal_matrix[:, col_index] = orthogonal_vector / vector_norm
        else:
            orthogonal_matrix[:, col_index] = orthogonal_vector
    
    return orthogonal_matrix, upper_triangular


def qr_algorithm_eigenvalues(
    matrix: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> dict:
    """Find eigenvalues using the QR algorithm.
    
    Parameters
    ----------
    matrix : np.ndarray
        Square matrix for which to find eigenvalues.
    max_iterations : int, optional
        Maximum number of QR iterations.
    tolerance : float, optional
        Convergence tolerance for off-diagonal elements.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'eigenvalues': array of eigenvalues
        - 'iterations': number of iterations performed
        - 'converged': whether the algorithm converged
        - 'final_matrix': the nearly-diagonal matrix
    
    Notes
    -----
    **QR Algorithm** iteratively computes eigenvalues by repeatedly
    factoring and recombining a matrix.
    
    **Algorithm:**
    
    1. Set A₀ = A
    2. For k = 0, 1, 2, ...:
       - Compute QR decomposition: Aₖ = QₖRₖ
       - Form new matrix: Aₖ₊₁ = RₖQₖ
    3. Continue until Aₖ converges to upper triangular form
    
    **Key Properties:**
    
    - All Aₖ are similar to A (same eigenvalues)
    - Under mild conditions, Aₖ → upper triangular
    - Diagonal entries of the limit are the eigenvalues
    
    **Convergence:**
    
    The algorithm converges when off-diagonal elements become negligible,
    measured by: √(Σᵢ≠ⱼ |Aᵢⱼ|²) < tolerance
    
    Examples
    --------
    >>> A = np.array([[2, 1], [1, 2]])
    >>> result = qr_algorithm_eigenvalues(A)
    >>> sorted(result['eigenvalues'])  # Eigenvalues: 1 and 3
    """
    current_matrix = matrix.astype(float).copy()
    num_rows = matrix.shape[0]
    
    for iteration in range(max_iterations):
        # QR decomposition
        orthogonal, upper_triangular = qr_decomposition(current_matrix)
        
        # Form RQ
        current_matrix = upper_triangular @ orthogonal
        
        # Check convergence: sum of squared off-diagonal elements
        off_diagonal_sum = 0.0
        for row in range(num_rows):
            for col in range(num_rows):
                if row != col:
                    off_diagonal_sum += current_matrix[row, col]**2
        
        if np.sqrt(off_diagonal_sum) < tolerance:
            return {
                'eigenvalues': np.diag(current_matrix),
                'iterations': iteration + 1,
                'converged': True,
                'final_matrix': current_matrix,
            }
    
    return {
        'eigenvalues': np.diag(current_matrix),
        'iterations': max_iterations,
        'converged': False,
        'final_matrix': current_matrix,
    }


# =============================================================================
# 2. REGRESSION & CURVE FITTING
# =============================================================================

def linear_regression(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> dict:
    """Fit a straight line y = a₀ + a₁x using least squares.
    
    Parameters
    ----------
    x_values : np.ndarray
        Independent variable values.
    y_values : np.ndarray
        Dependent variable values.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'intercept': a₀ (y-intercept)
        - 'slope': a₁ (gradient)
        - 'correlation': r (correlation coefficient)
        - 'r_squared': R² (coefficient of determination)
        - 'statistics': intermediate calculations
    
    Notes
    -----
    **Least Squares Method** minimises the sum of squared residuals:
    
        S = Σᵢ (yᵢ - a₀ - a₁xᵢ)²
    
    **Normal Equations** (from ∂S/∂a₀ = 0 and ∂S/∂a₁ = 0):
    
        a₁ = (n·Σxᵢyᵢ - Σxᵢ·Σyᵢ) / (n·Σxᵢ² - (Σxᵢ)²)
        a₀ = ȳ - a₁·x̄
    
    **Correlation Coefficient:**
    
        r = (n·Σxy - Σx·Σy) / √[(n·Σx² - (Σx)²)(n·Σy² - (Σy)²)]
    
    - r = 1: perfect positive correlation
    - r = -1: perfect negative correlation
    - r = 0: no linear correlation
    - R² = r²: proportion of variance explained
    
    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2.1, 4.0, 5.9, 8.1, 9.9])
    >>> result = linear_regression(x, y)
    >>> result['slope']  # Approximately 2.0
    """
    num_points = len(x_values)
    
    # Calculate sums
    sum_x = np.sum(x_values)
    sum_y = np.sum(y_values)
    sum_xy = np.sum(x_values * y_values)
    sum_x_squared = np.sum(x_values**2)
    sum_y_squared = np.sum(y_values**2)
    
    # Calculate means
    mean_x = sum_x / num_points
    mean_y = sum_y / num_points
    
    # Calculate slope
    numerator = num_points * sum_xy - sum_x * sum_y
    denominator = num_points * sum_x_squared - sum_x**2
    slope = numerator / denominator
    
    # Calculate intercept
    intercept = mean_y - slope * mean_x
    
    # Calculate correlation coefficient
    denominator_r = np.sqrt(
        (num_points * sum_x_squared - sum_x**2) *
        (num_points * sum_y_squared - sum_y**2)
    )
    correlation = numerator / denominator_r if denominator_r != 0 else 0.0
    
    return {
        'intercept': intercept,
        'slope': slope,
        'correlation': correlation,
        'r_squared': correlation**2,
        'statistics': {
            'sum_x': sum_x,
            'sum_y': sum_y,
            'sum_xy': sum_xy,
            'sum_x_squared': sum_x_squared,
            'mean_x': mean_x,
            'mean_y': mean_y,
            'n': num_points,
        },
    }


def polynomial_regression(
    x_values: np.ndarray,
    y_values: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Fit a polynomial y = a₀ + a₁x + a₂x² + ... + aₘxᵐ using least squares.
    
    Parameters
    ----------
    x_values : np.ndarray
        Independent variable values.
    y_values : np.ndarray
        Dependent variable values.
    degree : int
        Degree m of the polynomial.
    
    Returns
    -------
    np.ndarray
        Coefficients [a₀, a₁, ..., aₘ].
    
    Notes
    -----
    **Polynomial Regression** extends linear regression to higher-order terms.
    
    **Normal Equations** form a (m+1) × (m+1) system:
    
    For quadratic (m=2):
    
        [n      Σx     Σx²  ] [a₀]   [Σy    ]
        [Σx     Σx²    Σx³  ] [a₁] = [Σxy   ]
        [Σx²    Σx³    Σx⁴  ] [a₂]   [Σx²y  ]
    
    **General Pattern:**
    
    Matrix element [i,j] = Σxⁱ⁺ʲ
    Right-hand side [i] = Σxⁱy
    
    **Warning:** High-degree polynomials can overfit and become numerically
    unstable. Usually degree ≤ 5 is recommended.
    
    Examples
    --------
    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([1, 2, 5, 10])  # y ≈ 1 + x²
    >>> coeffs = polynomial_regression(x, y, degree=2)
    """
    num_params = degree + 1
    
    # Build coefficient matrix
    coefficient_matrix = np.zeros((num_params, num_params))
    for row in range(num_params):
        for col in range(num_params):
            coefficient_matrix[row, col] = np.sum(x_values**(row + col))
    
    # Build right-hand side
    right_hand_side = np.zeros(num_params)
    for row in range(num_params):
        right_hand_side[row] = np.sum(x_values**row * y_values)
    
    # Solve using Gaussian elimination
    return gauss_eliminate(coefficient_matrix, right_hand_side)


def multiple_linear_regression(
    x_matrix: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    """Fit y = a₀ + a₁x₁ + a₂x₂ + ... using least squares.
    
    Parameters
    ----------
    x_matrix : np.ndarray
        Matrix of independent variables, shape (n, p) where n is number
        of observations and p is number of predictors.
    y_values : np.ndarray
        Dependent variable values, shape (n,).
    
    Returns
    -------
    np.ndarray
        Coefficients [a₀, a₁, a₂, ...].
    
    Notes
    -----
    **Multiple Linear Regression** models the dependent variable as a
    linear combination of multiple independent variables.
    
    **Model:** y = a₀ + a₁x₁ + a₂x₂ + ... + aₚxₚ
    
    **Normal Equations** for two predictors:
    
        [n      Σx₁     Σx₂   ] [a₀]   [Σy     ]
        [Σx₁    Σx₁²    Σx₁x₂ ] [a₁] = [Σx₁y   ]
        [Σx₂    Σx₁x₂   Σx₂²  ] [a₂]   [Σx₂y   ]
    
    **Matrix Form:** Using design matrix Z with Z[:,0] = 1 (intercept):
    
        ZᵀZ·a = Zᵀy
    
    Examples
    --------
    >>> x1 = np.array([1, 2, 3, 4])
    >>> x2 = np.array([2, 4, 6, 8])
    >>> y = np.array([5, 10, 15, 20])
    >>> X = np.column_stack([x1, x2])
    >>> coeffs = multiple_linear_regression(X, y)
    """
    num_points = len(y_values)
    num_predictors = x_matrix.shape[1]
    
    # Build design matrix with intercept column
    design_matrix = np.column_stack([np.ones(num_points), x_matrix])
    
    # Form normal equations: ZᵀZ a = Zᵀy
    z_transpose_z = design_matrix.T @ design_matrix
    z_transpose_y = design_matrix.T @ y_values
    
    return gauss_eliminate(z_transpose_z, z_transpose_y)


def general_least_squares(
    x_values: np.ndarray,
    y_values: np.ndarray,
    basis_functions: list,
) -> dict:
    """Fit data using arbitrary basis functions.
    
    Parameters
    ----------
    x_values : np.ndarray
        Independent variable values.
    y_values : np.ndarray
        Dependent variable values.
    basis_functions : list
        List of basis functions [f₀, f₁, ..., fₘ] where each fₖ(x) → array.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'parameters': fitted coefficients [a₀, a₁, ..., aₘ]
        - 'design_matrix': the Z matrix
        - 'covariance_matrix': (ZᵀZ)⁻¹ × s²
        - 'residual_variance': s²
        - 'r_squared': coefficient of determination
    
    Notes
    -----
    **General Least Squares (GLS)** fits the model:
    
        y = a₀f₀(x) + a₁f₁(x) + ... + aₘfₘ(x)
    
    **Design Matrix Z:**
    
        Z[i,k] = fₖ(xᵢ)
    
    **Normal Equations:**
    
        ZᵀZ·a = Zᵀy
    
    **Residual Variance:**
    
        s² = Σ(yᵢ - ŷᵢ)² / (n - m - 1)
    
    **Covariance Matrix:**
    
        Cov(a) = s² × (ZᵀZ)⁻¹
    
    **Common Basis Function Choices:**
    
    - Polynomial: f₀=1, f₁=x, f₂=x², ...
    - Exponential: f₀=1, f₁=e⁻ˣ, f₂=e⁻²ˣ, ...
    - Trigonometric: f₀=1, f₁=cos(x), f₂=sin(x), ...
    
    Examples
    --------
    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([1.0, 2.7, 7.4, 20.1])
    >>> basis = [lambda x: np.ones_like(x), lambda x: np.exp(x)]
    >>> result = general_least_squares(x, y, basis)
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


def confidence_intervals(
    parameters: np.ndarray,
    covariance_matrix: np.ndarray,
    num_points: int,
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
    
    Notes
    -----
    **Confidence Intervals** quantify uncertainty in fitted parameters.
    
    **Standard Error:**
    
        SE(aₖ) = √(Cov(a)[k,k])
    
    **Confidence Interval:**
    
        aₖ ± t_{α/2, ν} × SE(aₖ)
    
    where:
    - t_{α/2, ν} is the critical t-value
    - ν = n - m - 1 is the degrees of freedom
    - α = 1 - confidence_level
    
    **Interpretation:** We are (confidence_level × 100)% confident that
    the true parameter value lies within the interval.
    
    Examples
    --------
    >>> params = np.array([2.0, 1.5])
    >>> cov = np.array([[0.01, 0], [0, 0.02]])
    >>> ci = confidence_intervals(params, cov, num_points=20)
    """
    num_params = len(parameters)
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


# =============================================================================
# 3. INTERPOLATION
# =============================================================================

def divided_differences(
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
        Row 0 contains the coefficients for Newton's polynomial.
    
    Notes
    -----
    **Divided Differences** are defined recursively:
    
    **Zeroth order:**
        f[xᵢ] = yᵢ
    
    **Higher orders:**
        f[xᵢ, xᵢ₊₁, ..., xᵢ₊ₖ] = (f[xᵢ₊₁, ..., xᵢ₊ₖ] - f[xᵢ, ..., xᵢ₊ₖ₋₁]) / (xᵢ₊ₖ - xᵢ)
    
    **Table Structure:**
    
        x₀  f[x₀]   f[x₀,x₁]   f[x₀,x₁,x₂]   ...
        x₁  f[x₁]   f[x₁,x₂]   f[x₁,x₂,x₃]   ...
        x₂  f[x₂]   f[x₂,x₃]   ...
        ...
    
    The first row contains Newton polynomial coefficients.
    
    Examples
    --------
    >>> x = np.array([0, 1, 3])
    >>> y = np.array([1, 3, 55])
    >>> table = divided_differences(x, y)
    >>> table[0, :]  # Newton coefficients: [1, 2, 8]
    """
    num_points = len(x_values)
    table = np.zeros((num_points, num_points))
    
    # First column: f[xᵢ] = yᵢ
    table[:, 0] = y_values
    
    # Fill subsequent columns
    for column in range(1, num_points):
        for row in range(num_points - column):
            numerator = table[row + 1, column - 1] - table[row, column - 1]
            denominator = x_values[row + column] - x_values[row]
            table[row, column] = numerator / denominator
    
    return table


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
    **Newton's Form** of the interpolating polynomial:
    
        P(x) = f[x₀] + f[x₀,x₁](x-x₀) + f[x₀,x₁,x₂](x-x₀)(x-x₁) + ...
    
    **Evaluation Algorithm:**
    
    1. Compute divided differences table
    2. Accumulate terms iteratively:
       - Start with result = f[x₀]
       - For each term k: multiply (x - xₖ₋₁) into product, add f[x₀,...,xₖ] × product
    
    **Advantages over Lagrange:**
    
    - Easy to add new data points (just add new column to table)
    - More efficient evaluation
    
    Examples
    --------
    >>> x = np.array([0, 1, 3])
    >>> y = np.array([1, 3, 55])
    >>> newton_interpolation(x, y, 2.0)  # Interpolate at x=2
    """
    table = divided_differences(x_values, y_values)
    num_points = len(x_values)
    
    # Evaluate Newton's polynomial
    result = table[0, 0]
    product_term = np.ones_like(x_evaluate, dtype=float)
    
    for term_index in range(1, num_points):
        product_term = product_term * (x_evaluate - x_values[term_index - 1])
        result = result + table[0, term_index] * product_term
    
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
    **Lagrange's Form** of the interpolating polynomial:
    
        P(x) = Σₖ yₖ Lₖ(x)
    
    **Lagrange Basis Polynomials:**
    
        Lₖ(x) = ∏_{j≠k} (x - xⱼ) / (xₖ - xⱼ)
    
    **Properties of Lₖ:**
    
    - Lₖ(xⱼ) = 1 if j = k, 0 otherwise
    - Each Lₖ is a polynomial of degree n-1
    
    **Comparison with Newton:**
    
    - Same polynomial, different form
    - Lagrange: conceptually simpler
    - Newton: easier to extend with new points
    
    Examples
    --------
    >>> x = np.array([0, 1, 3])
    >>> y = np.array([1, 3, 55])
    >>> lagrange_interpolation(x, y, 2.0)  # Same result as Newton
    """
    num_points = len(x_values)
    result = np.zeros_like(x_evaluate, dtype=float)
    
    for point_index in range(num_points):
        # Compute Lagrange basis polynomial Lₖ(x)
        basis = np.ones_like(x_evaluate, dtype=float)
        for other_index in range(num_points):
            if other_index != point_index:
                basis *= (x_evaluate - x_values[other_index]) / (x_values[point_index] - x_values[other_index])
        
        result += y_values[point_index] * basis
    
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
        2D array of z values, shape (len(y_grid), len(x_grid)).
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
    **Bilinear Interpolation** extends linear interpolation to 2D.
    
    **Algorithm:**
    
    1. Find the bounding cell: (x₁, y₁), (x₂, y₁), (x₁, y₂), (x₂, y₂)
    2. Compute normalised coordinates:
       - t = (x - x₁) / (x₂ - x₁)
       - s = (y - y₁) / (y₂ - y₁)
    
    3. Interpolate:
       z ≈ (1-t)(1-s)z₁₁ + t(1-s)z₂₁ + (1-t)s·z₁₂ + ts·z₂₂
    
    **Geometric Interpretation:**
    
    - First interpolate in x direction at both y₁ and y₂
    - Then interpolate in y direction between those results
    
    Examples
    --------
    >>> x = np.array([0, 1, 2])
    >>> y = np.array([0, 1, 2])
    >>> z = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    >>> bilinear_interpolation(x, y, z, 0.5, 0.5)
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


# =============================================================================
# 4. ROOT FINDING & OPTIMISATION
# =============================================================================

def newtons_method_root(
    func: callable,
    derivative: callable,
    initial_guess: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> dict:
    """Find a root of f(x) = 0 using Newton's method.
    
    Parameters
    ----------
    func : callable
        The function f(x) for which to find the root.
    derivative : callable
        The derivative f'(x).
    initial_guess : float
        Starting point x₀.
    tolerance : float, optional
        Convergence tolerance for |f(x)|.
    max_iterations : int, optional
        Maximum number of iterations.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'root': the found root
        - 'iterations': number of iterations used
        - 'history': list of (x, f(x), f'(x)) tuples
        - 'converged': whether the method converged
    
    Notes
    -----
    **Newton's Method** uses tangent line approximation to find roots.
    
    **Iteration Formula:**
    
        xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)
    
    **Derivation:**
    
    1. Taylor expansion: f(x) ≈ f(xₙ) + f'(xₙ)(x - xₙ)
    2. Set f(x) = 0 and solve for x
    3. This gives the iteration formula
    
    **Convergence:**
    
    - Quadratic convergence near a simple root
    - Error at step n+1 ≈ C × (error at step n)²
    - May fail if f'(x) = 0 or initial guess is poor
    
    **Example: Finding √a**
    
    To find √a, solve x² - a = 0:
    - f(x) = x² - a, f'(x) = 2x
    - xₙ₊₁ = xₙ - (xₙ² - a)/(2xₙ) = (xₙ + a/xₙ)/2
    
    Examples
    --------
    >>> def f(x): return x**2 - 2
    >>> def df(x): return 2*x
    >>> result = newtons_method_root(f, df, initial_guess=1.5)
    >>> result['root']  # √2 ≈ 1.4142
    """
    current_x = initial_guess
    history = []
    
    for iteration in range(max_iterations):
        function_value = func(current_x)
        derivative_value = derivative(current_x)
        
        history.append((current_x, function_value, derivative_value))
        
        if abs(function_value) < tolerance:
            return {
                'root': current_x,
                'iterations': iteration + 1,
                'history': history,
                'converged': True,
            }
        
        if abs(derivative_value) < 1e-15:
            return {
                'root': current_x,
                'iterations': iteration + 1,
                'history': history,
                'converged': False,
                'error': 'Derivative too small',
            }
        
        # Newton's update
        current_x = current_x - function_value / derivative_value
    
    return {
        'root': current_x,
        'iterations': max_iterations,
        'history': history,
        'converged': False,
        'error': 'Maximum iterations exceeded',
    }


def secant_method(
    func: callable,
    x_previous: float,
    x_current: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> dict:
    """Find a root of f(x) = 0 using the secant method.
    
    Parameters
    ----------
    func : callable
        The function f(x) for which to find the root.
    x_previous : float
        First starting point x₀.
    x_current : float
        Second starting point x₁.
    tolerance : float, optional
        Convergence tolerance for |f(x)|.
    max_iterations : int, optional
        Maximum number of iterations.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'root': the found root
        - 'iterations': number of iterations used
        - 'history': list of (x, f(x)) tuples
        - 'converged': whether the method converged
    
    Notes
    -----
    **Secant Method** approximates the derivative using finite differences.
    
    **Iteration Formula:**
    
        xₙ₊₁ = xₙ - f(xₙ) × (xₙ - xₙ₋₁) / (f(xₙ) - f(xₙ₋₁))
    
    **Comparison with Newton:**
    
    - Does NOT require computing f'(x)
    - Uses approximation: f'(xₙ) ≈ (f(xₙ) - f(xₙ₋₁)) / (xₙ - xₙ₋₁)
    - Superlinear convergence: order ≈ 1.618 (golden ratio)
    - Needs two starting points instead of one
    
    **When to Use:**
    
    - When f'(x) is difficult or expensive to compute
    - When two good initial estimates are available
    
    Examples
    --------
    >>> def f(x): return x**2 - 2
    >>> result = secant_method(f, x_previous=1.0, x_current=2.0)
    >>> result['root']  # √2 ≈ 1.4142
    """
    f_previous = func(x_previous)
    f_current = func(x_current)
    history = [(x_previous, f_previous), (x_current, f_current)]
    
    for iteration in range(max_iterations):
        if abs(f_current) < tolerance:
            return {
                'root': x_current,
                'iterations': iteration + 2,
                'history': history,
                'converged': True,
            }
        
        denominator = f_current - f_previous
        if abs(denominator) < 1e-15:
            return {
                'root': x_current,
                'iterations': iteration + 2,
                'history': history,
                'converged': False,
                'error': 'Function values too close',
            }
        
        # Secant update
        x_next = x_current - f_current * (x_current - x_previous) / denominator
        
        # Update for next iteration
        x_previous, f_previous = x_current, f_current
        x_current = x_next
        f_current = func(x_current)
        history.append((x_current, f_current))
    
    return {
        'root': x_current,
        'iterations': max_iterations + 2,
        'history': history,
        'converged': False,
        'error': 'Maximum iterations exceeded',
    }


def newtons_method_optimisation(
    func: callable,
    first_derivative: callable,
    second_derivative: callable,
    initial_guess: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> dict:
    """Find a stationary point of f(x) using Newton's method.
    
    Parameters
    ----------
    func : callable
        The function f(x) to optimise.
    first_derivative : callable
        The first derivative f'(x).
    second_derivative : callable
        The second derivative f''(x).
    initial_guess : float
        Starting point x₀.
    tolerance : float, optional
        Convergence tolerance for |f'(x)|.
    max_iterations : int, optional
        Maximum number of iterations.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'stationary_point': the found stationary point
        - 'function_value': f(x) at the stationary point
        - 'second_derivative': f''(x) at the stationary point
        - 'type': 'minimum', 'maximum', or 'inflection'
        - 'iterations': number of iterations used
        - 'converged': whether the method converged
    
    Notes
    -----
    **Optimisation via Root Finding**
    
    To find stationary points, we solve f'(x) = 0 using Newton's method
    applied to g(x) = f'(x):
    
        xₙ₊₁ = xₙ - f'(xₙ)/f''(xₙ)
    
    **Classifying Stationary Points:**
    
    At a stationary point x* where f'(x*) = 0:
    - f''(x*) > 0 → local minimum
    - f''(x*) < 0 → local maximum
    - f''(x*) = 0 → inflection point (need higher derivatives)
    
    **Convergence:**
    
    - Same quadratic convergence as Newton for root finding
    - Requires f''(x) ≠ 0 near the stationary point
    
    Examples
    --------
    >>> def f(x): return (x - 2)**2 + 1
    >>> def df(x): return 2*(x - 2)
    >>> def d2f(x): return 2
    >>> result = newtons_method_optimisation(f, df, d2f, initial_guess=0)
    >>> result['stationary_point']  # x = 2 (minimum)
    """
    current_x = initial_guess
    history = []
    
    for iteration in range(max_iterations):
        f_value = func(current_x)
        f_prime = first_derivative(current_x)
        f_double_prime = second_derivative(current_x)
        
        history.append((current_x, f_value, f_prime, f_double_prime))
        
        if abs(f_prime) < tolerance:
            # Determine type of stationary point
            if f_double_prime > tolerance:
                point_type = 'minimum'
            elif f_double_prime < -tolerance:
                point_type = 'maximum'
            else:
                point_type = 'inflection'
            
            return {
                'stationary_point': current_x,
                'function_value': f_value,
                'second_derivative': f_double_prime,
                'type': point_type,
                'iterations': iteration + 1,
                'history': history,
                'converged': True,
            }
        
        if abs(f_double_prime) < 1e-15:
            return {
                'stationary_point': current_x,
                'function_value': f_value,
                'iterations': iteration + 1,
                'history': history,
                'converged': False,
                'error': 'Second derivative too small',
            }
        
        # Newton's update for optimisation
        current_x = current_x - f_prime / f_double_prime
    
    return {
        'stationary_point': current_x,
        'function_value': func(current_x),
        'iterations': max_iterations,
        'history': history,
        'converged': False,
        'error': 'Maximum iterations exceeded',
    }


# =============================================================================
# 5. ORDINARY DIFFERENTIAL EQUATIONS
# =============================================================================

def solve_second_order_ode(
    source_function: callable,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    num_intervals: int,
) -> dict:
    """Solve d²y/dx² = g(x) with Dirichlet boundary conditions.
    
    Parameters
    ----------
    source_function : callable
        The source function g(x) on the right-hand side.
    x_start : float
        Left boundary x = a.
    x_end : float
        Right boundary x = b.
    y_start : float
        Boundary condition y(a).
    y_end : float
        Boundary condition y(b).
    num_intervals : int
        Number of intervals (n).
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'x_values': grid points including boundaries
        - 'y_values': solution at all grid points
        - 'step_size': the step size h
        - 'coefficient_matrix': the tridiagonal matrix
    
    Notes
    -----
    **Finite Difference Method** approximates derivatives using grid values.
    
    **Central Difference Approximation:**
    
        d²y/dx² ≈ (yᵢ₋₁ - 2yᵢ + yᵢ₊₁) / h²
    
    **Discretisation:**
    
    At each interior point xᵢ:
        (yᵢ₋₁ - 2yᵢ + yᵢ₊₁) / h² = g(xᵢ)
    
    Rearranging:
        yᵢ₋₁ - 2yᵢ + yᵢ₊₁ = h²g(xᵢ)
    
    **Linear System:**
    
    For interior points i = 1, 2, ..., n-1:
    
        [-2  1  0  0 ...] [y₁]   [h²g(x₁) - y₀    ]
        [ 1 -2  1  0 ...] [y₂]   [h²g(x₂)         ]
        [ 0  1 -2  1 ...] [y₃] = [h²g(x₃)         ]
        [... ... ...    ] [...]   [...              ]
        [... ... 1 -2   ] [yₙ₋₁] [h²g(xₙ₋₁) - yₙ  ]
    
    **Error:** O(h²) - second-order accurate
    
    Examples
    --------
    >>> # Solve y'' = -2 on [0,1] with y(0)=0, y(1)=0
    >>> # Exact solution: y = x(1-x)
    >>> result = solve_second_order_ode(
    ...     lambda x: -2*np.ones_like(x), 0, 1, 0, 0, num_intervals=10
    ... )
    """
    step_size = (x_end - x_start) / num_intervals
    num_interior = num_intervals - 1
    
    # Create grid points (including boundaries)
    x_values = np.linspace(x_start, x_end, num_intervals + 1)
    
    # Interior points
    x_interior = x_values[1:-1]
    
    # Build tridiagonal coefficient matrix
    coefficient_matrix = np.zeros((num_interior, num_interior))
    
    # Fill diagonal
    np.fill_diagonal(coefficient_matrix, -2)
    
    # Fill off-diagonals
    for row_index in range(num_interior - 1):
        coefficient_matrix[row_index, row_index + 1] = 1
        coefficient_matrix[row_index + 1, row_index] = 1
    
    # Build right-hand side vector
    right_hand_side = step_size**2 * source_function(x_interior)
    
    # Apply boundary conditions
    right_hand_side[0] -= y_start
    right_hand_side[-1] -= y_end
    
    # Solve the system
    y_interior = gauss_eliminate(coefficient_matrix, right_hand_side)
    
    # Assemble complete solution including boundaries
    y_values = np.zeros(num_intervals + 1)
    y_values[0] = y_start
    y_values[-1] = y_end
    y_values[1:-1] = y_interior
    
    return {
        'x_values': x_values,
        'y_values': y_values,
        'step_size': step_size,
        'coefficient_matrix': coefficient_matrix,
        'right_hand_side': right_hand_side,
    }


def solve_ode_with_y_term(
    source_function: callable,
    coefficient_k: float,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    num_intervals: int,
) -> dict:
    """Solve d²y/dx² + k·y = g(x) with Dirichlet boundary conditions.
    
    Parameters
    ----------
    source_function : callable
        The source function g(x) on the right-hand side.
    coefficient_k : float
        Coefficient of the y term.
    x_start : float
        Left boundary x = a.
    x_end : float
        Right boundary x = b.
    y_start : float
        Boundary condition y(a).
    y_end : float
        Boundary condition y(b).
    num_intervals : int
        Number of intervals (n).
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'x_values': grid points including boundaries
        - 'y_values': solution at all grid points
        - 'step_size': the step size h
    
    Notes
    -----
    **Extended Finite Difference Method** for ODEs with y term.
    
    **Discretisation:**
    
        (yᵢ₋₁ - 2yᵢ + yᵢ₊₁)/h² + k·yᵢ = g(xᵢ)
    
    Rearranging:
        yᵢ₋₁ + (-2 + k·h²)yᵢ + yᵢ₊₁ = h²g(xᵢ)
    
    **Tridiagonal System:**
    
    The diagonal entries become (-2 + k·h²) instead of -2.
    
    **Example:** y'' - y = 0 (k = -1) with y(0)=0, y(1)=sinh(1)
    has exact solution y = sinh(x).
    
    Examples
    --------
    >>> result = solve_ode_with_y_term(
    ...     lambda x: np.zeros_like(x),  # g(x) = 0
    ...     coefficient_k=-1,  # y'' - y = 0
    ...     x_start=0, x_end=1,
    ...     y_start=0, y_end=np.sinh(1),
    ...     num_intervals=10
    ... )
    """
    step_size = (x_end - x_start) / num_intervals
    num_interior = num_intervals - 1
    
    # Create grid points
    x_values = np.linspace(x_start, x_end, num_intervals + 1)
    x_interior = x_values[1:-1]
    
    # Build coefficient matrix
    # Diagonal element: -2 + k*h²
    diagonal_value = -2 + coefficient_k * step_size**2
    
    coefficient_matrix = np.zeros((num_interior, num_interior))
    np.fill_diagonal(coefficient_matrix, diagonal_value)
    
    for row_index in range(num_interior - 1):
        coefficient_matrix[row_index, row_index + 1] = 1
        coefficient_matrix[row_index + 1, row_index] = 1
    
    # Build right-hand side
    right_hand_side = step_size**2 * source_function(x_interior)
    right_hand_side[0] -= y_start
    right_hand_side[-1] -= y_end
    
    # Solve
    y_interior = gauss_eliminate(coefficient_matrix, right_hand_side)
    
    # Assemble solution
    y_values = np.zeros(num_intervals + 1)
    y_values[0] = y_start
    y_values[-1] = y_end
    y_values[1:-1] = y_interior
    
    return {
        'x_values': x_values,
        'y_values': y_values,
        'step_size': step_size,
        'coefficient_k': coefficient_k,
    }


# =============================================================================
# 6. FOURIER ANALYSIS
# =============================================================================

def fourier_coefficients(
    func: callable,
    period: float,
    num_terms: int,
    num_integration_points: int = 1000,
) -> dict:
    """Compute Fourier coefficients numerically using trapezoidal integration.
    
    Parameters
    ----------
    func : callable
        The periodic function f(x).
    period : float
        The period T of the function.
    num_terms : int
        Number of Fourier terms to compute (n = 1, 2, ..., N).
    num_integration_points : int, optional
        Number of points for numerical integration.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'a0': the constant term (a₀)
        - 'a_n': array of cosine coefficients [a₁, a₂, ..., aₙ]
        - 'b_n': array of sine coefficients [b₁, b₂, ..., bₙ]
        - 'period': the period T
        - 'omega': angular frequency ω = 2π/T
    
    Notes
    -----
    **Fourier Series** represents a periodic function as:
    
        f(x) = a₀/2 + Σₙ[aₙcos(nωx) + bₙsin(nωx)]
    
    where ω = 2π/T.
    
    **Fourier Coefficients:**
    
        a₀ = (2/T) ∫₀ᵀ f(x) dx
        aₙ = (2/T) ∫₀ᵀ f(x)cos(nωx) dx
        bₙ = (2/T) ∫₀ᵀ f(x)sin(nωx) dx
    
    **Numerical Integration:**
    
    Trapezoidal rule: ∫f(x)dx ≈ Σ[f(xᵢ) + f(xᵢ₊₁)]/2 × Δx
    
    **Common Results:**
    
    - Square wave: bₙ = 4/(nπ) for odd n, 0 for even n
    - Triangle wave: bₙ = 8(-1)^((n-1)/2)/(n²π²) for odd n
    - Sawtooth wave: bₙ = 2(-1)^(n+1)/(nπ)
    
    Examples
    --------
    >>> # Square wave coefficients
    >>> def square(x): return np.where((x % (2*np.pi)) < np.pi, 1, -1)
    >>> coeffs = fourier_coefficients(square, period=2*np.pi, num_terms=10)
    """
    omega = 2 * np.pi / period
    x_integration = np.linspace(0, period, num_integration_points)
    f_values = func(x_integration)
    
    # Compute a₀ using trapezoidal rule
    a0 = (2 / period) * np.trapezoid(f_values, x_integration)
    
    # Compute aₙ and bₙ for n = 1, 2, ..., N
    a_coefficients = np.zeros(num_terms)
    b_coefficients = np.zeros(num_terms)
    
    for harmonic_index in range(1, num_terms + 1):
        cos_values = f_values * np.cos(harmonic_index * omega * x_integration)
        sin_values = f_values * np.sin(harmonic_index * omega * x_integration)
        
        a_coefficients[harmonic_index - 1] = (
            (2 / period) * np.trapezoid(cos_values, x_integration)
        )
        b_coefficients[harmonic_index - 1] = (
            (2 / period) * np.trapezoid(sin_values, x_integration)
        )
    
    return {
        'a0': a0,
        'a_n': a_coefficients,
        'b_n': b_coefficients,
        'period': period,
        'omega': omega,
    }


def evaluate_fourier_series(
    x_values: np.ndarray,
    coefficients: dict,
    num_terms: int = None,
) -> np.ndarray:
    """Evaluate the Fourier series at given x values.
    
    Parameters
    ----------
    x_values : np.ndarray
        Points at which to evaluate the series.
    coefficients : dict
        Fourier coefficients from fourier_coefficients().
    num_terms : int, optional
        Number of terms to use. If None, uses all available.
    
    Returns
    -------
    np.ndarray
        Values of the Fourier series at x_values.
    
    Notes
    -----
    **Partial Sum:**
    
        Sₙ(x) = a₀/2 + Σₖ₌₁ⁿ[aₖcos(kωx) + bₖsin(kωx)]
    
    **Gibbs Phenomenon:**
    
    Near discontinuities, the partial sum overshoots by approximately 9%.
    This overshoot does not decrease as more terms are added.
    
    Examples
    --------
    >>> coeffs = fourier_coefficients(square_wave, 2*np.pi, 50)
    >>> x = np.linspace(0, 4*np.pi, 1000)
    >>> y = evaluate_fourier_series(x, coeffs, num_terms=10)
    """
    omega = coefficients['omega']
    a0 = coefficients['a0']
    a_n = coefficients['a_n']
    b_n = coefficients['b_n']
    
    if num_terms is None:
        num_terms = len(a_n)
    
    # Start with constant term
    result = np.ones_like(x_values) * a0 / 2
    
    # Add harmonic terms
    for harmonic_index in range(1, min(num_terms + 1, len(a_n) + 1)):
        result += (
            a_n[harmonic_index - 1] * np.cos(harmonic_index * omega * x_values) +
            b_n[harmonic_index - 1] * np.sin(harmonic_index * omega * x_values)
        )
    
    return result


def parseval_sum(coefficients: dict, num_terms: int = None) -> float:
    """Compute the Parseval sum from Fourier coefficients.
    
    Parameters
    ----------
    coefficients : dict
        Fourier coefficients.
    num_terms : int, optional
        Number of terms to include.
    
    Returns
    -------
    float
        The Parseval sum.
    
    Notes
    -----
    **Parseval's Theorem:**
    
        (1/T) ∫₀ᵀ |f(x)|² dx = (a₀/2)² + (1/2)Σₙ(aₙ² + bₙ²)
    
    **Interpretation:**
    
    - Left side: mean square value of f(x)
    - Right side: sum of squared amplitudes
    - Energy is conserved between time and frequency domains
    
    **Applications:**
    
    - Verify correctness of Fourier coefficients
    - Compute infinite series (e.g., Σ1/n² = π²/6)
    
    Examples
    --------
    >>> # For square wave: (1/T)∫|f|²dx = 1
    >>> coeffs = fourier_coefficients(square_wave, 2*np.pi, 100)
    >>> parseval_sum(coeffs)  # Should approach 1.0
    """
    a0 = coefficients['a0']
    a_n = coefficients['a_n']
    b_n = coefficients['b_n']
    
    if num_terms is None:
        num_terms = len(a_n)
    
    total = (a0 / 2)**2
    
    for term_index in range(min(num_terms, len(a_n))):
        total += 0.5 * (a_n[term_index]**2 + b_n[term_index]**2)
    
    return total


def mean_square_value(
    func: callable,
    period: float,
    num_integration_points: int = 1000,
) -> float:
    """Compute the mean square value of a periodic function.
    
    Parameters
    ----------
    func : callable
        The function f(x).
    period : float
        The period T.
    num_integration_points : int, optional
        Number of integration points.
    
    Returns
    -------
    float
        The mean square value: (1/T) ∫₀ᵀ |f(x)|² dx.
    
    Notes
    -----
    **Mean Square Value** measures the "average power" of a signal.
    
    **Formula:**
    
        ⟨f²⟩ = (1/T) ∫₀ᵀ f(x)² dx
    
    **Relation to Parseval:**
    
    By Parseval's theorem, this equals the sum of squared Fourier amplitudes.
    
    Examples
    --------
    >>> # Square wave has mean square value = 1
    >>> def square(x): return np.where((x % (2*np.pi)) < np.pi, 1, -1)
    >>> mean_square_value(square, 2*np.pi)  # Returns 1.0
    """
    x_integration = np.linspace(0, period, num_integration_points)
    f_squared = func(x_integration)**2
    
    return np.trapezoid(f_squared, x_integration) / period


# =============================================================================
# STANDARD WAVEFORMS (for Fourier analysis)
# =============================================================================

def square_wave(x: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    """Generate a square wave with amplitude ±1."""
    normalised_x = (x % period) / period
    return np.where(normalised_x < 0.5, 1.0, -1.0)


def triangle_wave(x: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    """Generate a triangle wave with amplitude ±1."""
    normalised_x = (x % period) / period
    return np.where(normalised_x < 0.5, 4 * normalised_x - 1, 3 - 4 * normalised_x)


def sawtooth_wave(x: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    """Generate a sawtooth wave with amplitude ±1."""
    normalised_x = (x % period) / period
    return 2 * normalised_x - 1
