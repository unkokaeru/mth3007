"""Session 9: Ordinary Differential Equations Solutions.

This module provides solutions to the Session 9 exercises covering:
- Finite difference methods for ODEs
- Second-order ODEs with boundary conditions
- QR algorithm for eigenvalues

Author: William Fayers
"""

import numpy as np
import matplotlib.pyplot as plt


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


def solve_second_order_ode(
    source_function: callable,
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    num_intervals: int,
) -> dict:
    """Solve d²y/dx² = g(x) with Dirichlet boundary conditions.
    
    Uses central finite differences:
        d²y/dx² ≈ (yᵢ₋₁ - 2yᵢ + yᵢ₊₁) / h²
    
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
    The discretisation leads to:
        yᵢ₋₁ - 2yᵢ + yᵢ₊₁ = h²g(xᵢ)
    
    For interior points i = 1, 2, ..., n-1, this gives a tridiagonal system.
    """
    step_size = (x_end - x_start) / num_intervals
    num_interior = num_intervals - 1
    
    # Create grid points (including boundaries)
    x_values = np.linspace(x_start, x_end, num_intervals + 1)
    
    # Interior points
    x_interior = x_values[1:-1]
    
    # Build tridiagonal coefficient matrix
    # The equation yᵢ₋₁ - 2yᵢ + yᵢ₊₁ = h²g(xᵢ) gives:
    # [-2  1  0  0 ...] [y₁]   [h²g(x₁) - y₀]
    # [ 1 -2  1  0 ...] [y₂]   [h²g(x₂)    ]
    # [ 0  1 -2  1 ...] [y₃] = [h²g(x₃)    ]
    # [... ... ...    ] [...]   [...         ]
    
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
    """Solve d²y/dx² + k*y = g(x) with Dirichlet boundary conditions.
    
    Uses central finite differences:
        (yᵢ₋₁ - 2yᵢ + yᵢ₊₁)/h² + k*yᵢ = g(xᵢ)
    
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
    The discretisation leads to:
        yᵢ₋₁ + (-2 + k*h²)yᵢ + yᵢ₊₁ = h²g(xᵢ)
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


def qr_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute QR decomposition using Gram-Schmidt orthogonalisation.
    
    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to decompose.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Q (orthogonal) and R (upper triangular) matrices such that A = QR.
    
    Notes
    -----
    Uses the Gram-Schmidt process to build orthogonal columns iteratively.
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
    The QR algorithm repeatedly computes A = QR, then A ← RQ.
    Under certain conditions, A converges to an upper triangular
    (or block upper triangular) matrix with eigenvalues on the diagonal.
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


def task1_simple_ode() -> dict:
    """Solve d²y/dx² = g(x) with finite differences.
    
    Example: d²y/dx² = 2 on [0, 1] with y(0) = 0, y(1) = 0.
    Exact solution: y = x(1 - x)
    
    Returns
    -------
    dict
        Results including numerical and exact solutions.
    """
    def source_function(x):
        return -2 * np.ones_like(x)  # For y = x(1-x), y'' = -2
    
    def exact_solution(x):
        return x * (1 - x)
    
    # Solve with different grid sizes
    results = {}
    for num_intervals in [4, 8, 16, 32]:
        numerical_result = solve_second_order_ode(
            source_function,
            x_start=0.0,
            x_end=1.0,
            y_start=0.0,
            y_end=0.0,
            num_intervals=num_intervals,
        )
        
        # Calculate exact solution and error
        exact_values = exact_solution(numerical_result['x_values'])
        max_error = np.max(np.abs(numerical_result['y_values'] - exact_values))
        
        results[num_intervals] = {
            'numerical': numerical_result,
            'exact': exact_values,
            'max_error': max_error,
        }
    
    return results


def task2_ode_with_y() -> dict:
    """Solve d²y/dx² + k*y = g(x) with finite differences.
    
    Example: d²y/dx² - y = 0 on [0, 1] with y(0) = 0, y(1) = sinh(1).
    Exact solution: y = sinh(x)
    
    Returns
    -------
    dict
        Results including numerical and exact solutions.
    """
    def source_function(x):
        return np.zeros_like(x)
    
    def exact_solution(x):
        return np.sinh(x)
    
    # Solve with k = -1 (i.e., y'' - y = 0)
    results = {}
    for num_intervals in [4, 8, 16, 32]:
        numerical_result = solve_ode_with_y_term(
            source_function,
            coefficient_k=-1.0,
            x_start=0.0,
            x_end=1.0,
            y_start=0.0,
            y_end=np.sinh(1.0),
            num_intervals=num_intervals,
        )
        
        # Calculate exact solution and error
        exact_values = exact_solution(numerical_result['x_values'])
        max_error = np.max(np.abs(numerical_result['y_values'] - exact_values))
        
        results[num_intervals] = {
            'numerical': numerical_result,
            'exact': exact_values,
            'max_error': max_error,
        }
    
    return results


def task3_qr_eigenvalues() -> dict:
    """Demonstrate the QR algorithm for finding eigenvalues.
    
    Returns
    -------
    dict
        Results for test matrices.
    """
    results = {}
    
    # Test matrix 1: Symmetric matrix
    matrix_1 = np.array([
        [4, 1, 1],
        [1, 3, 2],
        [1, 2, 5],
    ])
    results['symmetric'] = {
        'matrix': matrix_1,
        'qr_result': qr_algorithm_eigenvalues(matrix_1),
        'numpy_eigenvalues': np.linalg.eigvals(matrix_1),
    }
    
    # Test matrix 2: Simple 2x2
    matrix_2 = np.array([
        [2, 1],
        [1, 2],
    ])
    results['simple_2x2'] = {
        'matrix': matrix_2,
        'qr_result': qr_algorithm_eigenvalues(matrix_2),
        'numpy_eigenvalues': np.linalg.eigvals(matrix_2),
    }
    
    # Test matrix 3: Diagonal-dominant
    matrix_3 = np.array([
        [10, 1, 0],
        [1, 8, 1],
        [0, 1, 6],
    ])
    results['diagonal_dominant'] = {
        'matrix': matrix_3,
        'qr_result': qr_algorithm_eigenvalues(matrix_3),
        'numpy_eigenvalues': np.linalg.eigvals(matrix_3),
    }
    
    return results


def plot_ode_solutions(
    results: dict,
    title: str,
    exact_func: callable = None,
) -> None:
    """Plot ODE solutions for different grid sizes.
    
    Parameters
    ----------
    results : dict
        Results from task1_simple_ode or task2_ode_with_y.
    title : str
        Plot title.
    exact_func : callable, optional
        Exact solution function for plotting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['blue', 'green', 'orange', 'red']
    
    # Plot 1: Solutions
    for i, (num_intervals, data) in enumerate(results.items()):
        axes[0].plot(
            data['numerical']['x_values'],
            data['numerical']['y_values'],
            f'{colors[i]}o-',
            markersize=6,
            label=f'n = {num_intervals}',
        )
    
    # Plot exact solution
    if exact_func is not None:
        x_exact = np.linspace(0, 1, 100)
        y_exact = exact_func(x_exact)
        axes[0].plot(x_exact, y_exact, 'k--', linewidth=2, label='Exact')
    
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'{title}: Numerical Solutions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Error convergence
    num_intervals_list = list(results.keys())
    errors = [results[num_intervals]['max_error'] for num_intervals in num_intervals_list]
    step_sizes = [1 / num_intervals for num_intervals in num_intervals_list]
    
    axes[1].loglog(step_sizes, errors, 'bo-', markersize=10, linewidth=2, label='Max error')
    
    # Add reference line for O(h²) convergence
    h_ref = np.array(step_sizes)
    error_ref = errors[0] * (h_ref / step_sizes[0])**2
    axes[1].loglog(h_ref, error_ref, 'r--', linewidth=1, label='O(h²) reference')
    
    axes[1].set_xlabel('Step size h')
    axes[1].set_ylabel('Maximum error')
    axes[1].set_title('Error Convergence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()


def display_coefficient_matrix(
    matrix: np.ndarray,
    step_size: float,
    ode_type: str = "simple",
) -> None:
    """Display the tridiagonal coefficient matrix.
    
    Parameters
    ----------
    matrix : np.ndarray
        The coefficient matrix.
    step_size : float
        The step size h.
    ode_type : str
        Type of ODE ('simple' or 'with_y').
    """
    print("\nCoefficient Matrix (tridiagonal structure):")
    print("-" * 50)
    print(f"Step size h = {step_size:.4f}")
    print()
    
    num_rows = matrix.shape[0]
    if num_rows <= 6:
        print(matrix)
    else:
        print("(Matrix too large to display fully)")
        print(f"Diagonal elements: {matrix[0, 0]:.4f}")
        print(f"Off-diagonal elements: {matrix[0, 1]:.4f} (if exists)")


def main() -> None:
    """Execute all Session 9 tasks and display results."""
    print("=" * 60)
    print("SESSION 9: ORDINARY DIFFERENTIAL EQUATIONS")
    print("=" * 60)
    
    # Task 1: Simple second-order ODE
    print("\n--- Task 1: d²y/dx² = g(x) ---\n")
    print("Example: d²y/dx² = -2 on [0, 1]")
    print("Boundary conditions: y(0) = 0, y(1) = 0")
    print("Exact solution: y = x(1 - x)")
    
    task1_results = task1_simple_ode()
    
    print("\nFinite Difference Discretisation:")
    print("  d²y/dx² ≈ (yᵢ₋₁ - 2yᵢ + yᵢ₊₁) / h²")
    print("\nThis gives the tridiagonal system:")
    print("  yᵢ₋₁ - 2yᵢ + yᵢ₊₁ = h²g(xᵢ)")
    
    # Display coefficient matrix for n=4
    display_coefficient_matrix(
        task1_results[4]['numerical']['coefficient_matrix'],
        task1_results[4]['numerical']['step_size'],
    )
    
    print("\nConvergence Analysis:")
    print("  n   |    h    |   Max Error   |  Error/h²")
    print("------|---------|---------------|------------")
    for num_intervals in [4, 8, 16, 32]:
        step_size = 1 / num_intervals
        max_error = task1_results[num_intervals]['max_error']
        ratio = max_error / step_size**2
        print(f" {num_intervals:4d} | {step_size:.5f} | {max_error:.6e} | {ratio:.6f}")
    
    print("\n  The ratio Error/h² is approximately constant,")
    print("  confirming O(h²) convergence.")
    
    # Plot results
    print("\nGenerating plots...")
    plot_ode_solutions(task1_results, "Task 1", exact_func=lambda x: x * (1 - x))
    
    # Task 2: ODE with y term
    print("\n" + "=" * 60)
    print("Task 2: d²y/dx² + k·y = g(x)")
    print("=" * 60)
    
    print("\nExample: d²y/dx² - y = 0 on [0, 1]")
    print("Boundary conditions: y(0) = 0, y(1) = sinh(1)")
    print("Exact solution: y = sinh(x)")
    
    task2_results = task2_ode_with_y()
    
    print("\nModified Discretisation:")
    print("  (yᵢ₋₁ - 2yᵢ + yᵢ₊₁)/h² + k·yᵢ = g(xᵢ)")
    print("  → yᵢ₋₁ + (-2 + k·h²)yᵢ + yᵢ₊₁ = h²g(xᵢ)")
    
    print("\nConvergence Analysis:")
    print("  n   |    h    |   Max Error   |  Error/h²")
    print("------|---------|---------------|------------")
    for num_intervals in [4, 8, 16, 32]:
        step_size = 1 / num_intervals
        max_error = task2_results[num_intervals]['max_error']
        ratio = max_error / step_size**2
        print(f" {num_intervals:4d} | {step_size:.5f} | {max_error:.6e} | {ratio:.6f}")
    
    # Plot results
    plot_ode_solutions(task2_results, "Task 2", exact_func=np.sinh)
    
    # Task 3: QR Algorithm
    print("\n" + "=" * 60)
    print("Task 3: QR Algorithm for Eigenvalues")
    print("=" * 60)
    
    task3_results = task3_qr_eigenvalues()
    
    for name, data in task3_results.items():
        print(f"\n--- {name.replace('_', ' ').title()} Matrix ---")
        print("\nOriginal matrix:")
        print(data['matrix'])
        
        qr_result = data['qr_result']
        print(f"\nQR Algorithm:")
        print(f"  Converged: {qr_result['converged']}")
        print(f"  Iterations: {qr_result['iterations']}")
        print(f"  Eigenvalues: {np.sort(qr_result['eigenvalues'])[::-1]}")
        
        print(f"\nNumPy verification:")
        print(f"  Eigenvalues: {np.sort(np.real(data['numpy_eigenvalues']))[::-1]}")
    
    # Summary of methods
    print("\n" + "=" * 60)
    print("METHOD SUMMARY")
    print("=" * 60)
    print("""
    FINITE DIFFERENCE METHOD FOR ODEs:
    
    1. For d²y/dx² = g(x):
       - Use central difference: y'' ≈ (yᵢ₋₁ - 2yᵢ + yᵢ₊₁)/h²
       - Leads to tridiagonal system: Ay = b
       - Convergence: O(h²)
    
    2. For d²y/dx² + k·y = g(x):
       - Modify diagonal: -2 → -2 + k·h²
       - Same tridiagonal structure
       - Convergence: O(h²)
    
    QR ALGORITHM FOR EIGENVALUES:
    
    1. Compute A = QR (QR decomposition)
    2. Form A ← RQ
    3. Repeat until A is nearly diagonal
    4. Eigenvalues appear on the diagonal
    
    The QR algorithm converges for most matrices, with
    faster convergence for matrices with well-separated eigenvalues.
    """)
    
    print("\n" + "=" * 60)
    print("SESSION 9 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
