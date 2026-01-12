"""Session 8: Root Finding and Optimisation Solutions.

This module provides solutions to the Session 8 exercises covering:
- Newton's method for root finding
- Secant method for root finding
- Newton's method for optimisation

Author: William Fayers
"""

import numpy as np
import matplotlib.pyplot as plt


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
    Newton's iteration formula:
        xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)
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
    Secant iteration formula:
        xₙ₊₁ = xₙ - f(xₙ)(xₙ - xₙ₋₁)/(f(xₙ) - f(xₙ₋₁))
    
    The secant method approximates the derivative by:
        f'(xₙ) ≈ (f(xₙ) - f(xₙ₋₁))/(xₙ - xₙ₋₁)
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
        - 'history': list of (x, f(x), f'(x), f''(x)) tuples
        - 'converged': whether the method converged
    
    Notes
    -----
    For optimisation, we find roots of f'(x) = 0:
        xₙ₊₁ = xₙ - f'(xₙ)/f''(xₙ)
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


def task1_square_root_612() -> dict:
    """Find √612 using Newton's method.
    
    Solves x² = 612, i.e., f(x) = x² - 612 = 0.
    
    Returns
    -------
    dict
        Results from Newton's method.
    
    Notes
    -----
    For f(x) = x² - a:
        f'(x) = 2x
        xₙ₊₁ = xₙ - (xₙ² - a)/(2xₙ) = (xₙ + a/xₙ)/2
    """
    def function(x):
        return x**2 - 612
    
    def derivative(x):
        return 2 * x
    
    # Initial guess: √612 ≈ √625 = 25
    initial_guess = 25.0
    
    return newtons_method_root(function, derivative, initial_guess)


def task1_cube_root_cos() -> dict:
    """Find the root of x³ = cos(x) using Newton's method.
    
    Solves f(x) = x³ - cos(x) = 0.
    
    Returns
    -------
    dict
        Results from Newton's method.
    """
    def function(x):
        return x**3 - np.cos(x)
    
    def derivative(x):
        return 3 * x**2 + np.sin(x)
    
    # Initial guess near where x³ ≈ cos(x)
    # For small x: cos(x) ≈ 1, so x ≈ 1
    initial_guess = 1.0
    
    return newtons_method_root(function, derivative, initial_guess)


def task2_secant_comparisons() -> dict:
    """Compare secant method with different starting points.
    
    Returns
    -------
    dict
        Comparison results for both problems.
    """
    # Problem 1: √612
    def sqrt_function(x):
        return x**2 - 612
    
    sqrt_results = {
        'guess_24_26': secant_method(sqrt_function, 24.0, 26.0),
        'guess_20_30': secant_method(sqrt_function, 20.0, 30.0),
        'guess_10_50': secant_method(sqrt_function, 10.0, 50.0),
    }
    
    # Problem 2: x³ = cos(x)
    def cube_cos_function(x):
        return x**3 - np.cos(x)
    
    cube_cos_results = {
        'guess_0_1': secant_method(cube_cos_function, 0.0, 1.0),
        'guess_0.5_1.5': secant_method(cube_cos_function, 0.5, 1.5),
        'guess_0.8_1.0': secant_method(cube_cos_function, 0.8, 1.0),
    }
    
    return {
        'sqrt_612': sqrt_results,
        'cube_cos': cube_cos_results,
    }


def task3_optimisation_example() -> dict:
    """Find stationary points using Newton's method for optimisation.
    
    Example function: f(x) = x³ - 6x² + 9x + 1
    
    Returns
    -------
    dict
        Results for finding stationary points.
    """
    def function(x):
        return x**3 - 6 * x**2 + 9 * x + 1
    
    def first_derivative(x):
        return 3 * x**2 - 12 * x + 9
    
    def second_derivative(x):
        return 6 * x - 12
    
    # Find stationary points from different starting points
    results = {
        'from_x0': newtons_method_optimisation(
            function, first_derivative, second_derivative, 0.0
        ),
        'from_x4': newtons_method_optimisation(
            function, first_derivative, second_derivative, 4.0
        ),
    }
    
    return results


def plot_newton_convergence(results: dict, func: callable, title: str) -> None:
    """Plot the convergence of Newton's method.
    
    Parameters
    ----------
    results : dict
        Results from Newton's method.
    func : callable
        The function being solved.
    title : str
        Plot title.
    """
    history = results['history']
    iterations = [iteration_number for iteration_number in range(len(history))]
    x_values = [history_entry[0] for history_entry in history]
    function_values = [history_entry[1] for history_entry in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Convergence to root
    axes[0].plot(iterations, x_values, 'bo-', markersize=8)
    axes[0].axhline(y=results['root'], color='r', linestyle='--', label=f"Root = {results['root']:.10f}")
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('x')
    axes[0].set_title(f'{title}: Convergence of x')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: |f(x)| convergence (log scale)
    abs_function_values = [abs(function_value) for function_value in function_values]
    axes[1].semilogy(iterations, abs_function_values, 'go-', markersize=8)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('|f(x)|')
    axes[1].set_title(f'{title}: Convergence of |f(x)|')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_function_and_root(
    func: callable,
    root: float,
    x_range: tuple,
    title: str,
) -> None:
    """Plot a function and mark its root.
    
    Parameters
    ----------
    func : callable
        The function to plot.
    root : float
        The root to mark.
    x_range : tuple
        (x_min, x_max) for plotting.
    title : str
        Plot title.
    """
    x_values = np.linspace(x_range[0], x_range[1], 200)
    y_values = func(x_values)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, 'b-', linewidth=2, label='f(x)')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.scatter([root], [0], color='red', s=150, zorder=5, label=f'Root: x = {root:.6f}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.close()


def display_iteration_table(results: dict, method_name: str) -> None:
    """Display a table of iterations.
    
    Parameters
    ----------
    results : dict
        Results from a root-finding method.
    method_name : str
        Name of the method for display.
    """
    print(f"\n{method_name} Iteration Table:")
    print("-" * 60)
    
    if 'history' in results and len(results['history']) > 0:
        if len(results['history'][0]) == 3:  # Newton's method
            print("  n  |       xₙ        |      f(xₙ)       |     f'(xₙ)")
            print("-----|-----------------|------------------|----------------")
            for i, (x, fx, fpx) in enumerate(results['history']):
                print(f"  {i:2d} | {x:15.10f} | {fx:16.10e} | {fpx:14.6f}")
        else:  # Secant method
            print("  n  |       xₙ        |      f(xₙ)")
            print("-----|-----------------|------------------")
            for i, (x, fx) in enumerate(results['history']):
                print(f"  {i:2d} | {x:15.10f} | {fx:16.10e}")


def main() -> None:
    """Execute all Session 8 tasks and display results."""
    print("=" * 60)
    print("SESSION 8: ROOT FINDING AND OPTIMISATION")
    print("=" * 60)
    
    # Task 1: Newton's Method
    print("\n--- Task 1: Newton's Method ---\n")
    
    # Problem 1: √612
    print("Problem 1: Find √612 (solve x² - 612 = 0)")
    sqrt_results = task1_square_root_612()
    
    print(f"\n  Initial guess: 25.0 (since √625 = 25)")
    print(f"  Converged: {sqrt_results['converged']}")
    print(f"  Iterations: {sqrt_results['iterations']}")
    print(f"  Root: {sqrt_results['root']:.15f}")
    print(f"  Verification: ({sqrt_results['root']:.10f})² = {sqrt_results['root']**2:.10f}")
    print(f"  Exact value (NumPy): {np.sqrt(612):.15f}")
    
    display_iteration_table(sqrt_results, "Newton's Method (√612)")
    
    # Plot convergence
    def sqrt_func(x):
        return x**2 - 612
    plot_newton_convergence(sqrt_results, sqrt_func, "√612")
    plot_function_and_root(sqrt_func, sqrt_results['root'], (20, 30), "f(x) = x² - 612")
    
    # Problem 2: x³ = cos(x)
    print("\n" + "-" * 40)
    print("\nProblem 2: Find root of x³ = cos(x)")
    cube_cos_results = task1_cube_root_cos()
    
    print(f"\n  Initial guess: 1.0")
    print(f"  Converged: {cube_cos_results['converged']}")
    print(f"  Iterations: {cube_cos_results['iterations']}")
    print(f"  Root: {cube_cos_results['root']:.15f}")
    root = cube_cos_results['root']
    print(f"  Verification: ({root:.10f})³ = {root**3:.10f}")
    print(f"                cos({root:.10f}) = {np.cos(root):.10f}")
    
    display_iteration_table(cube_cos_results, "Newton's Method (x³ = cos(x))")
    
    def cube_cos_func(x):
        return x**3 - np.cos(x)
    plot_newton_convergence(cube_cos_results, cube_cos_func, "x³ = cos(x)")
    plot_function_and_root(cube_cos_func, cube_cos_results['root'], (0, 2), "f(x) = x³ - cos(x)")
    
    # Task 2: Secant Method Comparisons
    print("\n" + "=" * 60)
    print("Task 2: Secant Method Comparisons")
    print("=" * 60)
    
    secant_results = task2_secant_comparisons()
    
    print("\nProblem 1: √612 with different starting points")
    print("-" * 50)
    for name, result in secant_results['sqrt_612'].items():
        print(f"\n  {name}:")
        print(f"    Converged: {result['converged']}")
        print(f"    Iterations: {result['iterations']}")
        print(f"    Root: {result['root']:.15f}")
    
    print("\n\nProblem 2: x³ = cos(x) with different starting points")
    print("-" * 50)
    for name, result in secant_results['cube_cos'].items():
        print(f"\n  {name}:")
        print(f"    Converged: {result['converged']}")
        print(f"    Iterations: {result['iterations']}")
        print(f"    Root: {result['root']:.15f}")
    
    # Task 3: Newton's Method for Optimisation
    print("\n" + "=" * 60)
    print("Task 3: Newton's Method for Optimisation")
    print("=" * 60)
    
    print("\nFunction: f(x) = x³ - 6x² + 9x + 1")
    print("f'(x) = 3x² - 12x + 9 = 3(x - 1)(x - 3)")
    print("f''(x) = 6x - 12")
    print("\nStationary points occur at x = 1 and x = 3")
    
    opt_results = task3_optimisation_example()
    
    for start_name, result in opt_results.items():
        print(f"\n  Starting from {start_name.replace('from_', 'x = ')}:")
        print(f"    Converged: {result['converged']}")
        print(f"    Iterations: {result['iterations']}")
        print(f"    Stationary point: x = {result['stationary_point']:.10f}")
        print(f"    f(x) = {result['function_value']:.10f}")
        print(f"    f''(x) = {result['second_derivative']:.6f}")
        print(f"    Type: {result['type']}")
    
    # Plot the optimisation function
    x_plot = np.linspace(-1, 4, 200)
    y_plot = x_plot**3 - 6 * x_plot**2 + 9 * x_plot + 1
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = x³ - 6x² + 9x + 1')
    
    for result in opt_results.values():
        x_stat = result['stationary_point']
        y_stat = result['function_value']
        marker = 'o' if result['type'] == 'maximum' else 's'
        color = 'red' if result['type'] == 'maximum' else 'green'
        plt.scatter([x_stat], [y_stat], c=color, s=150, marker=marker, zorder=5,
                    label=f"{result['type'].capitalize()} at x = {x_stat:.2f}")
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title("Newton's Method for Optimisation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.close()
    
    # Summary of convergence rates
    print("\n" + "=" * 60)
    print("CONVERGENCE RATE SUMMARY")
    print("=" * 60)
    print("""
    Newton's Method:
    - Convergence order: 2 (quadratic)
    - Error at step n+1 ≈ C · (error at step n)²
    - Requires: f(x) and f'(x)
    - Fast convergence near the root
    
    Secant Method:
    - Convergence order: φ ≈ 1.618 (golden ratio)
    - Error at step n+1 ≈ C · (error at step n)^φ
    - Requires: f(x) only (no derivative)
    - Slower than Newton but no derivative needed
    
    For optimisation:
    - Find roots of f'(x) = 0
    - Classify using f''(x):
      • f''(x) > 0: local minimum
      • f''(x) < 0: local maximum
      • f''(x) = 0: inflection point (or higher-order test needed)
    """)
    
    print("\n" + "=" * 60)
    print("SESSION 8 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
