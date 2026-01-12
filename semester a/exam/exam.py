"""
MTH3007 Numerical Methods: William Fayers
"""
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

# Create figures directory
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def question_one() -> None:
    """
    Find the solutions to the following equation using a Newton-type solver:
    f(x) = -1.259731x^3 + 20.155701x^2 - 85.661729x + 100.778505 = 0

    Using initial guesses of 0, 5, and 10, find all of the solutions to the equation,
    via the Newton-Raphson method and corresponding iterative formula.

    Then, find the maxima/minima of f(x) by locating the roots of f'(x).
    """
    # Initial parameters
    def func(x):
        return -1.259731 * x**3 + 20.155701 * x**2 - 85.661729 * x + 100.778505

    def func_deriv(x):
        return -3.779193 * x**2 + 40.311402 * x - 85.661729

    # Newton's method implementation
    def newton_method(function, derivative, guess, tolerance=1e-10, max_iterations=100):
        current_guess = guess

        for iteration in range(max_iterations):
            function_value = function(current_guess)
            derivative_value = derivative(current_guess)

            if abs(function_value) < tolerance:
                return current_guess

            if derivative_value == 0:
                raise ValueError("Derivative is zero. No solution found.")

            current_guess -= function_value / derivative_value

        raise ValueError("Maximum iterations reached. No solution found.")

    # Find all of the solutions using different initial guesses
    initial_guesses = [0, 5, 10]
    solutions = []

    for guess in initial_guesses:
        try:
            root = newton_method(func, func_deriv, guess)
            if all(abs(root - sol) > 1e-5 for sol in solutions):
                solutions.append(root)
        except ValueError:
            continue

    print("1a. Solutions found...")
    for sol in solutions:
        print(f"x = {sol:.10f}")

    print("\n1b. Found using the Newton-Raphson method with the iterative formula:")
    print("x_{n+1} = x_n - f(x_n) / f'(x_n)")
    print("Along with the initial guesses of 0, 5, and 10.")

    # Find maxima/minima of f(x)
    def func_second_deriv(x):
        return -7.558386 * x + 40.311402

    critical_points = []
    for guess in [2, 7]:  # Guesses near expected critical points
        try:
            crit_point = newton_method(func_deriv, func_second_deriv, guess)
            if all(abs(crit_point - cp) > 1e-5 for cp in critical_points):
                critical_points.append(crit_point)
        except ValueError:
            continue

    print("\n1c. Critical points found...")
    for cp in critical_points:
        second_deriv_value = func_second_deriv(cp)
        nature = "minimum" if second_deriv_value > 0 else "maximum"
        print(f"x = {cp:.10f} is a {nature}")


def question_two() -> None:
    """
    Using the least squares method to fit a model function to the data points.

    We'll fit the data to a cubic polynomial of the form:
    y = a*x^3 + b*x^2 + c*x + d

    Found by visualising the data points first.
    """
    x_data_points = [
        -1.0, -0.89473684, -0.78947368, -0.68421053, -0.57894737,
        -0.47368421, -0.36842105, -0.26315789, -0.15789474, -0.05263158,
        0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
        0.57894737, 0.68421053, 0.78947368, 0.89473684, 1.0
    ]  # Independent variable

    y_data_points = [
        0.20095699, 0.37525348, 0.51164189, 0.61629633, 0.69539091,
        0.75509975, 0.80159698, 0.8410567, 0.87965303, 0.9235601,
        0.97895201, 1.05200289, 1.14888685, 1.27577801, 1.43885048,
        1.64427839, 1.89823585, 2.20689697, 2.57643588, 3.01302669
    ]  # Dependent variable

    # Plot the data points to visualize
    plt.scatter(x_data_points, y_data_points, label='Data Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data Points Visualization')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'q2_data_points.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to figures/q2_data_points.png")

    # Fit a cubic polynomial using least squares
    coefficients = np.polyfit(x_data_points, y_data_points, 3)
    a, b, c, d = coefficients

    print("2a. Fitted cubic polynomial coefficients:")
    print(f"a = {a:.10f}, b = {b:.10f}, c = {c:.10f}, d = {d:.10f}")
    print("\n2b. The model function is:")
    print(f"y = {a:.10f}*x^3 + {b:.10f}*x^2 + {c:.10f}*x + {d:.10f}")

    # Plot the fitted cubic polynomial along with the data points
    x_fit = np.linspace(min(x_data_points), max(x_data_points), 100)
    y_fit = a * x_fit**3 + b * x_fit**2 + c * x_fit + d

    plt.scatter(x_data_points, y_data_points, label='Data Points')
    plt.plot(x_fit, y_fit, color='red', label='Fitted Cubic Polynomial')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cubic Polynomial Fit')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'q2_cubic_fit.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to figures/q2_cubic_fit.png")


def question_three() -> None:
    """
    Find the condition number (ratio of largest to smallest matrix eigenvalues) of:
    A = [[11, 10, 14],
         [12, 11, -13],
         [14, 13, -66]]
         
    Do this using the Power Method to find the largest eigenvalue, and then again
    on the inverse of A to find the smallest eigenvalue. Then compare with
    numpy's built-in `np.linalg.eigs` function - actually `np.linalg.eigvals`,
    I think.

    Then, find the solutions of Ax = b, where:
    b = [1.000, 0.999, 1.000] and b = [0.999, 1.000, 1.000]
    """
    A = np.array([[11, 10, 14],
                  [12, 11, -13],
                  [14, 13, -66]], dtype=float)

    # Power Method implementation, based on Wikipedia linked in question
    def power_method(matrix, num_iterations=1000, tolerance=1e-10):
        b_k = np.random.rand(matrix.shape[1])
        for _ in range(num_iterations):
            b_k1 = np.dot(matrix, b_k)
            b_k1_norm = np.linalg.norm(b_k1)
            b_k = b_k1 / b_k1_norm
        eigenvalue = np.dot(b_k.T, np.dot(matrix, b_k)) / np.dot(b_k.T, b_k)
        return eigenvalue

    largest_eigenvalue = power_method(A)
    smallest_eigenvalue = 1 / power_method(np.linalg.inv(A))
    condition_number = abs(largest_eigenvalue / smallest_eigenvalue)

    print("3a. Condition number calculated using Power Method:")
    print(f"Condition Number = {condition_number:.10f}")

    # Compare with numpy's built-in function
    eigenvalues = np.linalg.eigvals(A)
    abs_eigenvalues = np.abs(eigenvalues)
    numpy_condition_number = np.max(abs_eigenvalues) / np.min(abs_eigenvalues)

    print("\n3b. Condition number calculated using numpy's built-in function:")
    print(f"Condition Number = {numpy_condition_number:.10f}")

    # Solve Ax = b for two different b vectors
    b1 = np.array([1.000, 0.999, 1.000], dtype=float)
    b2 = np.array([0.999, 1.000, 1.000], dtype=float)

    x1 = np.linalg.solve(A, b1)
    x2 = np.linalg.solve(A, b2)
    print("\n3c. Solutions for Ax = b:")
    print(f"For b = [1.000, 0.999, 1.000], x = {x1}")
    print(f"For b = [0.999, 1.000, 1.000], x = {x2}")


def question_four() -> None:
    """
    Numerically approximate the solution (phi) of:
    d phi(x)^2 / dx^2 = alpha exp(-rho (x - x0)^2)
    where x in (-5,5), alpha=0.5, rho=300, and x0=0.5, with boundary conditions:
    phi(-5) = 0, phi(5) = 0

    Then, plot the solution and check it has converged and is indeed a solution to the equation.
    """
    # Parameters
    alpha = 0.5
    rho = 300
    x0 = 0.5
    x_start, x_end = -5, 5
    phi_start, phi_end = 0, 0

    def source_function(x):
        return alpha * np.exp(-rho * (x - x0)**2)

    def solve_ode(num_intervals):
        h = (x_end - x_start) / num_intervals
        n = num_intervals - 1
        x = np.linspace(x_start, x_end, num_intervals + 1)

        # Tridiagonal system: (phi_{i-1} - 2*phi_i + phi_{i+1}) / h^2 = g(x_i)
        A = np.diag(-2 * np.ones(n)) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
        b = h**2 * source_function(x[1:-1])
        b[0] -= phi_start
        b[-1] -= phi_end

        phi_interior = np.linalg.solve(A, b)
        phi = np.concatenate([[phi_start], phi_interior, [phi_end]])
        return x, phi, h

    # Solve with increasing resolution
    print("4a. Solving using finite differences...")
    solutions = []
    for n in [100, 200, 400, 800]:
        x, phi, h = solve_ode(n)
        solutions.append((x, phi, h))
        print(f"n = {n}: phi(x0) = {phi[np.argmin(np.abs(x - x0))]:.10f}")

    # Convergence check
    print("\n4b. Convergence check...")
    for i in range(1, len(solutions)):
        phi_coarse = np.interp(solutions[i][0], solutions[i-1][0], solutions[i-1][1])
        max_diff = np.max(np.abs(solutions[i][1] - phi_coarse))
        print(f"Difference between successive refinements: {max_diff:.2e}")

    # Verify solution
    x_final, phi_final, h_final = solutions[-1]
    d2phi = (phi_final[:-2] - 2*phi_final[1:-1] + phi_final[2:]) / h_final**2
    residual = np.max(np.abs(d2phi - source_function(x_final[1:-1])))
    print(f"\n4c. Max residual: {residual:.2e}")

    # Plot solution
    plt.figure()
    plt.plot(x_final, phi_final, 'b-')
    plt.xlabel('x')
    plt.ylabel('φ(x)')
    plt.title('Solution of the ODE')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, 'q4_solution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to figures/q4_solution.png")


def question_five() -> None:
    """
    Calculate the Fourier series for the function:
    s(x) = 4A/P^2 * (x - P/2)^2,  for 0 <= x <= P
    with P = 5 and A = 1 where the function has been sampled at 10 evenly spaced points.

    Then, plot the value of the cosine coefficients of the series against their index n.

    Finally, check against the results from `np.fft.rfft`.
    """
    # Parameters
    P = 5
    A = 1
    num_samples = 10

    def s_function(x):
        return (4 * A / P**2) * (x - P / 2)**2

    # Calculate Fourier coefficients using trapezoidal integration
    omega = 2 * np.pi / P
    x_integration = np.linspace(0, P, 1000)
    f_values = s_function(x_integration)

    a0 = (2 / P) * np.trapezoid(f_values, x_integration)

    num_terms = num_samples // 2
    a_n = np.zeros(num_terms)
    b_n = np.zeros(num_terms)

    for n in range(1, num_terms + 1):
        cos_values = f_values * np.cos(n * omega * x_integration)
        sin_values = f_values * np.sin(n * omega * x_integration)
        a_n[n - 1] = (2 / P) * np.trapezoid(cos_values, x_integration)
        b_n[n - 1] = (2 / P) * np.trapezoid(sin_values, x_integration)

    print("5a. Fourier coefficients calculated:")
    print(f"a0 = {a0:.10f}")
    for n in range(num_terms):
        print(f"a{n+1} = {a_n[n]:.10f}, b{n+1} = {b_n[n]:.10f}")

    # Plot cosine coefficients
    plt.figure()
    plt.stem(range(1, num_terms + 1), a_n)
    plt.xlabel('n')
    plt.ylabel('a_n')
    plt.title('Cosine Coefficients of the Fourier Series')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, 'q5_cosine_coefficients.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to figures/q5_cosine_coefficients.png")

    # Check against np.fft.rfft
    x_samples = np.linspace(0, P, num_samples, endpoint=False)
    s_samples = s_function(x_samples)
    fft_coeffs = np.fft.rfft(s_samples)

    # Convert FFT to standard Fourier form for comparison
    a0_fft = 2 * fft_coeffs[0].real / num_samples
    a_n_fft = 2 * fft_coeffs[1:-1].real / num_samples
    b_n_fft = -2 * fft_coeffs[1:-1].imag / num_samples

    print("\n5b. Comparison with np.fft.rfft:")
    print(f"a0: trapezoidal = {a0:.10f}, FFT = {a0_fft:.10f}")
    for n in range(len(a_n_fft)):
        print(f"a{n+1}: trapezoidal = {a_n[n]:.10f}, FFT = {a_n_fft[n]:.10f}")
        print(f"b{n+1}: trapezoidal = {b_n[n]:.10f}, FFT = {b_n_fft[n]:.10f}")


if __name__ == "__main__":
    # question_one()
    # question_two()
    # question_three()
    # question_four()
    question_five()
