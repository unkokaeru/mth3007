"""
MTH3007 Numerical Methods: William Fayers
"""


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


if __name__ == "__main__":
    question_one()
