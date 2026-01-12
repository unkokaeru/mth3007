# Session 8: Root Finding and Optimisation

> **MTH3007 Numerical Methods** · Matt Watkins · <mwatkins@lincoln.ac.uk>

---

## Overview

This session covers numerical methods for finding roots of equations and optimising functions.

## Learning Outcomes

- Understand and implement root-finding algorithms.
- Apply optimisation methods to find function extrema.
- Extend methods to multiple dimensions.

## Recommended Reading

- Numerical Recipes, Chapter 9
- Chapra and Canale, Chapters 5–7

---

## Non-Linear Equations

A function $f(x)$ is **linear** if it satisfies:

$$f(x + y) = f(x) + f(y) \quad \text{and} \quad f(\alpha x) = \alpha f(x)$$

This implies the **superposition principle**: $f(\alpha x + \beta y) = \alpha f(x) + \beta f(y)$

Any equation that doesn't satisfy these properties is **non-linear**.

**Example:** $x^2 + x - 1 = 0$ is non-linear.

---

## Root Finding vs Optimisation

| Problem | Goal | Solve |
| ------- | ---- | ----- |
| Root finding | Find $x$ where $f(x) = 0$ | $f(x) = 0$ |
| Optimisation | Find $x$ where $f(x)$ is minimum/maximum | $f'(x) = 0$ |

Optimisation is root finding applied to the derivative!

---

## Bisection Method

The simplest and most robust root-finding algorithm.

### Algorithm

1. Find an interval $[a, b]$ where $f(a)$ and $f(b)$ have opposite signs (a **bracket**)
2. Compute midpoint $c = (a + b) / 2$
3. If $f(a) \cdot f(c) < 0$, the root is in $[a, c]$; otherwise in $[c, b]$
4. Repeat until $|b - a|$ is sufficiently small

### Bisection Properties

- **Always converges** (if initial bracket contains a root)
- **Slow:** Gains exactly one bit of accuracy per iteration
- Convergence: $|x_{n+1} - x^*| \leq \frac{1}{2}|x_n - x^*|$ (linear)

---

## Newton's Method (Newton-Raphson)

Uses the tangent line to approximate the root.

### Derivation

The tangent line at $(x_n, f(x_n))$ is:

$$y = f(x_n) + f'(x_n)(x - x_n)$$

Setting $y = 0$ and solving for $x$:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

### Newton's Properties

- **Fast convergence:** Quadratic near simple roots ($|x_{n+1} - x^*| \propto |x_n - x^*|^2$)
- **Requires derivative:** Must compute $f'(x)$
- **May diverge:** Sensitive to initial guess
- **Fails when $f'(x_n) = 0$**

---

## Secant Method

Approximates the derivative using finite differences.

### Formula

$$f'(x_n) \approx \frac{f(x_n) - f(x_{n-1})}{x_n - x_{n-1}}$$

Substituting into Newton's method:

$$x_{n+1} = x_n - f(x_n) \frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}$$

Or equivalently:

$$x_{n+1} = \frac{x_{n-1} f(x_n) - x_n f(x_{n-1})}{f(x_n) - f(x_{n-1})}$$

### Secant Properties

- **No derivative needed**
- **Requires two starting points**
- **Superlinear convergence:** Order $\approx 1.618$ (golden ratio)
- Useful when $f'(x)$ is expensive or unavailable

---

## Newton's Method for Optimisation

To find extrema, we need $f'(x) = 0$.

Using a quadratic approximation around $x_k$:

$$f(x) \approx f(x_k) + f'(x_k)(x - x_k) + \frac{1}{2}f''(x_k)(x - x_k)^2$$

Differentiating and setting to zero:

$$\frac{d}{dx}\left[f(x_k) + f'(x_k)(x - x_k) + \frac{1}{2}f''(x_k)(x - x_k)^2\right] = 0$$

$$f'(x_k) + f''(x_k)(x - x_k) = 0$$

Solving for $x = x_{k+1}$:

$$x_{k+1} = x_k - \frac{f'(x_k)}{f''(x_k)}$$

---

## Multidimensional Root Finding

For $\mathbf{F}(\mathbf{x}) = \mathbf{0}$ where $\mathbf{x} \in \mathbb{R}^n$ and $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n$:

### Newton's Method

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \mathbf{J}(\mathbf{x}_k)^{-1} \mathbf{F}(\mathbf{x}_k)$$

where $\mathbf{J}$ is the **Jacobian matrix**:

$$J_{ij} = \frac{\partial F_i}{\partial x_j}$$

In practice, solve the linear system:

$$\mathbf{J}(\mathbf{x}_k) \Delta\mathbf{x} = -\mathbf{F}(\mathbf{x}_k)$$

then update $\mathbf{x}_{k+1} = \mathbf{x}_k + \Delta\mathbf{x}$.

---

## Multidimensional Optimisation

To minimise $f(\mathbf{x})$ where $\mathbf{x} \in \mathbb{R}^n$:

### Optimisation Derivation

where:

- $\nabla f$ is the **gradient**: $(\nabla f)_i = \frac{\partial f}{\partial x_i}$
- $\mathbf{H}$ is the **Hessian**: $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$

The Hessian is the Jacobian of the gradient.

---

## Python Implementation

```python
"""Root finding and optimisation methods."""

import numpy as np
from typing import Callable


def newton_root(
    target_function: Callable[[float], float],
    derivative_function: Callable[[float], float],
    initial_guess: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    """Find root using Newton's method.
    
    Args:
        target_function: Function to find root of.
        derivative_function: Derivative of target_function.
        initial_guess: Initial guess for the root.
        tolerance: Convergence tolerance.
        max_iterations: Maximum iterations.
    
    Returns:
        Approximate root.
    """
    current_estimate = initial_guess
    for _ in range(max_iterations):
        function_value = target_function(current_estimate)
        if abs(function_value) < tolerance:
            return current_estimate
        current_estimate = (
            current_estimate -
            function_value / derivative_function(current_estimate)
        )
    return current_estimate


def secant_method(
    target_function: Callable[[float], float],
    first_guess: float,
    second_guess: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    """Find root using the secant method.
    
    Args:
        target_function: Function to find root of.
        first_guess: First initial guess.
        second_guess: Second initial guess.
        tolerance: Convergence tolerance.
        max_iterations: Maximum iterations.
    
    Returns:
        Approximate root.
    """
    previous_estimate = first_guess
    current_estimate = second_guess
    
    for _ in range(max_iterations):
        previous_value = target_function(previous_estimate)
        current_value = target_function(current_estimate)
        
        if abs(current_value) < tolerance:
            return current_estimate
        
        next_estimate = (
            (previous_estimate * current_value - current_estimate * previous_value) /
            (current_value - previous_value)
        )
        previous_estimate = current_estimate
        current_estimate = next_estimate
    
    return current_estimate


def newton_optimise(
    target_function: Callable[[float], float],
    first_derivative: Callable[[float], float],
    second_derivative: Callable[[float], float],
    initial_guess: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    """Find extremum using Newton's method for optimisation.
    
    Args:
        target_function: Function to optimise.
        first_derivative: First derivative of target_function.
        second_derivative: Second derivative of target_function.
        initial_guess: Initial guess for the extremum.
        tolerance: Convergence tolerance.
        max_iterations: Maximum iterations.
    
    Returns:
        Approximate location of extremum.
    """
    current_estimate = initial_guess
    for _ in range(max_iterations):
        derivative_value = first_derivative(current_estimate)
        if abs(derivative_value) < tolerance:
            return current_estimate
        current_estimate = (
            current_estimate -
            derivative_value / second_derivative(current_estimate)
        )
    return current_estimate


def main() -> None:
    """Demonstrate root finding methods."""
    # Find sqrt(612) by solving x^2 - 612 = 0
    target_function = lambda value: value**2 - 612
    derivative_function = lambda value: 2 * value
    
    root_newton = newton_root(
        target_function, derivative_function, initial_guess=25.0
    )
    root_secant = secant_method(
        target_function, first_guess=20.0, second_guess=30.0
    )
    
    print(f"sqrt(612) via Newton: {root_newton:.10f}")
    print(f"sqrt(612) via Secant: {root_secant:.10f}")
    print(f"Actual value: {np.sqrt(612):.10f}")


if __name__ == "__main__":
    main()
```

---

## Summary

| Method | Convergence | Requirements |
| ------ | ----------- | ------------ |
| Bisection | Linear | Bracket |
| Newton (root) | Quadratic | $f'(x)$ |
| Secant | Superlinear (~1.618) | Two initial points |
| Newton (optimise) | Quadratic | $f'(x)$, $f''(x)$ |
