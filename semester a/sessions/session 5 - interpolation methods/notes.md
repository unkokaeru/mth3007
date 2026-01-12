# Session 5: Interpolation Methods

> Matt Watkins · <mwatkins@lincoln.ac.uk> (edited by William Fayers)

---

## Overview

This session covers methods for estimating function values at points between known data.

## Learning Outcomes

- Understand when to use interpolation versus regression.
- Know the most common interpolation methods.
- Implement Newton's divided difference and Lagrange polynomials.

## Recommended Reading

- Chapra and Canale, Introduction to Part 5 and Chapter 18
- Numerical Recipes, Chapter 3

---

## What is Interpolation?

Given a table of values from an unknown function $f$:

| $x$ | $f(x)$  |
|-----|---------|
| 0   | 0.0000  |
| 1   | 0.8415  |
| 2   | 0.9093  |
| 3   | 0.1411  |
| 4   | -0.7658 |
| 5   | -0.9589 |
| 6   | -0.2794 |

**Interpolation** estimates $f(x)$ at intermediate points (e.g., $x = 2.5$).

### Interpolation vs Regression

| Interpolation | Regression |
| ------------- | ---------- |
| Passes through all data points | Minimises total error |
| Exact at data points | Approximate at data points |
| For precise data | For noisy data |

---

## Linear Interpolation

Given two points $(x_0, y_0)$ and $(x_1, y_1)$, the linear interpolant is:

$$y = y_0 + (x - x_0) \frac{y_1 - y_0}{x_1 - x_0}$$

Or equivalently:

$$y = \frac{y_0(x_1 - x) + y_1(x - x_0)}{x_1 - x_0}$$

This is sometimes called a "lerp" (linear interpolation).

> **Note:** Linear interpolation is the first-order term of a Taylor series expansion.

---

## Newton's Divided Difference Interpolation

### Divided Differences

**First divided differences:**

$$f[x_0, x_1] = \frac{f(x_1) - f(x_0)}{x_1 - x_0}$$

**Higher divided differences** are built recursively:

$$f[x_0, x_1, x_2] = \frac{f[x_1, x_2] - f[x_0, x_1]}{x_2 - x_0}$$

$$f[x_0, x_1, \ldots, x_k] = \frac{f[x_1, \ldots, x_k] - f[x_0, \ldots, x_{k-1}]}{x_k - x_0}$$

### Newton's Polynomial

$$P(x) = f(x_0) + (x - x_0)f[x_0, x_1] + (x - x_0)(x - x_1)f[x_0, x_1, x_2] + \ldots$$

The polynomial can be truncated at any term, with the next term providing an error estimate.

---

## Lagrange Interpolating Polynomials

The Lagrange form of the interpolating polynomial is:

$$p(x) = \sum_{i=0}^{n} L_i(x) \cdot y_i$$

where the **Lagrange basis polynomials** are:

$$L_i(x) = \prod_{\substack{j=0 \\ j \neq i}}^{n} \frac{x - x_j}{x_i - x_j}$$

### Properties

- At $x = x_i$: $L_i(x_i) = 1$ and $L_j(x_i) = 0$ for $j \neq i$
- Therefore $p(x_i) = y_i$ (passes through all data points)

### Example

For points $(0, 1)$, $(1, 3)$, $(3, 55)$:

$$L_0(x) = \frac{(x-1)(x-3)}{(0-1)(0-3)} = \frac{(x-1)(x-3)}{3}$$

$$L_1(x) = \frac{(x-0)(x-3)}{(1-0)(1-3)} = \frac{x(x-3)}{-2}$$

$$L_2(x) = \frac{(x-0)(x-1)}{(3-0)(3-1)} = \frac{x(x-1)}{6}$$

$$p(x) = 1 \cdot L_0(x) + 3 \cdot L_1(x) + 55 \cdot L_2(x)$$

> **Warning:** Polynomial interpolation should not be used for extrapolation — accuracy degrades outside the data range.

---

## Splines

Instead of fitting one high-order polynomial through many points, **splines** join simple low-order polynomials together.

### Requirements for Cubic Splines

1. **Pass through data points:** Provides $2n$ equations
2. **Continuity of first derivative:** Provides $n-1$ equations
3. **Continuity of second derivative:** Provides $n-1$ equations
4. **Boundary conditions:** 2 equations (typically zero curvature at endpoints)

This gives $4n$ equations for the $4n$ coefficients of $n$ cubic polynomials.

### Advantages of Splines

- Avoid oscillations of high-order polynomials
- More stable numerically
- Local control (changing one point affects only nearby segments)

---

## Trigonometric Interpolation

Using trigonometric basis functions:

$$y = a_0 + \sum_{k=1}^{N} \left[ a_k \cos(k\omega x) + b_k \sin(k\omega x) \right]$$

where $\omega = 2\pi / L$ and $L$ is the range of the data.

**Advantages:**

- Works best for periodic or nearly periodic data
- Avoids wild oscillations of polynomial interpolation
- Natural basis for Fourier analysis (see Session 10)

---

## Multidimensional Interpolation

### Bilinear Interpolation

For a function $f(x, y)$ known at four corners of a rectangle:

**Step 1: Interpolate in $x$** at $y = y_1$ and $y = y_2$:

$$f(x, y_1) = \frac{x_2 - x}{x_2 - x_1} f(x_1, y_1) + \frac{x - x_1}{x_2 - x_1} f(x_2, y_1)$$

$$f(x, y_2) = \frac{x_2 - x}{x_2 - x_1} f(x_1, y_2) + \frac{x - x_1}{x_2 - x_1} f(x_2, y_2)$$

**Step 2: Interpolate in $y$**:

$$f(x, y) = \frac{y_2 - y}{y_2 - y_1} f(x, y_1) + \frac{y - y_1}{y_2 - y_1} f(x, y_2)$$

**Application:** Image processing (digital zoom, rotation).

---

## Python Implementation

```python
"""Interpolation methods implementation."""

import numpy as np


def lagrange_interpolate(
    x_data: np.ndarray,
    y_data: np.ndarray,
    target_x: float,
) -> float:
    """Compute Lagrange interpolation at a target point.
    
    Args:
        x_data: Array of x coordinates of data points.
        y_data: Array of y coordinates of data points.
        target_x: Point at which to interpolate.
    
    Returns:
        Interpolated value at target_x.
    """
    num_points = len(x_data)
    result = 0.0
    
    for basis_index in range(num_points):
        # Compute Lagrange basis polynomial L_i(x)
        basis_value = 1.0
        for point_index in range(num_points):
            if basis_index != point_index:
                basis_value *= (
                    (target_x - x_data[point_index]) /
                    (x_data[basis_index] - x_data[point_index])
                )
        result += y_data[basis_index] * basis_value
    
    return result


def newton_divided_diff(
    x_data: np.ndarray,
    y_data: np.ndarray,
    target_x: float,
) -> float:
    """Compute Newton's divided difference interpolation at a target point.
    
    Args:
        x_data: Array of x coordinates of data points.
        y_data: Array of y coordinates of data points.
        target_x: Point at which to interpolate.
    
    Returns:
        Interpolated value at target_x.
    """
    num_points = len(x_data)
    
    # Build divided difference table
    coefficients = np.copy(y_data).astype(float)
    for difference_order in range(1, num_points):
        for index in range(num_points - 1, difference_order - 1, -1):
            coefficients[index] = (
                (coefficients[index] - coefficients[index - 1]) /
                (x_data[index] - x_data[index - difference_order])
            )
    
    # Evaluate polynomial using Horner's method
    result = coefficients[num_points - 1]
    for index in range(num_points - 2, -1, -1):
        result = result * (target_x - x_data[index]) + coefficients[index]
    
    return result


def main() -> None:
    """Demonstrate interpolation methods."""
    x_data = np.array([0.0, 1.0, 3.0])
    y_data = np.array([1.0, 3.0, 55.0])
    
    target_x = 2.0
    
    lagrange_result = lagrange_interpolate(x_data, y_data, target_x)
    newton_result = newton_divided_diff(x_data, y_data, target_x)
    
    print(f"Lagrange interpolation at x={target_x}: {lagrange_result:.4f}")
    print(f"Newton interpolation at x={target_x}: {newton_result:.4f}")


if __name__ == "__main__":
    main()
```

---

## Summary

| Method | Order | Complexity | Best For |
| ------ | ----- | ---------- | -------- |
| Linear | 1 | $O(1)$ | Quick estimates |
| Lagrange | $n-1$ | $O(n^2)$ | Small datasets |
| Newton | $n-1$ | $O(n^2)$ | Adding new points |
| Cubic spline | 3 | $O(n)$ | Smooth curves |
| Bilinear | 1 | $O(1)$ | 2D grid data |
