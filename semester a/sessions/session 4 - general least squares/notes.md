# Session 4: General Linear Least Squares

> Matt Watkins · <mwatkins@lincoln.ac.uk> (edited by William Fayers)

---

## Overview

This session generalises regression to arbitrary basis functions and introduces error estimation for fitted parameters.

## Learning Outcomes

- Generalise regression to any set of basis functions.
- Understand the matrix formulation of least squares.
- Estimate confidence intervals for fitted parameters.

## Recommended Reading

- Chapra and Canale, Introduction to Part 5 and Chapter 17
- Numerical Recipes, Chapters 14–15

---

## General Linear Least Squares

Simple linear, polynomial, and multiple linear regression can all be generalised to:

$$y_i = a_0 z_0(x_i) + a_1 z_1(x_i) + \ldots + a_{m-1} z_{m-1}(x_i) + e_i$$

The functions $z_0(x), z_1(x), \ldots, z_{m-1}(x)$ are **basis functions**. They define the model and depend only on the independent variable(s).

This is called *linear* least squares because the parameters $a_j$ appear linearly — the basis functions themselves can be highly non-linear in $x$.

### Examples

| Model | Basis Functions |
| ----- | --------------- |
| Linear | $z_0 = 1$, $z_1 = x$ |
| Quadratic | $z_0 = 1$, $z_1 = x$, $z_2 = x^2$ |
| Trigonometric | $z_0 = 1$, $z_1 = \cos(\omega x)$, $z_2 = \sin(\omega x)$ |
| Exponential | $z_0 = 1$, $z_1 = e^{-x}$, $z_2 = e^{-2x}$ |

---

## Matrix Notation

We can write:

$$\mathbf{y} = \mathbf{Za} + \mathbf{e}$$

where:

$$\mathbf{Z} =
\begin{pmatrix}
z_0(x_0) & z_1(x_0) & \cdots & z_{m-1}(x_0) \\
z_0(x_1) & z_1(x_1) & \cdots & z_{m-1}(x_1) \\
\vdots & \vdots & \ddots & \vdots \\
z_0(x_{n-1}) & z_1(x_{n-1}) & \cdots & z_{m-1}(x_{n-1})
\end{pmatrix}$$

$$\mathbf{y} = \begin{pmatrix} y_0 \\ y_1 \\ \vdots \\ y_{n-1} \end{pmatrix}, \quad
\mathbf{a} = \begin{pmatrix} a_0 \\ a_1 \\ \vdots \\ a_{m-1} \end{pmatrix}, \quad
\mathbf{e} = \begin{pmatrix} e_0 \\ e_1 \\ \vdots \\ e_{n-1} \end{pmatrix}$$

---

## Solving the Normal Equations

The sum of squared errors is:

$$S_r = \sum_{i=0}^{n-1} \left(y_i - \sum_{j=0}^{m-1} a_j z_j(x_i)\right)^2 = \mathbf{e}^T \mathbf{e} = (\mathbf{y} - \mathbf{Za})^T (\mathbf{y} - \mathbf{Za})$$

Minimising with respect to $\mathbf{a}$ yields the **normal equations**:

$$\mathbf{Z}^T \mathbf{Z} \mathbf{a} = \mathbf{Z}^T \mathbf{y}$$

This is of the form $\mathbf{Ax} = \mathbf{b}$ and can be solved using Gaussian elimination.

---

## Example: Exponential Basis Functions

Fit the model $y = a_0 + a_1 e^{-x} + a_2 e^{-2x}$ to:

```python
x = [-3.0, -2.3, -1.6, -0.9, -0.2, 0.5, 1.2, 1.9, 2.6, 3.3, 4.0]
y = [8.264, 6.440, 4.749, 4.566, 3.610, 3.327, 2.964, 1.022, 1.095, 1.841, 1.491]
```

**Basis functions:** $z_0 = 1$, $z_1 = e^{-x}$, $z_2 = e^{-2x}$

**Construct Z matrix:**

```python
import numpy as np

x = np.array([-3.0, -2.3, -1.6, -0.9, -0.2, 0.5, 1.2, 1.9, 2.6, 3.3, 4.0])
y = np.array([8.264, 6.440, 4.749, 4.566, 3.610, 3.327, 2.964, 1.022, 1.095, 1.841, 1.491])

Z = np.column_stack([np.ones_like(x), np.exp(-x), np.exp(-2*x)])
```

**Solve normal equations:**

$$\mathbf{Z}^T \mathbf{Z} = \begin{pmatrix}
11 & 39.88 & 535.5 \\
39.88 & 535.5 & 9234 \\
535.5 & 9234 & 173293
\end{pmatrix}$$

$$\mathbf{Z}^T \mathbf{y} = \begin{pmatrix} 39.37 \\ 272.6 \\ 4126 \end{pmatrix}$$

**Solution:** $\mathbf{a} \approx (2.138, 0.586, -0.015)^T$

---

## Statistical Interpretation

The matrix $(\mathbf{Z}^T \mathbf{Z})^{-1}$ contains statistical information about the fitted parameters.

### Standard Error

$$s_{y/x} = \sqrt{\frac{1}{n-m} \sum_{i=0}^{n-1} \left(y_i - \sum_{j=0}^{m-1} a_j z_j(x_i)\right)^2}$$

### Parameter Variance

$$\text{Var}(a_i) = s^2(a_i) = s_{y/x}^2 \cdot [(\mathbf{Z}^T \mathbf{Z})^{-1}]_{ii}$$

### Confidence Intervals

The parameters follow a t-distribution with $n - m$ degrees of freedom:

$$P\left(a_i \in \left(a_i - t_{\alpha/2, n-m} \cdot s(a_i), \; a_i + t_{\alpha/2, n-m} \cdot s(a_i)\right)\right) = 1 - \alpha$$

For 95% confidence, use $\alpha = 0.05$.

---

## Python Implementation

```python
"""General linear least squares fitting with error estimation."""

import numpy as np
from scipy import stats


def fit_general_least_squares(
    x_values: np.ndarray,
    y_values: np.ndarray,
    basis_functions: list,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit data using general linear least squares.
    
    Args:
        x_values: Independent variable values.
        y_values: Dependent variable values.
        basis_functions: List of callable basis functions.
    
    Returns:
        Tuple of (parameters, parameter_variances, standard_error).
    """
    num_data_points = len(x_values)
    num_parameters = len(basis_functions)
    
    # Build design matrix
    design_matrix = np.column_stack(
        [basis_func(x_values) for basis_func in basis_functions]
    )
    
    # Solve normal equations
    normal_matrix = design_matrix.T @ design_matrix
    normal_rhs = design_matrix.T @ y_values
    parameters = np.linalg.solve(normal_matrix, normal_rhs)
    
    # Calculate residuals and standard error
    residuals = y_values - design_matrix @ parameters
    standard_error = np.sqrt(
        np.sum(residuals**2) / (num_data_points - num_parameters)
    )
    
    # Parameter variances
    normal_matrix_inverse = np.linalg.inv(normal_matrix)
    parameter_variances = standard_error**2 * np.diag(normal_matrix_inverse)
    
    return parameters, parameter_variances, standard_error


def main() -> None:
    """Demonstrate general least squares fitting."""
    x_data = np.array([10.0, 16.3, 23.0, 27.5, 31.0, 35.6, 39.0, 41.5,
                       42.9, 45.0, 46.0, 45.5, 46.0, 49.0, 50.0])
    y_data = np.array([8.953, 16.405, 22.607, 27.769, 32.065, 35.641,
                       38.617, 41.095, 43.156, 44.872, 46.301, 47.490,
                       48.479, 49.303, 49.988])
    
    # Linear model: y = a0 + a1*x
    basis_functions = [
        lambda x_val: np.ones_like(x_val),
        lambda x_val: x_val,
    ]
    
    parameters, variances, standard_error = fit_general_least_squares(
        x_data, y_data, basis_functions
    )
    
    print(f"Parameters: a0 = {parameters[0]:.4f}, a1 = {parameters[1]:.4f}")
    print(f"Standard errors: s(a0) = {np.sqrt(variances[0]):.4f}, "
          f"s(a1) = {np.sqrt(variances[1]):.4f}")
    
    # 95% confidence intervals
    num_data_points = len(x_data)
    num_parameters = len(basis_functions)
    t_critical = stats.t.ppf(0.975, num_data_points - num_parameters)
    
    for param_index, param_name in enumerate(["a0", "a1"]):
        margin = t_critical * np.sqrt(variances[param_index])
        print(f"95% CI for {param_name}: ({parameters[param_index] - margin:.4f}, "
              f"{parameters[param_index] + margin:.4f})")


if __name__ == "__main__":
    main()
```

---

## Summary

| Concept | Formula |
|---------|---------|
| General model | $y_i = \sum_{j=0}^{m-1} a_j z_j(x_i) + e_i$ |
| Normal equations | $\mathbf{Z}^T \mathbf{Z} \mathbf{a} = \mathbf{Z}^T \mathbf{y}$ |
| Standard error | $s_{y/x} = \sqrt{\frac{1}{n-m} \sum (y_i - \hat{y}_i)^2}$ |
| Parameter variance | $\text{Var}(a_i) = s_{y/x}^2 [(\mathbf{Z}^T \mathbf{Z})^{-1}]_{ii}$ |
