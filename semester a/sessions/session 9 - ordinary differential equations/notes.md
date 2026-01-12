# Session 9: Ordinary Differential Equations

> **MTH3007 Numerical Methods** · Matt Watkins · <mwatkins@lincoln.ac.uk>

---

## Overview

This session covers numerical methods for solving ordinary differential equations using finite differences, and introduces LU and QR matrix decompositions.

## Learning Outcomes

- Solve second-order ODEs numerically using finite differences.
- Understand LU decomposition for efficient equation solving.
- Apply QR decomposition for eigenvalue problems.

## Recommended Reading

- Numerical Recipes, Chapters 2, 11
- Chapra and Canale, Chapters 10, 27

---

## Discretisation

We approximate a function $y(x)$ on a grid of $n$ points with spacing $\Delta x$:

$$x_i = x_{\min} + \frac{i(x_{\max} - x_{\min})}{n - 1} \quad \text{for } i = 0, 1, \ldots, n-1$$

---

## Finite Difference Approximations

### First Derivative (Central Difference)

$$\frac{dy}{dx}(x_i) \approx \frac{y_{i+1} - y_{i-1}}{2\Delta x}$$

### Second Derivative (Central Difference)

$$\frac{d^2 y}{dx^2}(x_i) \approx \frac{y_{i-1} - 2y_i + y_{i+1}}{\Delta x^2}$$

This is derived by applying the first derivative formula twice.

---

## Solving $\frac{d^2y}{dx^2} = g(x)$

Using the finite difference approximation:

$$\frac{y_{i-1} - 2y_i + y_{i+1}}{\Delta x^2} = g(x_i)$$

With boundary conditions $y(x_{\min}) = y_L$ and $y(x_{\max}) = y_R$, we can write this as a matrix equation:

$$\mathbf{My} = \mathbf{b}$$

### Matrix Form

$$\begin{pmatrix}
-2 & 0 & 0 & \cdots & 0 \\
1 & -2 & 1 & \cdots & 0 \\
0 & \ddots & \ddots & \ddots & 0 \\
0 & \cdots & 1 & -2 & 1 \\
0 & \cdots & 0 & 0 & -2
\end{pmatrix}
\begin{pmatrix}
y_0 \\ y_1 \\ \vdots \\ y_{n-2} \\ y_{n-1}
\end{pmatrix}
=
\begin{pmatrix}
-2y_L \\
g(x_1)\Delta x^2 \\
\vdots \\
g(x_{n-2})\Delta x^2 \\
-2y_R
\end{pmatrix}$$

The first and last rows enforce the boundary conditions.

---

## Including a $y$ Term: $\frac{d^2y}{dx^2} + ky = g(x)$

Substituting the finite difference:

$$\frac{y_{i-1} - 2y_i + y_{i+1}}{\Delta x^2} + ky_i = g(x_i)$$

Rearranging:

$$y_{i-1} + (k\Delta x^2 - 2)y_i + y_{i+1} = g(x_i)\Delta x^2$$

The matrix $\mathbf{M}$ has diagonal elements $k\Delta x^2 - 2$ instead of $-2$.

---

## LU Decomposition

For repeated solutions of $\mathbf{Ax} = \mathbf{b}$ with different $\mathbf{b}$ vectors, LU decomposition is more efficient than Gaussian elimination.

### Concept

Decompose $\mathbf{A}$ into:

$$\mathbf{A} = \mathbf{LU}$$

where:

- $\mathbf{L}$ is **lower triangular** with 1s on the diagonal
- $\mathbf{U}$ is **upper triangular**

### Solving $\mathbf{Ax} = \mathbf{b}$

1. **Decompose once:** $\mathbf{A} = \mathbf{LU}$
2. **Forward substitution:** Solve $\mathbf{Ld} = \mathbf{b}$ for $\mathbf{d}$
3. **Back substitution:** Solve $\mathbf{Ux} = \mathbf{d}$ for $\mathbf{x}$

### Advantages

- Decomposition: $O(n^3)$ once
- Each solve: $O(n^2)$
- Matrix inversion: Solve for each column of $\mathbf{I}$
- Determinant: $\det(\mathbf{A}) = \det(\mathbf{L})\det(\mathbf{U}) = \prod u_{ii}$

---

## QR Decomposition

Any real square matrix $\mathbf{A}$ can be decomposed as:

$$\mathbf{A} = \mathbf{QR}$$

where:

- $\mathbf{Q}$ is **orthogonal**: $\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$
- $\mathbf{R}$ is **upper triangular**

### Gram-Schmidt Process

Build orthogonal columns iteratively:

1. Keep first column: $\mathbf{u}_1 = \mathbf{a}_1$
2. For each subsequent column, subtract projections onto previous columns:
   $$\mathbf{u}_k = \mathbf{a}_k - \sum_{j=1}^{k-1} \frac{\mathbf{u}_j^T \mathbf{a}_k}{\mathbf{u}_j^T \mathbf{u}_j} \mathbf{u}_j$$
3. Normalise: $\mathbf{q}_k = \mathbf{u}_k / \|\mathbf{u}_k\|$

Then $\mathbf{R} = \mathbf{Q}^T\mathbf{A}$.

---

## QR Algorithm for Eigenvalues

The sequence:

$$\mathbf{A}^{(0)} = \mathbf{A}$$
$$\mathbf{A}^{(k)} = \mathbf{Q}^{(k)}\mathbf{R}^{(k)} \quad \text{(QR decomposition)}$$
$$\mathbf{A}^{(k+1)} = \mathbf{R}^{(k)}\mathbf{Q}^{(k)}$$

converges to an upper triangular matrix with eigenvalues on the diagonal.

### Finding Eigenvectors

Once an eigenvalue $\lambda$ is known, solve:

$$(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$$

This matrix is singular, so use Gaussian elimination to find the null space.

---

## Python Implementation

```python
"""ODE solver using finite differences with LU decomposition."""

import numpy as np
from scipy import linalg


def solve_ode_dirichlet(
    g: callable,
    x_min: float,
    x_max: float,
    y_left: float,
    y_right: float,
    n: int,
    k: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve y'' + k*y = g(x) with Dirichlet boundary conditions.
    
    Args:
        g: Source function g(x).
        x_min, x_max: Domain boundaries.
        y_left, y_right: Boundary values.
        n: Number of grid points.
        k: Coefficient of y term (default 0).
    
    Returns:
        Tuple of (x_values, y_values).
    """
    x = np.linspace(x_min, x_max, n)
    dx = x[1] - x[0]
    
    # Build tridiagonal matrix
    diag = (k * dx**2 - 2) * np.ones(n)
    diag[0] = -2
    diag[-1] = -2
    
    off_diag = np.ones(n - 1)
    off_diag[0] = 0
    off_diag[-1] = 0
    
    M = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    
    # Build RHS vector
    b = g(x) * dx**2
    b[0] = -2 * y_left
    b[-1] = -2 * y_right
    
    # Solve using LU decomposition
    lu, piv = linalg.lu_factor(M)
    y = linalg.lu_solve((lu, piv), b)
    
    return x, y


def qr_eigenvalues(A: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Find eigenvalues using QR algorithm.
    
    Args:
        A: Square matrix.
        max_iter: Maximum iterations.
    
    Returns:
        Array of eigenvalues (diagonal of converged matrix).
    """
    Ak = A.copy()
    for _ in range(max_iter):
        Q, R = np.linalg.qr(Ak)
        Ak = R @ Q
    return np.diag(Ak)


def main() -> None:
    """Demonstrate ODE solving."""
    # Solve y'' = x with y(0) = 0.2, y(1) = 1.5
    g = lambda x: x
    x, y = solve_ode_dirichlet(g, 0.0, 1.0, 0.2, 1.5, n=100)
    
    # Analytical solution: y = x^3/6 + Ax + B
    # With BCs: y(0) = 0.2 -> B = 0.2
    #           y(1) = 1.5 -> 1/6 + A + 0.2 = 1.5 -> A = 7/6
    y_exact = x**3 / 6 + 7/6 * x + 0.2
    
    print(f"Maximum error: {np.max(np.abs(y - y_exact)):.2e}")


if __name__ == "__main__":
    main()
```

---

## Summary

| Method | Purpose | Complexity |
|--------|---------|------------|
| Finite differences | Discretise ODEs | $O(n)$ setup |
| LU decomposition | Efficient repeated solves | $O(n^3)$ once, $O(n^2)$ per solve |
| QR decomposition | Orthogonalisation | $O(n^3)$ |
| QR algorithm | Eigenvalue computation | $O(n^3)$ per iteration |
