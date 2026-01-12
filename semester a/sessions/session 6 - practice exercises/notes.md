# Session 6: Practice Exercises

> **MTH3007 Numerical Methods** · Matt Watkins · <mwatkins@lincoln.ac.uk>

---

## Overview

This session provides practice using the methods covered so far and introduces code organisation techniques.

## Learning Outcomes

- Practice matrix operations, linear equation solving, and curve fitting.
- Learn to organise code into reusable libraries.
- Understand the power method for finding eigenvalues.

## Recommended Reading

- Chapra and Canale, Chapters 9–10, 17–18

---

## Reusing Code

Copying code between projects is error-prone. Instead, organise reusable functions into libraries.

### Creating a Header Library (C++)

1. Create a file `my_library.h`
2. Move reusable functions (not `main`) into this file
3. Include in your project with `#include "my_library.h"`

### Visual Studio Setup

1. Right-click on the project name → Properties
2. Navigate to C/C++ → General
3. Add your library directory to "Additional Include Directories"

### Python Approach

```python
# my_library.py
"""Numerical methods library."""

import numpy as np


def gauss_eliminate(
    coefficient_matrix: np.ndarray,
    right_hand_side: np.ndarray,
) -> np.ndarray:
    """Solve Ax = b using Gaussian elimination."""
    # Implementation here
    pass
```

```python
# main.py
from my_library import gauss_eliminate

result = gauss_eliminate(coefficient_matrix, right_hand_side)
```

---

## The Power Method

The **power method** finds the eigenvector corresponding to the largest eigenvalue of a matrix.

### Algorithm

1. Start with an initial guess $\mathbf{x}_0$ (must not be orthogonal to the true eigenvector)
2. Iterate:
   $$\mathbf{x}_{k+1} = \frac{\mathbf{A}\mathbf{x}_k}{\|\mathbf{A}\mathbf{x}_k\|}$$
3. Continue until convergence

### Why It Works

If $\mathbf{A}$ is diagonalisable with eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$ and eigenvalues $|\lambda_1| > |\lambda_2| \geq \ldots \geq |\lambda_n|$, then any vector can be written as:

$$\mathbf{x}_0 = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \ldots + c_n \mathbf{v}_n$$

Applying $\mathbf{A}$ repeatedly:

$$\mathbf{A}^k \mathbf{x}_0 = c_1 \lambda_1^k \mathbf{v}_1 + c_2 \lambda_2^k \mathbf{v}_2 + \ldots$$

$$= c_1 \lambda_1^k \left[ \mathbf{v}_1 + \frac{c_2}{c_1}\left(\frac{\lambda_2}{\lambda_1}\right)^k \mathbf{v}_2 + \ldots \right]$$

Since $|\lambda_1| > |\lambda_j|$ for $j > 1$, the ratios $(\lambda_j/\lambda_1)^k \to 0$ as $k \to \infty$.

### Eigenvalue Estimation

Once the eigenvector $\mathbf{v}$ converges, the eigenvalue can be estimated using the **Rayleigh quotient**:

$$\lambda = \frac{\mathbf{v}^T \mathbf{A} \mathbf{v}}{\mathbf{v}^T \mathbf{v}}$$

---

## Python Implementation

```python
"""Power method for finding dominant eigenvector."""

import numpy as np


def power_method(
    input_matrix: np.ndarray,
    max_iterations: int = 1000,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Find the dominant eigenvector and eigenvalue using the power method.
    
    Args:
        input_matrix: Square matrix.
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance.
    
    Returns:
        Tuple of (eigenvector, eigenvalue).
    """
    matrix_size = input_matrix.shape[0]
    current_vector = np.random.rand(matrix_size)
    current_vector = current_vector / np.linalg.norm(current_vector)
    
    for _ in range(max_iterations):
        next_vector = input_matrix @ current_vector
        next_vector = next_vector / np.linalg.norm(next_vector)
        
        if np.linalg.norm(next_vector - current_vector) < tolerance:
            break
        current_vector = next_vector
    
    # Rayleigh quotient for eigenvalue
    eigenvalue = (
        (current_vector.T @ input_matrix @ current_vector) /
        (current_vector.T @ current_vector)
    )
    
    return current_vector, eigenvalue


def main() -> None:
    """Demonstrate the power method."""
    test_matrix = np.array([
        [1, -1, -1, -1],
        [-1, 2, 0, 0],
        [-1, 0, 3, 1],
        [-1, 0, 1, 4],
    ], dtype=float)
    
    eigenvector, eigenvalue = power_method(test_matrix)
    
    print(f"Dominant eigenvalue: {eigenvalue:.6f}")
    print(f"Eigenvector: {eigenvector}")
    
    # Verify: A @ v should equal lambda * v
    residual = test_matrix @ eigenvector - eigenvalue * eigenvector
    print(f"Verification (should be ~0): {np.linalg.norm(residual):.2e}")


if __name__ == "__main__":
    main()
```

---

## Practice Exercises

### Exercise 1: Power Method

Find the largest eigenvector of:

$$\mathbf{A} = \begin{pmatrix}
1 & -1 & -1 & -1 \\
-1 & 2 & 0 & 0 \\
-1 & 0 & 3 & 1 \\
-1 & 0 & 1 & 4
\end{pmatrix}$$

### Exercise 2: General Least Squares Review

Using the general least squares method from Session 4:

1. Fit a quadratic $y = a_0 + a_1 x + a_2 x^2$ to data of your choice
2. Verify the result using a plotting library
3. Compare with a built-in fitting function (e.g., `numpy.polyfit`)

### Exercise 3: Interpolation Review

1. Generate data from $f(x) = \sin(x)$ at 5 equally spaced points in $[0, \pi]$
2. Use Lagrange interpolation to estimate $f(\pi/4)$
3. Compare with the true value and discuss the error

---

## Summary

| Technique | Purpose |
|-----------|---------|
| Code libraries | Reuse functions across projects |
| Power method | Find dominant eigenvalue/eigenvector |
| Rayleigh quotient | Estimate eigenvalue from eigenvector |

---

## Further Reading

Before Session 8, review:

- Root finding methods (Numerical Recipes, Chapter 9)
- Optimisation techniques (Chapra and Canale, Chapter 7)
