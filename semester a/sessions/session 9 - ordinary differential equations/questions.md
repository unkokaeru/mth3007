# Session 9: Questions

> **MTH3007 Numerical Methods**

---

## Task 1: Solving $\frac{d^2y}{dx^2} = g(x)$

To solve a second-order ODE of the form:

$$\frac{d^2 y}{dx^2} = g(x)$$

construct the following $n \times n$ matrix from finite differences:

$$\mathbf{M} = \begin{pmatrix}
-2 & 0 & 0 & \cdots & 0 \\
1 & -2 & 1 & \cdots & 0 \\
0 & \ddots & \ddots & \ddots & 0 \\
0 & \cdots & 1 & -2 & 1 \\
0 & \cdots & 0 & 0 & -2
\end{pmatrix}$$

The diagonal elements are $M_{ii} = -2$, and the off-diagonal elements are $M_{i,i\pm 1} = 1$ except for the first and last rows (which enforce boundary conditions).

The right-hand side vector is:

$$\mathbf{b} = \begin{pmatrix}
-2y_L \\
g(x_1)\Delta x^2 \\
\vdots \\
g(x_{n-2})\Delta x^2 \\
-2y_R
\end{pmatrix}$$

Solve $\mathbf{My} = \mathbf{b}$ using Gaussian elimination.

**Tasks:**

**i)** Solve for $g(x) = x$ with $y_L = 0.2$, $y_R = 1.5$ on $[0, 1]$ for:
- a) $n = 10$
- b) $n = 100$
- c) $n = 500$

Compare with the analytical solution $y = \frac{x^3}{6} + Ax + B$ (find $A$, $B$ from boundary conditions).

**ii)** Solve for a discrete delta function: $g(x_{n/2}) = 1/\Delta x$, all other $g(x_i) = 0$, with $y_L = 0$, $y_R = 0$ on $[-5, 5]$ for:
- a) $n = 10$
- b) $n = 100$
- c) $n = 500$

---

## Task 2: ODEs with $y$ Terms

For an ODE of the form:

$$\frac{d^2 y}{dx^2} + ky = g(x)$$

substitute the finite difference approximation:

$$\frac{y_{i-1} - 2y_i + y_{i+1}}{\Delta x^2} + ky_i = g(x_i)$$

Rearranging:

$$y_{i-1} + (k\Delta x^2 - 2)y_i + y_{i+1} = g(x_i)\Delta x^2$$

The matrix $\mathbf{M}$ now has diagonal elements $k\Delta x^2 - 2$ instead of $-2$.

**Tasks:**

**i)** Solve $\frac{d^2 y}{dx^2} - 3y = -1$ with $y_L = 5$, $y_R = 0$ on $[-3, 3]$ for $n = 500$.

*Note: Here $k = -3$ and $g(x) = -1$.*

**ii)** Generalise your code to solve any ODE of the form $\frac{d^2 y}{dx^2} + ky = c$ where $k$ and $c$ are constants.

---

## Task 3: QR Algorithm (Extension)

Use the QR algorithm to find eigenvalues of:

$$\mathbf{A} = \begin{pmatrix}
1 & -1 & -1 & -1 \\
-1 & 2 & 0 & 0 \\
-1 & 0 & 3 & 1 \\
-1 & 0 & 1 & 4
\end{pmatrix}$$

**Tasks:**

**i)** Implement the QR algorithm and find all eigenvalues.

**ii)** Verify each eigenvalue by solving $(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$ to find the corresponding eigenvector.

**iii)** Confirm by checking that $\mathbf{Av} = \lambda\mathbf{v}$.

---

## Further Reading

- Numerical Recipes, Chapters 2, 11
- Chapra and Canale, Chapters 10, 27
