# Session 2: Gaussian Elimination

> Matt Watkins · <mwatkins@lincoln.ac.uk> (edited by William Fayers)

---

## Overview

This session introduces systems of linear equations and methods for solving them.

## Learning Outcomes

- Recall methods of solution of inhomogeneous systems of linear equations.
- Understand elimination methods: Gauss elimination and Gauss-Jordan elimination.
- Implement and use Gauss-Jordan elimination to solve systems of equations.

## Recommended Reading

- Chapra and Canale, Introduction to Part 3 and Chapter 9

---

## Solving a System of Equations

Suppose we want to solve a system of equations $\mathbf{Ax} = \mathbf{b}$ where:

- $\mathbf{A}$ is a matrix of coefficients of our unknowns $\mathbf{x}$
- $\mathbf{b}$ is a vector of constants

For a set of 4 equations with 4 unknowns:

$$
\begin{pmatrix}
a_{00} & a_{01} & a_{02} & a_{03} \\
a_{10} & a_{11} & a_{12} & a_{13} \\
a_{20} & a_{21} & a_{22} & a_{23} \\
a_{30} & a_{31} & a_{32} & a_{33}
\end{pmatrix}
\begin{pmatrix}
x_0 \\ x_1 \\ x_2 \\ x_3
\end{pmatrix}
=
\begin{pmatrix}
b_0 \\ b_1 \\ b_2 \\ b_3
\end{pmatrix}
$$

It is useful to rewrite this as an **augmented matrix**:

$$
\left(\begin{array}{cccc|c}
a_{00} & a_{01} & a_{02} & a_{03} & b_0 \\
a_{10} & a_{11} & a_{12} & a_{13} & b_1 \\
a_{20} & a_{21} & a_{22} & a_{23} & b_2 \\
a_{30} & a_{31} & a_{32} & a_{33} & b_3
\end{array}\right)
$$

---

## Gaussian Elimination

### Step 1: Triangularisation

Reduce the augmented matrix to **row echelon form**.

**Initial augmented matrix:**

$$
\left(\begin{array}{cccc|c}
2 & 2 & 4 & -2 & 10 \\
1 & 3 & 2 & 4 & 17 \\
3 & 1 & 3 & 1 & 18 \\
1 & 3 & 4 & 2 & 27
\end{array}\right)
$$

**Pivoting around row 0** (eliminate all entries below the diagonal in column 0):

$$
\left(\begin{array}{cccc|c}
2 & 2 & 4 & -2 & 10 \\
0 & 2 & 0 & 5 & 12 \\
0 & -2 & -3 & 4 & 3 \\
0 & 2 & 2 & 3 & 22
\end{array}\right)
$$

**Pivoting around row 1:**

$$
\left(\begin{array}{cccc|c}
2 & 2 & 4 & -2 & 10 \\
0 & 2 & 0 & 5 & 12 \\
0 & 0 & -3 & 9 & 15 \\
0 & 0 & 2 & -2 & 10
\end{array}\right)
$$

**Pivoting around row 2:**

$$
\left(\begin{array}{cccc|c}
2 & 2 & 4 & -2 & 10 \\
0 & 2 & 0 & 5 & 12 \\
0 & 0 & -3 & 9 & 15 \\
0 & 0 & 0 & 4 & 20
\end{array}\right)
$$

### Step 2: Back Substitution

From the triangularised matrix, solve for $\mathbf{x}$ starting from the last row:

**Row 3:** $4x_3 = 20 \implies x_3 = 5$

**Row 2:** $-3x_2 + 9(5) = 15 \implies x_2 = 10$

**Row 1:** $2x_1 + 0(10) + 5(5) = 12 \implies x_1 = -6.5$

**Row 0:** $2x_0 + 2(-6.5) + 4(10) + (-2)(5) = 10 \implies x_0 = -3.5$

**Solution:** $\mathbf{x} = (-3.5, -6.5, 10, 5)^T$

---

## Gauss-Jordan Elimination

Gauss-Jordan elimination extends Gaussian elimination by reducing to **reduced row echelon form** — a completely diagonal matrix.

The key difference: when pivoting, eliminate entries **above and below** the diagonal, not just below.

**After full elimination and row scaling:**

$$
\left(\begin{array}{cccc|c}
1 & 0 & 0 & 0 & -3.5 \\
0 & 1 & 0 & 0 & -6.5 \\
0 & 0 & 1 & 0 & 10 \\
0 & 0 & 0 & 1 & 5
\end{array}\right)
$$

The solutions can be read directly from the augmented column.

---

## Gauss-Jordan for Matrix Inversion

To find $\mathbf{A}^{-1}$, solve $\mathbf{AX} = \mathbf{I}$:

**Initial augmented matrix:**

$$
\left(\begin{array}{ccc|ccc}
2 & 1 & 1 & 1 & 0 & 0 \\
1 & 0 & -1 & 0 & 1 & 0 \\
2 & -1 & 2 & 0 & 0 & 1
\end{array}\right)
$$

**After Gauss-Jordan elimination:**

$$
\left(\begin{array}{ccc|ccc}
1 & 0 & 0 & 0.143 & 0.429 & 0.143 \\
0 & 1 & 0 & 0.571 & -0.286 & -0.429 \\
0 & 0 & 1 & 0.143 & -0.571 & 0.143
\end{array}\right)
$$

The right-hand side is $\mathbf{A}^{-1}$.

---

## Exercises

### Exercise 1: Solve Systems

Use your code to find the solutions of:

**System 1:**

$$
\begin{aligned}
3x_0 + 4x_1 - 7x_2 &= 23 \\
7x_0 - x_1 + 2x_2 &= 14 \\
x_0 + 10x_1 - 2x_2 &= 33
\end{aligned}
$$

**System 2:**

$$
\begin{aligned}
x_0 + 2x_1 + 3x_2 &= 1 \\
4x_0 + 5x_1 + 6x_2 &= 2 \\
7x_0 + 8x_1 + 9x_2 &= 3
\end{aligned}
$$

Can you find the solutions to System 2? Why or why not?

### Exercise 2: Gauss-Jordan Implementation

1. Copy your Gauss elimination code and modify it for Gauss-Jordan elimination.
2. Change the second loop to iterate over all rows.
3. Add an `if` statement to skip the pivot row.
4. When the matrix is diagonal, divide each row by its diagonal element.
5. Test regularly as you make alterations.

### Exercise 3: Matrix Inversion

Find the inverse of:

$$
\begin{pmatrix}
2 & 2 & 4 & -2 \\
1 & 3 & 2 & 4 \\
3 & 1 & 3 & 1 \\
1 & 3 & 4 & 2
\end{pmatrix}
$$

Check your solution by multiplying the original and inverse matrices.

---

## Summary

| Method | Output | Complexity |
| ------ | ------ | ---------- |
| Gauss Elimination | Row echelon form + back substitution | $O(n^3)$ |
| Gauss-Jordan Elimination | Reduced row echelon form (direct solution) | $O(n^3)$ |
| Matrix Inversion | $\mathbf{A}^{-1}$ via augmented identity | $O(n^3)$ |

---

## Further Reading

Before next session, read about:

- Pivoting strategies to improve numerical stability
- LU decomposition of square matrices (Chapra and Canale, Chapter 10)
