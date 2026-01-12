# Session 3: Polynomial and Multiple Regression

> **MTH3007 Numerical Methods** · Matt Watkins · <mwatkins@lincoln.ac.uk>

---

## Overview

This session extends linear regression to polynomial fitting and multiple independent variables.

## Learning Outcomes

- Extend the work on linear regression to polynomial and multiple variables.
- Combine programming with analytical solutions.
- Check your code works correctly via an external reference.

## Recommended Reading

- Chapra and Canale, Introduction to Part 5 and Chapter 17
- Numerical Recipes, Chapters 14–15

---

## Recap: Linear Least Squares

From Session 1, given a straight line model:

$$y_i = a_0 + a_1 x_i + e_i$$

The sum of squared errors is:

$$S_r = \sum_{i=0}^{n-1}(y_i - a_0 - a_1 x_i)^2$$

The optimal parameters are:

$$a_1 = \frac{n \sum x_i y_i - \sum x_i \sum y_i}{n \sum x_i^2 - (\sum x_i)^2}$$

$$a_0 = \bar{y} - a_1 \bar{x}$$

### Correlation Coefficient

The correlation coefficient measures how well the data fits a linear model:

$$r = \frac{n \sum x_i y_i - (\sum x_i)(\sum y_i)}{\sqrt{n \sum x_i^2 - (\sum x_i)^2} \sqrt{n \sum y_i^2 - (\sum y_i)^2}}$$

Values close to $\pm 1$ indicate strong linear correlation.

---

## Polynomial Least Squares Regression

We can extend our method to polynomials:

$$y_i = a_0 + a_1 x_i + a_2 x_i^2 + \ldots + a_m x_i^m + e_i$$

The error function becomes:

$$S_r = \sum_{i=0}^{n-1} \left(y_i - \sum_{j=0}^{m} a_j x_i^j\right)^2$$

Taking partial derivatives with respect to each parameter and setting them to zero gives a system of equations.

---

## Fitting a Quadratic

For a quadratic model:

$$y_i = a_0 + a_1 x_i + a_2 x_i^2 + e_i$$

The error function is:

$$S_r = \sum_{i=0}^{n-1} (y_i - a_0 - a_1 x_i - a_2 x_i^2)^2$$

Setting $\frac{\partial S_r}{\partial a_k} = 0$ for $k = 0, 1, 2$ gives:

$$n \cdot a_0 + (\sum x_i) a_1 + (\sum x_i^2) a_2 = \sum y_i$$

$$(\sum x_i) a_0 + (\sum x_i^2) a_1 + (\sum x_i^3) a_2 = \sum x_i y_i$$

$$(\sum x_i^2) a_0 + (\sum x_i^3) a_1 + (\sum x_i^4) a_2 = \sum x_i^2 y_i$$

### Matrix Form

$$
\begin{pmatrix}
n & \sum x_i & \sum x_i^2 \\
\sum x_i & \sum x_i^2 & \sum x_i^3 \\
\sum x_i^2 & \sum x_i^3 & \sum x_i^4
\end{pmatrix}
\begin{pmatrix}
a_0 \\ a_1 \\ a_2
\end{pmatrix}
=
\begin{pmatrix}
\sum y_i \\
\sum x_i y_i \\
\sum x_i^2 y_i
\end{pmatrix}
$$

This is of the form $\mathbf{Ax} = \mathbf{b}$ and can be solved using Gaussian elimination.

---

## Multiple Linear Regression

Instead of powers of a single variable, our model may involve several independent variables:

$$y_i = a_0 + a_1 x_{1,i} + a_2 x_{2,i} + \ldots + a_m x_{m,i} + e_i$$

### Example: Two Independent Variables

$$y_i = a_0 + a_1 x_{1,i} + a_2 x_{2,i} + e_i$$

The normal equations become:

$$n \cdot a_0 + (\sum x_{1,i}) a_1 + (\sum x_{2,i}) a_2 = \sum y_i$$

$$(\sum x_{1,i}) a_0 + (\sum x_{1,i}^2) a_1 + (\sum x_{1,i} x_{2,i}) a_2 = \sum x_{1,i} y_i$$

$$(\sum x_{2,i}) a_0 + (\sum x_{1,i} x_{2,i}) a_1 + (\sum x_{2,i}^2) a_2 = \sum x_{2,i} y_i$$

---

## Linearisation of Non-Linear Data

Multiple linear regression can handle data with non-linear relationships through transformation.

**Example:** For data following $y = a_0 x_1^{a_1} x_2^{a_2}$

Take logarithms:

$$\ln(y) = \ln(a_0) + a_1 \ln(x_1) + a_2 \ln(x_2)$$

Let $Y = \ln(y)$, $A_0 = \ln(a_0)$, $X_1 = \ln(x_1)$, $X_2 = \ln(x_2)$:

$$Y = A_0 + a_1 X_1 + a_2 X_2$$

This is now a linear regression problem!

---

## Summary

| Model Type | Form | Parameters |
| ---------- | ---- | ---------- |
| Linear | $y = a_0 + a_1 x$ | 2 |
| Quadratic | $y = a_0 + a_1 x + a_2 x^2$ | 3 |
| Polynomial (degree $m$) | $y = \sum_{j=0}^{m} a_j x^j$ | $m+1$ |
| Multiple linear | $y = a_0 + \sum_{j=1}^{m} a_j x_j$ | $m+1$ |
