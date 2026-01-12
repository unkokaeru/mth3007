# Session 3: Questions

> **MTH3007 Numerical Methods**

---

## Task 1: Extending Sum Calculations

Recall the sum calculations from Session 1. Extend your code to calculate the following.

Let $x_i = i$ and $y_i = 2i + 0.3$ for $i = 1, 2, \ldots, 10$.

1. Calculate $\sum_i x_i$
2. Calculate $\sum_i y_i$
3. Calculate $\sum_i x_i y_i$
4. Calculate $\bar{x}$ and $\bar{y}$
5. Find the parameters $a_0$ and $a_1$ for a linear regression model
6. Verify your results using matplotlib or Excel

---

## Task 2: Deriving Quadratic Regression

When fitting a quadratic, the individual error is:

$$e_i = y_i - a_0 - a_1 x_i - a_2 x_i^2$$

Let $f_i = a_0 + a_1 x_i + a_2 x_i^2$. The total error function is:

$$S = \sum_{i=0}^{n-1} e_i^2 = \sum_{i=0}^{n-1} (y_i - f_i)^2$$

**i)** Find all partial derivatives $\frac{\partial f_i}{\partial a_k}$ for $k = 0, 1, 2$.

**ii)** Complete the derivation for $\frac{\partial S}{\partial a_k}$:

$$\frac{\partial S}{\partial a_k} = \frac{\partial}{\partial a_k} \sum_{i=0}^{n-1} (y_i - f_i)^2 = -2\sum_{i=0}^{n-1} (y_i - f_i) \frac{\partial f_i}{\partial a_k}$$

Set the result equal to zero and derive the normal equations.

**iii)** Write out the three equations formed by letting $k = 0, 1, 2$.

**iv)** Express the system as a matrix equation $\mathbf{Ax} = \mathbf{b}$.

**v)** Using data from Blackboard and your Gaussian elimination code, solve for $a_0$, $a_1$, and $a_2$.

---

## Task 3: Multiple Linear Regression

For a model with multiple independent variables:

$$y_i = a_0 + a_1 x_{1,i} + a_2 x_{2,i} + e_i$$

**i)** Derive the normal equations by minimising:

$$S = \sum_{i=0}^{n-1} (y_i - a_0 - a_1 x_{1,i} - a_2 x_{2,i})^2$$

**ii)** Use the following data to solve for $a_0$, $a_1$, and $a_2$:

| $x_1$ | $x_2$ | $y$  |
|-------|-------|------|
| 0     | 0     | 5    |
| 2     | 1     | 10   |
| 2.5   | 2     | 9    |
| 1     | 3     | 0    |
| 4     | 6     | 3    |
| 7     | 2     | 27   |

**iii)** Plot your fitted function against the data to verify the fit.

---

## Task 4: Linearisation

How would you apply multiple linear regression to data that follows:

$$y = a_0 x_1^{a_1} x_2^{a_2} \cdots x_n^{a_n}$$

*Hint: Consider taking logarithms of both sides.*

---

## Further Reading

- Chapra and Canale, Introduction to Part 5 and Chapter 17
- Numerical Recipes, Chapters 14–15
