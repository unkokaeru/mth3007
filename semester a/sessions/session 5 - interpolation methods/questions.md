# Session 5: Questions

> **MTH3007 Numerical Methods**

---

## Task 1: Polynomial Interpolation

### Newton's Divided Difference

The Newton interpolating polynomial is:

$$P(x) = f(x_0) + (x - x_0)f[x_0, x_1] + (x - x_0)(x - x_1)f[x_0, x_1, x_2] + \ldots$$

where:

$$f[x_i, x_{i+1}] = \frac{f(x_{i+1}) - f(x_i)}{x_{i+1} - x_i}$$

$$f[x_i, \ldots, x_{i+k}] = \frac{f[x_{i+1}, \ldots, x_{i+k}] - f[x_i, \ldots, x_{i+k-1}]}{x_{i+k} - x_i}$$

### Lagrange Interpolation

$$p(x) = \sum_{i=0}^{n} \left( \prod_{\substack{j=0 \\ j \neq i}}^{n} \frac{x - x_j}{x_i - x_j} \right) y_i$$

**Tasks:**

**i)** Determine the interpolating polynomial for the data $(0, 1)$, $(1, 3)$, $(3, 55)$ using Newton's divided difference method.

**ii)** Determine the interpolating polynomial for the same data using Lagrange interpolation. Verify that both methods give the same polynomial.

**iii)** Write a program to perform Lagrange interpolation on arbitrary coordinate data.

---

## Task 2: Bilinear Interpolation

For a function $f(x, y)$ known at four corners, bilinear interpolation proceeds in two steps:

**Step 1:** Interpolate in $x$ at $y = y_1$ and $y = y_2$:

$$f(x, y_1) = \frac{x_2 - x}{x_2 - x_1} f(x_1, y_1) + \frac{x - x_1}{x_2 - x_1} f(x_2, y_1)$$

$$f(x, y_2) = \frac{x_2 - x}{x_2 - x_1} f(x_1, y_2) + \frac{x - x_1}{x_2 - x_1} f(x_2, y_2)$$

**Step 2:** Interpolate in $y$:

$$f(x, y) = \frac{y_2 - y}{y_2 - y_1} f(x, y_1) + \frac{y - y_1}{y_2 - y_1} f(x, y_2)$$

**Tasks:**

**i)** Use bilinear interpolation to predict $T(5.25, 4.8)$ using the following temperature data:

| $x$ | $y$ | $T(x, y)$ |
|-----|-----|-----------|
| 2   | 2   | 60        |
| 2   | 6   | 55        |
| 9   | 1   | 57.5      |
| 9   | 6   | 70        |

*Note: You'll need to identify which corners correspond to which $(x_1, y_1)$, $(x_2, y_2)$ positions.*

**ii)** Generalise your program to conduct bilinear interpolation. Test your code using $f(x, y) = xy$ and verify by evaluating the function directly.

---

## Further Reading

- Chapra and Canale, Chapter 18
- Numerical Recipes, Chapter 3
