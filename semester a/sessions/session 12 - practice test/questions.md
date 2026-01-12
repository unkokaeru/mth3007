# Practice Test Questions

## Question 1

Perform linear least squares regression on the following data:

| x        | y           |
| -------- | ----------- |
| 0.000000 | 85.358681   |
| 1.428571 | 14.165804   |
| 2.857143 | -26.825715  |
| 4.285714 | -4.997952   |
| 5.714286 | -94.922415  |
| 7.142857 | -164.033960 |
| 8.571429 | -180.876888 |
| 10.000000| -206.688112 |

Give the correlation coefficient for the data correct to 3 decimal places.
Include a plot of your model along with the original data.

---

## Question 2

Which of the following statements about the assumptions behind linear regression is **FALSE**?

- A) The y values are randomly distributed with the same variance.
- B) The y values of the data are normally distributed.
- C) All the data points are measured exactly.
- D) Each x value is precisely measured.

---

## Question 3

After fitting a model to data, what checks would you perform to confirm that this has been done correctly?

---

## Question 4

Find the sum of the solutions, $x_0 + x_1 + x_2$, of the following set of equations:

$$
\begin{bmatrix}
9.88147746930662 & 9.446294330943026 & 6.867343600899789 \\
3.3013023006376 & 8.866849207559063 & 2.5855821434029 \\
7.897613140999123 & 2.5810421630855136 & 9.843121360475344
\end{bmatrix}
\begin{bmatrix}
x_0 \\ x_1 \\ x_2
\end{bmatrix}
=
\begin{bmatrix}
1.7147395700538763 \\
3.82744793002092 \\
8.883242030765913
\end{bmatrix}
$$

---

## Question 5

Using Newton's method or the Secant method, find the root(s) of:

$$x^2 + 2.5x = 4.6875$$

Give a numerical answer.
Explain the Secant method and explain how it is related to Newton's method.

---

## Question 6

Solve the following differential equation using finite differences and solving the resulting set of linear equations:

$$q \frac{d^2H}{dg^2} + kH(g) = b$$

where $g \in [0, 5]$, $H_L = 5$, $H_R = 5$, $q = -3.0$, $k = 0$, $b = 6$.

Give the maximum value of $H(g)$ as your numerical answer.

Plot the function $H(g)$ in the interval $[0,5]$ using 10, 50, 100, 200 points when discretising the independent variable.

If $k = 5.538013$, how would you tackle the problem and how would the solution change?

---

## Question 7

The cosine coefficients in a Fourier series expansion of a periodic function on the interval $[-l, l)$ can be defined as:

$$a_m = \frac{1}{l} \int_{-l}^{l} \phi_m(x) f(x) \, dx, \quad \text{where } \phi_m(x) = \cos\left(\frac{m \pi x}{l}\right)$$

which can be approximated numerically as:

$$a_m \approx \frac{1}{l} \sum_{i=0}^{n} \phi_m(x_i) f(x_i) \, \Delta x$$

on a regular grid of values, $x_i$, of size $N$.

Calculate numerically $a_1$ (the coefficient of $\cos(2\pi x)$) in the Fourier series expansion of $f(x) = 4(x - 1/2)^2$ defined on $[0, 1)$.

Explain the numerical approximation to obtain $a_m$.