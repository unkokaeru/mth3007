# Session 10: Questions

> **MTH3007 Numerical Methods**

---

## Task 1: Computing Fourier Coefficients

Use numerical integration to calculate the Fourier coefficients $a_n$ and $b_n$ for the following periodic functions defined on one period $[-L, L]$:

**i) Square wave:**

$$f(x) = \begin{cases}
1 & 0 \leq x < L \\
-1 & -L \leq x < 0
\end{cases}$$

**ii) Triangle wave:**

$$f(x) = |x| \quad \text{for } -L \leq x \leq L$$

**iii) Sawtooth wave:**

$$f(x) = x \quad \text{for } -L \leq x < L$$

**For each function:**

a) Calculate the first 10 Fourier coefficients ($a_0$ to $a_9$ and $b_1$ to $b_9$).

b) Plot the original function and the Fourier series reconstruction using 5, 10, and 20 terms.

c) Verify Parseval's theorem:
$$\int_{-L}^{L} |f(x)|^2 \, dx = L\left[\frac{a_0^2}{2} + \sum_{n=1}^{N} (a_n^2 + b_n^2)\right]$$

---

## Task 2: Non-Periodic Functions

The Fourier series can be applied to non-periodic functions on a finite interval, though boundary behaviour becomes important.

**i) Gaussian function:**

$$f(x) = e^{-x^2} \quad \text{on } [-5, 5]$$

a) Calculate the Fourier coefficients numerically.

b) Plot the original function and reconstructions with 5, 10, 20, and 50 terms.

c) Discuss: Why do smooth functions like the Gaussian converge faster than functions with discontinuities?

**ii) Step function:**

$$f(x) = \begin{cases}
1 & |x| \leq 1 \\
0 & |x| > 1
\end{cases} \quad \text{on } [-5, 5]$$

a) Calculate the Fourier coefficients numerically.

b) Observe the **Gibbs phenomenon** — the overshoot near discontinuities that persists regardless of the number of terms.

c) How many terms are needed for a reasonable approximation away from the discontinuities?

---

## Task 3: Analytical Fourier Coefficients (Extension)

For the square wave in Task 1(i), derive the Fourier coefficients analytically:

**i)** Show that $a_n = 0$ for all $n$ (the function is odd).

**ii)** Compute $b_n$ by evaluating:
$$b_n = \frac{1}{L}\left[\int_{-L}^{0} (-1) \sin\frac{n\pi x}{L} \, dx + \int_{0}^{L} (1) \sin\frac{n\pi x}{L} \, dx\right]$$

**iii)** Show that:
$$b_n = \begin{cases}
\frac{4}{n\pi} & n \text{ odd} \\
0 & n \text{ even}
\end{cases}$$

**iv)** Compare your analytical results with the numerical coefficients from Task 1.

---

## Further Reading

- Chapra and Canale, Chapter 19
- Numerical Recipes, Chapter 12
