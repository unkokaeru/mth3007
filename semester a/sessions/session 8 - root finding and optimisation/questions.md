# Session 8: Questions

> **MTH3007 Numerical Methods**

---

## Task 1: Newton's Root-Finding Method

Newton's method uses the iteration:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

Use this to find all roots of:

**i)** $x^2 = 612$

*Hint: Rewrite as $f(x) = x^2 - 612 = 0$, so $f'(x) = 2x$.*

**ii)** $x^3 = \cos(x)$

*Hint: Rewrite as $f(x) = x^3 - \cos(x) = 0$.*

---

## Task 2: Secant Method

The secant method uses:

$$x_{n+1} = \frac{x_{n-1} f(x_n) - x_n f(x_{n-1})}{f(x_n) - f(x_{n-1})}$$

This requires two initial guesses.

**Tasks:**

**i)** Find the roots of $x^2 = 612$ using the secant method.

**ii)** Find the roots of $x^3 = \cos(x)$ using the secant method.

**iii)** Compare the number of iterations required for Newton's method versus the secant method to achieve the same accuracy. Implement a convergence check based on $|x_{n+1} - x_n|$.

---

## Task 3: Newton's Method for Optimisation

Newton's method for optimisation finds extrema by iterating:

$$x_{k+1} = x_k - \frac{f'(x_k)}{f''(x_k)}$$

### Derivation

Starting from the second-order Taylor expansion:

$$f(x_{k+1}) \approx f(x_k) + f'(x_k)(x_{k+1} - x_k) + \frac{1}{2}f''(x_k)(x_{k+1} - x_k)^2$$

To find extrema, differentiate this approximation with respect to $x_{k+1}$ and set equal to zero:

$$\frac{d}{dx_{k+1}}\left[f(x_k) + f'(x_k)(x_{k+1} - x_k) + \frac{1}{2}f''(x_k)(x_{k+1} - x_k)^2\right] = 0$$

**Tasks:**

**i)** Complete the derivation to obtain the iteration formula.

**ii)** Use Newton's method for optimisation to find all extrema of:

- $f(x) = x^2 - \cos(x)$
- $f(x) = x^4 - 14x^3 + 60x^2 - 70x$

**iii)** Classify each extremum as a minimum or maximum by examining $f''(x)$ at that point.

---

## Further Reading

- Numerical Recipes, Chapter 9
- Chapra and Canale, Chapters 5–7
