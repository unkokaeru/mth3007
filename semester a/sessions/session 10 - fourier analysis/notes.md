# Session 10: Fourier Analysis

> **MTH3007 Numerical Methods** · Matt Watkins · <mwatkins@lincoln.ac.uk>

---

## Overview

This session covers Fourier series and their numerical computation, including orthogonal function expansions and Parseval's theorem.

## Learning Outcomes

- Understand Fourier series as expansions in orthogonal functions.
- Compute Fourier coefficients numerically.
- Apply Parseval's theorem to verify results.

## Recommended Reading

- Chapra and Canale, Chapter 19
- Numerical Recipes, Chapter 12

---

## Fourier Series

Any periodic function $f(x)$ with period $2L$ (i.e., $f(x) = f(x + 2L)$) can be expanded as:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos\left(\frac{n\pi x}{L}\right) + b_n \sin\left(\frac{n\pi x}{L}\right) \right]$$

The cosines and sines are **orthogonal basis functions**.

---

## Orthogonality Relations

The basis functions satisfy:

$$\int_{-L}^{L} \cos\left(\frac{n\pi x}{L}\right) \cos\left(\frac{m\pi x}{L}\right) dx =
\begin{cases}
2L & n = m = 0 \\
L & n = m \neq 0 \\
0 & n \neq m
\end{cases}$$

$$\int_{-L}^{L} \sin\left(\frac{n\pi x}{L}\right) \sin\left(\frac{m\pi x}{L}\right) dx =
\begin{cases}
0 & n = 0 \text{ or } m = 0 \\
L & n = m \\
0 & n \neq m
\end{cases}$$

$$\int_{-L}^{L} \cos\left(\frac{n\pi x}{L}\right) \sin\left(\frac{m\pi x}{L}\right) dx = 0 \quad \text{for all } n, m$$

Using the Kronecker delta $\delta_{nm}$:

$$\int_{-L}^{L} \cos\left(\frac{n\pi x}{L}\right) \cos\left(\frac{m\pi x}{L}\right) dx = L(1 + \delta_{n0})\delta_{nm}$$

---

## Fourier Coefficients

Multiply the series by a basis function and integrate. Orthogonality eliminates all but one term:

$$a_n = \frac{1}{L(1 + \delta_{n0})} \int_{-L}^{L} f(x) \cos\left(\frac{n\pi x}{L}\right) dx$$

$$b_n = \frac{1}{L} \int_{-L}^{L} f(x) \sin\left(\frac{n\pi x}{L}\right) dx$$

For $n = 0$:

$$a_0 = \frac{1}{L} \int_{-L}^{L} f(x) \, dx$$

(The factor of $\frac{1}{2}$ in the series compensates for the $2L$ normalisation.)

---

## Numerical Computation

Use the **trapezoidal rule** for numerical integration:

$$\int_a^b g(x) \, dx \approx h\left(\frac{g_0}{2} + g_1 + g_2 + \cdots + g_{N-1} + \frac{g_N}{2}\right)$$

where $h = (b - a)/N$.

For Fourier coefficients:

$$a_n \approx \frac{2}{N(1 + \delta_{n0})} \left(\frac{f_{-L}\cos(0)}{2} + \sum_{k=1}^{N-1} f_k \cos\left(\frac{n\pi x_k}{L}\right) + \frac{f_L\cos(n\pi)}{2}\right)$$

For sine coefficients, $\sin(n\pi x/L)$ is zero at $x = \pm L$, so the boundary terms vanish:

$$b_n \approx \frac{2}{N} \sum_{k=1}^{N-1} f_k \sin\left(\frac{n\pi x_k}{L}\right)$$

---

## Parseval's Theorem

The integral of $|f(x)|^2$ equals the sum of squared coefficients:

$$\int_{-L}^{L} |f(x)|^2 \, dx = L\left[\frac{a_0^2}{2} + \sum_{n=1}^{\infty} (a_n^2 + b_n^2)\right]$$

### Physical Interpretation

- Left side: Total "energy" in the function
- Right side: Sum of energies in each frequency component

This provides a useful check on numerical Fourier coefficient calculations.

---

## Complex Exponential Form

Using Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$:

$$f(x) = \sum_{n=-\infty}^{\infty} c_n e^{in\pi x/L}$$

where:

$$c_n = \frac{1}{2L} \int_{-L}^{L} f(x) e^{-in\pi x/L} \, dx$$

Relationship to real coefficients:

$$c_0 = \frac{a_0}{2}, \quad c_n = \frac{a_n - ib_n}{2}, \quad c_{-n} = \frac{a_n + ib_n}{2}$$

---

## Python Implementation

```python
"""Numerical Fourier series computation."""

import numpy as np
import matplotlib.pyplot as plt


def compute_fourier_coefficients(
    f: callable,
    L: float,
    N_points: int,
    N_terms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Fourier coefficients using trapezoidal integration.
    
    Args:
        f: Function to analyse.
        L: Half-period (function defined on [-L, L]).
        N_points: Number of integration points.
        N_terms: Number of Fourier terms to compute.
    
    Returns:
        Tuple of (a_coefficients, b_coefficients).
    """
    x = np.linspace(-L, L, N_points + 1)
    h = x[1] - x[0]
    f_vals = f(x)
    
    a = np.zeros(N_terms)
    b = np.zeros(N_terms)
    
    for n in range(N_terms):
        cos_vals = np.cos(n * np.pi * x / L)
        sin_vals = np.sin(n * np.pi * x / L)
        
        # Trapezoidal rule for a_n
        integrand = f_vals * cos_vals
        integral = h * (0.5 * integrand[0] + np.sum(integrand[1:-1]) + 0.5 * integrand[-1])
        a[n] = integral / (L * (1 + (n == 0)))
        
        # Trapezoidal rule for b_n (boundary terms are zero for sine)
        integrand = f_vals * sin_vals
        b[n] = h * np.sum(integrand[1:-1]) / L
    
    return a, b


def reconstruct_fourier(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    L: float,
) -> np.ndarray:
    """Reconstruct function from Fourier coefficients.
    
    Args:
        x: Points at which to evaluate.
        a: Cosine coefficients.
        b: Sine coefficients.
        L: Half-period.
    
    Returns:
        Reconstructed function values.
    """
    result = np.zeros_like(x)
    for n in range(len(a)):
        result += a[n] * np.cos(n * np.pi * x / L)
        result += b[n] * np.sin(n * np.pi * x / L)
    return result


def verify_parseval(
    f: callable,
    a: np.ndarray,
    b: np.ndarray,
    L: float,
    N_points: int,
) -> tuple[float, float]:
    """Verify Parseval's theorem.
    
    Args:
        f: Original function.
        a: Cosine coefficients.
        b: Sine coefficients.
        L: Half-period.
        N_points: Integration points.
    
    Returns:
        Tuple of (LHS integral, RHS sum of squares).
    """
    x = np.linspace(-L, L, N_points + 1)
    h = x[1] - x[0]
    f_vals = f(x)
    
    # LHS: integral of |f|^2
    integrand = f_vals**2
    lhs = h * (0.5 * integrand[0] + np.sum(integrand[1:-1]) + 0.5 * integrand[-1])
    
    # RHS: L * (a0^2/2 + sum(an^2 + bn^2))
    rhs = L * (a[0]**2 / 2 + np.sum(a[1:]**2 + b[1:]**2))
    
    return lhs, rhs


def main() -> None:
    """Demonstrate Fourier series computation."""
    L = np.pi
    N_points = 200
    N_terms = 20
    
    # Square wave
    def square_wave(x):
        return np.where(x >= 0, 1.0, -1.0)
    
    a, b = compute_fourier_coefficients(square_wave, L, N_points, N_terms)
    
    print("Square wave Fourier coefficients:")
    print(f"a_0 = {a[0]:.6f}")
    for n in range(1, min(5, N_terms)):
        print(f"a_{n} = {a[n]:.6f}, b_{n} = {b[n]:.6f}")
    
    # Verify Parseval's theorem
    lhs, rhs = verify_parseval(square_wave, a, b, L, N_points)
    print(f"\nParseval verification: LHS = {lhs:.6f}, RHS = {rhs:.6f}")
    
    # Plot reconstruction
    x_plot = np.linspace(-L, L, 500)
    y_original = square_wave(x_plot)
    y_reconstructed = reconstruct_fourier(x_plot, a, b, L)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_original, "b-", label="Original", linewidth=2)
    plt.plot(x_plot, y_reconstructed, "r--", label=f"Fourier ({N_terms} terms)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Fourier Series Approximation")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
```

---

## Applications

| Application | Use of Fourier Analysis |
|-------------|------------------------|
| Signal processing | Decompose into frequency components |
| Image compression | JPEG uses discrete cosine transform |
| Heat equation | Separate variables in PDEs |
| Vibrations | Analyse resonant frequencies |
| Quantum mechanics | Momentum space representation |

---

## Summary

| Concept | Formula |
|---------|---------|
| Fourier series | $f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[a_n \cos\frac{n\pi x}{L} + b_n \sin\frac{n\pi x}{L}\right]$ |
| Cosine coefficient | $a_n = \frac{1}{L} \int_{-L}^{L} f(x) \cos\frac{n\pi x}{L} \, dx$ (for $n > 0$) |
| Sine coefficient | $b_n = \frac{1}{L} \int_{-L}^{L} f(x) \sin\frac{n\pi x}{L} \, dx$ |
| Parseval's theorem | $\int_{-L}^{L} \|f(x)\|^2 dx = L\left[\frac{a_0^2}{2} + \sum_{n=1}^{\infty}(a_n^2 + b_n^2)\right]$ |
