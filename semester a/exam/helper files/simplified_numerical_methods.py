"""MTH3007 Simplified Numerical Methods - Quick Reference with Worked Examples.

Each section shows a SPECIFIC EXAMPLE with step-by-step calculation.
Numbers are commented to show where they come from (question data, calculations, etc.)

Author: William Fayers
"""

import numpy as np

# =============================================================================
# 1. GAUSSIAN ELIMINATION
# =============================================================================
# EXAMPLE: Solve the system:
#   2x + y = 4
#   x + 3y = 5

print("=" * 60)
print("1. GAUSSIAN ELIMINATION")
print("=" * 60)
print("Solve: 2x + y = 4, x + 3y = 5")

# From the question:
A = np.array([
    [2, 1],  # coefficients from equation 1: 2x + 1y
    [1, 3]   # coefficients from equation 2: 1x + 3y
], dtype=float)
b = np.array([4, 5], dtype=float)  # right-hand sides from equations

# Create augmented matrix [A|b]
augmented = np.column_stack([A, b])
print(f"Augmented matrix:\n{augmented}")

# STEP 1: Forward elimination - eliminate x from row 2
multiplier = augmented[1, 0] / augmented[0, 0]  # = 1/2 = 0.5
print(f"Multiplier m = {augmented[1,0]}/{augmented[0,0]} = {multiplier}")
augmented[1] = augmented[1] - multiplier * augmented[0]
print(f"After elimination:\n{augmented}")

# STEP 2: Back substitution
y = augmented[1, 2] / augmented[1, 1]  # y = 3/2.5 = 1.2... wait let me recalc
# Actually: row 2 is now [0, 2.5, 3], so y = 3/2.5 = 1.2
# Then x = (4 - 1*y)/2 = (4 - 1.2)/2 = 1.4
# Hmm, let me verify with numpy
print(f"Solution: x = {np.linalg.solve(A, b)}")


# =============================================================================
# 2. LINEAR REGRESSION
# =============================================================================
# EXAMPLE: Fit y = a0 + a1*x to data points:
#   x: 1, 2, 3, 4, 5
#   y: 2.1, 4.0, 5.9, 8.1, 9.9

print("\n" + "=" * 60)
print("2. LINEAR REGRESSION")
print("=" * 60)
print("Fit y = a0 + a1*x to: x=[1,2,3,4,5], y=[2.1,4.0,5.9,8.1,9.9]")

# From the question:
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 4.0, 5.9, 8.1, 9.9])
n = len(x)  # n = 5 data points

# Calculate required sums
sum_x = np.sum(x)           # = 1+2+3+4+5 = 15
sum_y = np.sum(y)           # = 2.1+4.0+5.9+8.1+9.9 = 30
sum_xy = np.sum(x * y)      # = 1*2.1 + 2*4.0 + 3*5.9 + 4*8.1 + 5*9.9 = 108.0
sum_x2 = np.sum(x**2)       # = 1+4+9+16+25 = 55
sum_y2 = np.sum(y**2)       # for correlation

print(f"n = {n}, Σx = {sum_x}, Σy = {sum_y}, Σxy = {sum_xy}, Σx² = {sum_x2}")

# Slope formula: a1 = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
numerator = n * sum_xy - sum_x * sum_y    # = 5*108 - 15*30 = 540 - 450 = 90
denominator = n * sum_x2 - sum_x**2       # = 5*55 - 15² = 275 - 225 = 50
a1 = numerator / denominator               # = 90/50 = 1.8

# Intercept formula: a0 = y_mean - a1 * x_mean
x_mean = sum_x / n  # = 15/5 = 3
y_mean = sum_y / n  # = 30/5 = 6
a0 = y_mean - a1 * x_mean  # = 6 - 1.8*3 = 6 - 5.4 = 0.6

print(f"Slope a1 = {numerator}/{denominator} = {a1}")
print(f"Intercept a0 = {y_mean} - {a1}*{x_mean} = {a0}")
print(f"Best fit line: y = {a0} + {a1}x")

# Correlation coefficient
denom_r = np.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
r = numerator / denom_r
print(f"Correlation r = {r:.4f}, R² = {r**2:.4f}")


# =============================================================================
# 3. POLYNOMIAL REGRESSION (QUADRATIC)
# =============================================================================
# EXAMPLE: Fit y = a0 + a1*x + a2*x² to:
#   x: 0, 1, 2, 3
#   y: 1, 2, 5, 10

print("\n" + "=" * 60)
print("3. POLYNOMIAL REGRESSION (QUADRATIC)")
print("=" * 60)
print("Fit y = a0 + a1*x + a2*x² to: x=[0,1,2,3], y=[1,2,5,10]")

# From the question:
x = np.array([0, 1, 2, 3])
y = np.array([1, 2, 5, 10])
n = len(x)  # = 4

# Calculate sums for normal equations
# Matrix: [[n,    Σx,   Σx²],
#          [Σx,   Σx²,  Σx³],
#          [Σx²,  Σx³,  Σx⁴]]
sum_x = np.sum(x)      # = 0+1+2+3 = 6
sum_x2 = np.sum(x**2)  # = 0+1+4+9 = 14
sum_x3 = np.sum(x**3)  # = 0+1+8+27 = 36
sum_x4 = np.sum(x**4)  # = 0+1+16+81 = 98

# RHS: [Σy, Σxy, Σx²y]
sum_y = np.sum(y)           # = 1+2+5+10 = 18
sum_xy = np.sum(x * y)      # = 0+2+10+30 = 42
sum_x2y = np.sum(x**2 * y)  # = 0+2+20+90 = 112

print(f"Σx={sum_x}, Σx²={sum_x2}, Σx³={sum_x3}, Σx⁴={sum_x4}")
print(f"Σy={sum_y}, Σxy={sum_xy}, Σx²y={sum_x2y}")

# Build and solve normal equations
A = np.array([
    [n,      sum_x,  sum_x2],
    [sum_x,  sum_x2, sum_x3],
    [sum_x2, sum_x3, sum_x4]
], dtype=float)
b = np.array([sum_y, sum_xy, sum_x2y], dtype=float)

print(f"Normal equations matrix:\n{A}")
print(f"RHS: {b}")

coeffs = np.linalg.solve(A, b)
print(f"Solution: a0={coeffs[0]:.4f}, a1={coeffs[1]:.4f}, a2={coeffs[2]:.4f}")
print(f"Fitted curve: y = {coeffs[0]:.2f} + {coeffs[1]:.2f}x + {coeffs[2]:.2f}x²")


# =============================================================================
# 4. GENERAL LEAST SQUARES (EXPONENTIAL FIT)
# =============================================================================
# EXAMPLE: Fit y = a0 + a1*exp(x) to:
#   x: 0, 1, 2
#   y: 2.0, 4.7, 10.4

print("\n" + "=" * 60)
print("4. GENERAL LEAST SQUARES (EXPONENTIAL FIT)")
print("=" * 60)
print("Fit y = a0 + a1*exp(x) to: x=[0,1,2], y=[2.0,4.7,10.4]")

# From the question:
x = np.array([0, 1, 2])
y = np.array([2.0, 4.7, 10.4])
n = len(x)  # = 3

# Design matrix Z: columns are [1, exp(x)]
# Z = [[1, e^0],    = [[1, 1.000],
#      [1, e^1],       [1, 2.718],
#      [1, e^2]]       [1, 7.389]]
Z = np.column_stack([np.ones(n), np.exp(x)])
print(f"Design matrix Z:\n{Z}")

# Normal equations: Z'Z * a = Z'y
ZtZ = Z.T @ Z
Zty = Z.T @ y
print(f"Z'Z:\n{ZtZ}")
print(f"Z'y: {Zty}")

coeffs = np.linalg.solve(ZtZ, Zty)
print(f"Solution: a0={coeffs[0]:.4f}, a1={coeffs[1]:.4f}")
print(f"Fitted curve: y = {coeffs[0]:.2f} + {coeffs[1]:.2f}*exp(x)")


# =============================================================================
# 5. NEWTON DIVIDED DIFFERENCES
# =============================================================================
# EXAMPLE: Build divided difference table for:
#   x: 0, 1, 3
#   y: 1, 3, 55

print("\n" + "=" * 60)
print("5. NEWTON DIVIDED DIFFERENCES")
print("=" * 60)
print("Data: x=[0,1,3], y=[1,3,55]")

# From the question:
x = np.array([0, 1, 3])
y = np.array([1, 3, 55])
n = len(x)  # = 3

# Build divided difference table
table = np.zeros((n, n))
table[:, 0] = y  # First column is just y values

# Column 1: first divided differences
# f[x0,x1] = (f[x1] - f[x0]) / (x1 - x0) = (3-1)/(1-0) = 2
# f[x1,x2] = (f[x2] - f[x1]) / (x2 - x1) = (55-3)/(3-1) = 26
table[0, 1] = (table[1, 0] - table[0, 0]) / (x[1] - x[0])  # = 2
table[1, 1] = (table[2, 0] - table[1, 0]) / (x[2] - x[1])  # = 26

# Column 2: second divided difference
# f[x0,x1,x2] = (f[x1,x2] - f[x0,x1]) / (x2 - x0) = (26-2)/(3-0) = 8
table[0, 2] = (table[1, 1] - table[0, 1]) / (x[2] - x[0])  # = 8

print("Divided difference table:")
print(f"  x    f[xi]  f[.,.]  f[.,.,.]]")
for i in range(n):
    row = f"{x[i]:3.0f}   {table[i,0]:5.1f}"
    if i < n-1:
        row += f"   {table[i,1]:5.1f}"
    if i < n-2:
        row += f"    {table[i,2]:5.1f}"
    print(row)

print(f"\nNewton coefficients (first row): {table[0, :]}")
print(f"P(x) = {table[0,0]} + {table[0,1]}(x-{x[0]}) + {table[0,2]}(x-{x[0]})(x-{x[1]})")


# =============================================================================
# 6. NEWTON INTERPOLATION - EVALUATE AT A POINT
# =============================================================================
# EXAMPLE: Using the above polynomial, find P(2)

print("\n" + "=" * 60)
print("6. NEWTON INTERPOLATION - EVALUATE AT POINT")
print("=" * 60)
print("Evaluate P(2) using Newton polynomial from above")

x_eval = 2  # Point to evaluate at

# P(x) = f[x0] + f[x0,x1](x-x0) + f[x0,x1,x2](x-x0)(x-x1)
# P(2) = 1 + 2*(2-0) + 8*(2-0)*(2-1)
#      = 1 + 2*2 + 8*2*1
#      = 1 + 4 + 16 = 21

term0 = table[0, 0]                                    # = 1
term1 = table[0, 1] * (x_eval - x[0])                  # = 2 * (2-0) = 4
term2 = table[0, 2] * (x_eval - x[0]) * (x_eval - x[1])  # = 8 * 2 * 1 = 16

result = term0 + term1 + term2
print(f"P(2) = {term0} + {table[0,1]}*(2-{x[0]}) + {table[0,2]}*(2-{x[0]})*(2-{x[1]})")
print(f"     = {term0} + {term1} + {term2} = {result}")


# =============================================================================
# 7. LAGRANGE INTERPOLATION
# =============================================================================
# EXAMPLE: Same data, using Lagrange basis polynomials
#   x: 0, 1, 3
#   y: 1, 3, 55

print("\n" + "=" * 60)
print("7. LAGRANGE INTERPOLATION")
print("=" * 60)
print("Evaluate P(2) using Lagrange: x=[0,1,3], y=[1,3,55]")

# From the question:
x = np.array([0, 1, 3])
y = np.array([1, 3, 55])
x_eval = 2

# L0(2) = (2-1)(2-3) / (0-1)(0-3) = (1)(-1) / (-1)(-3) = -1/3
L0 = ((x_eval - x[1]) * (x_eval - x[2])) / ((x[0] - x[1]) * (x[0] - x[2]))
print(f"L0(2) = (2-{x[1]})(2-{x[2]}) / ({x[0]}-{x[1]})({x[0]}-{x[2]}) = {L0:.4f}")

# L1(2) = (2-0)(2-3) / (1-0)(1-3) = (2)(-1) / (1)(-2) = 1
L1 = ((x_eval - x[0]) * (x_eval - x[2])) / ((x[1] - x[0]) * (x[1] - x[2]))
print(f"L1(2) = (2-{x[0]})(2-{x[2]}) / ({x[1]}-{x[0]})({x[1]}-{x[2]}) = {L1:.4f}")

# L2(2) = (2-0)(2-1) / (3-0)(3-1) = (2)(1) / (3)(2) = 1/3
L2 = ((x_eval - x[0]) * (x_eval - x[1])) / ((x[2] - x[0]) * (x[2] - x[1]))
print(f"L2(2) = (2-{x[0]})(2-{x[1]}) / ({x[2]}-{x[0]})({x[2]}-{x[1]}) = {L2:.4f}")

# P(2) = y0*L0 + y1*L1 + y2*L2 = 1*(-1/3) + 3*1 + 55*(1/3) = -1/3 + 3 + 55/3 = 21
result = y[0]*L0 + y[1]*L1 + y[2]*L2
print(f"P(2) = {y[0]}*{L0:.4f} + {y[1]}*{L1:.4f} + {y[2]}*{L2:.4f} = {result:.1f}")


# =============================================================================
# 8. BILINEAR INTERPOLATION
# =============================================================================
# EXAMPLE: Interpolate z at (0.5, 0.5) given grid:
#   x: 0, 1
#   y: 0, 1
#   z: [[0, 1],    (z at (0,0)=0, (1,0)=1)
#       [2, 3]]   (z at (0,1)=2, (1,1)=3)

print("\n" + "=" * 60)
print("8. BILINEAR INTERPOLATION")
print("=" * 60)
print("Interpolate z at (0.5, 0.5) on 2x2 grid")

# From the question:
x_grid = np.array([0, 1])
y_grid = np.array([0, 1])
z = np.array([[0, 1],   # z values at y=0: z(0,0)=0, z(1,0)=1
              [2, 3]])  # z values at y=1: z(0,1)=2, z(1,1)=3

x_eval, y_eval = 0.5, 0.5

# Corner values
z00 = z[0, 0]  # = 0 (bottom-left)
z10 = z[0, 1]  # = 1 (bottom-right)
z01 = z[1, 0]  # = 2 (top-left)
z11 = z[1, 1]  # = 3 (top-right)

# Normalised coordinates
t = (x_eval - 0) / (1 - 0)  # = 0.5
s = (y_eval - 0) / (1 - 0)  # = 0.5

print(f"Corner values: z00={z00}, z10={z10}, z01={z01}, z11={z11}")
print(f"Normalised: t={t}, s={s}")

# Formula: z = (1-t)(1-s)*z00 + t(1-s)*z10 + (1-t)s*z01 + ts*z11
# z = 0.5*0.5*0 + 0.5*0.5*1 + 0.5*0.5*2 + 0.5*0.5*3
#   = 0 + 0.25 + 0.5 + 0.75 = 1.5
z_interp = (1-t)*(1-s)*z00 + t*(1-s)*z10 + (1-t)*s*z01 + t*s*z11
print(f"z(0.5,0.5) = (1-{t})(1-{s})*{z00} + {t}(1-{s})*{z10} + (1-{t}){s}*{z01} + {t}*{s}*{z11}")
print(f"          = {z_interp}")


# =============================================================================
# 9. NEWTON'S METHOD FOR ROOT FINDING
# =============================================================================
# EXAMPLE: Find √2 by solving x² - 2 = 0, starting at x0 = 1.5

print("\n" + "=" * 60)
print("9. NEWTON'S METHOD FOR ROOT FINDING")
print("=" * 60)
print("Find √2 by solving x² - 2 = 0, starting at x0 = 1.5")

# From the question:
x = 1.5  # initial guess

# Functions:
# f(x) = x² - 2
# f'(x) = 2x

for i in range(5):
    f_val = x**2 - 2        # f(x)
    f_prime = 2 * x         # f'(x)
    x_new = x - f_val / f_prime  # Newton's formula
    
    print(f"Iteration {i+1}: x={x:.6f}, f(x)={f_val:.6f}, f'(x)={f_prime:.6f}")
    print(f"  x_new = {x:.6f} - {f_val:.6f}/{f_prime:.6f} = {x_new:.6f}")
    x = x_new

print(f"\nFinal answer: √2 ≈ {x:.10f}")
print(f"Actual √2 = {np.sqrt(2):.10f}")


# =============================================================================
# 10. SECANT METHOD
# =============================================================================
# EXAMPLE: Find √2 by solving x² - 2 = 0, starting at x0=1, x1=2

print("\n" + "=" * 60)
print("10. SECANT METHOD")
print("=" * 60)
print("Find √2 by solving x² - 2 = 0, starting at x0=1, x1=2")

# From the question:
x_prev = 1.0
x_curr = 2.0

f_prev = x_prev**2 - 2  # = 1 - 2 = -1
f_curr = x_curr**2 - 2  # = 4 - 2 = 2

print(f"x0={x_prev}, f(x0)={f_prev}")
print(f"x1={x_curr}, f(x1)={f_curr}")

for i in range(5):
    # Secant formula: x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
    x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
    f_new = x_new**2 - 2
    
    print(f"\nIteration {i+1}:")
    print(f"  x_new = {x_curr:.6f} - {f_curr:.6f}*({x_curr:.6f}-{x_prev:.6f})/({f_curr:.6f}-{f_prev:.6f})")
    print(f"       = {x_new:.6f}, f(x_new)={f_new:.6f}")
    
    x_prev, f_prev = x_curr, f_curr
    x_curr, f_curr = x_new, f_new

print(f"\nFinal answer: √2 ≈ {x_curr:.10f}")


# =============================================================================
# 11. NEWTON'S METHOD FOR OPTIMISATION
# =============================================================================
# EXAMPLE: Find minimum of f(x) = (x-2)² + 1, starting at x0 = 0

print("\n" + "=" * 60)
print("11. NEWTON'S METHOD FOR OPTIMISATION")
print("=" * 60)
print("Find minimum of f(x) = (x-2)² + 1, starting at x0 = 0")

# From the question:
x = 0.0  # initial guess

# Functions:
# f(x) = (x-2)² + 1
# f'(x) = 2(x-2)
# f''(x) = 2

for i in range(3):
    f_val = (x - 2)**2 + 1     # f(x)
    f_prime = 2 * (x - 2)       # f'(x)
    f_double_prime = 2          # f''(x) (constant)
    
    x_new = x - f_prime / f_double_prime  # Newton's formula for optimisation
    
    print(f"Iteration {i+1}: x={x:.4f}, f(x)={f_val:.4f}, f'(x)={f_prime:.4f}, f''(x)={f_double_prime}")
    print(f"  x_new = {x:.4f} - {f_prime:.4f}/{f_double_prime} = {x_new:.4f}")
    x = x_new

print(f"\nMinimum at x = {x}, f(x) = {(x-2)**2 + 1}")


# =============================================================================
# 12. FINITE DIFFERENCE - SECOND ORDER ODE
# =============================================================================
# EXAMPLE: Solve y'' = -2 on [0,1] with y(0)=0, y(1)=0
# Exact solution: y = x(1-x)

print("\n" + "=" * 60)
print("12. FINITE DIFFERENCE - SECOND ORDER ODE")
print("=" * 60)
print("Solve y'' = -2 on [0,1], y(0)=0, y(1)=0 with n=4 intervals")

# From the question:
x_start, x_end = 0, 1
y_start, y_end = 0, 0  # boundary conditions
n = 4                   # number of intervals

h = (x_end - x_start) / n  # = 0.25
x_grid = np.linspace(x_start, x_end, n + 1)  # [0, 0.25, 0.5, 0.75, 1]

print(f"Step size h = {h}")
print(f"Grid points: {x_grid}")
print(f"Interior points: {x_grid[1:-1]}")  # [0.25, 0.5, 0.75]

# Number of interior points = n - 1 = 3
num_interior = n - 1

# Tridiagonal system: [−2, 1, 0] [y1]   [h²g(x1) - y0]
#                     [1, −2, 1] [y2] = [h²g(x2)     ]
#                     [0, 1, −2] [y3]   [h²g(x3) - yn]

A = np.array([
    [-2,  1,  0],
    [ 1, -2,  1],
    [ 0,  1, -2]
], dtype=float)

# RHS: h²*g(xi) where g(x) = -2, and adjust for boundaries
g = -2  # from the ODE: y'' = -2
rhs = h**2 * g * np.ones(num_interior)  # = 0.25² * (-2) * [1,1,1] = [-0.125, -0.125, -0.125]
rhs[0] -= y_start   # subtract y(0) = 0
rhs[-1] -= y_end    # subtract y(1) = 0

print(f"Coefficient matrix:\n{A}")
print(f"RHS (h²g = {h**2}*{g} = {h**2*g}): {rhs}")

# Solve
y_interior = np.linalg.solve(A, rhs)
print(f"Interior solution: {y_interior}")

# Full solution
y_full = np.zeros(n + 1)
y_full[0] = y_start
y_full[-1] = y_end
y_full[1:-1] = y_interior

print(f"\nNumerical: {y_full}")
print(f"Exact y=x(1-x): {x_grid * (1 - x_grid)}")


# =============================================================================
# 13. FOURIER COEFFICIENTS (NUMERICAL)
# =============================================================================
# EXAMPLE: Find first 3 Fourier coefficients of square wave f(x) = ±1

print("\n" + "=" * 60)
print("13. FOURIER COEFFICIENTS")
print("=" * 60)
print("Find Fourier coefficients of square wave on [0, 2π]")

# From the question:
T = 2 * np.pi  # period
omega = 2 * np.pi / T  # = 1

# Define square wave: +1 for 0 < x < π, -1 for π < x < 2π
def square_wave(x):
    return np.where((x % T) < T/2, 1.0, -1.0)

# Numerical integration points
x = np.linspace(0, T, 1000)
f = square_wave(x)

# a0 = (2/T) ∫ f(x) dx  (should be 0 for symmetric square wave)
a0 = (2/T) * np.trapezoid(f, x)
print(f"a0 = {a0:.6f} (should be 0)")

# For square wave: an = 0 (no cosine terms due to symmetry)
# bn = 4/(nπ) for odd n, 0 for even n
print("\nFourier coefficients:")
for n in range(1, 6):
    an = (2/T) * np.trapezoid(f * np.cos(n * omega * x), x)
    bn = (2/T) * np.trapezoid(f * np.sin(n * omega * x), x)
    print(f"  n={n}: a{n} = {an:8.4f}, b{n} = {bn:8.4f} (theory: b{n} = {4/(n*np.pi) if n%2==1 else 0:.4f})")


# =============================================================================
# 14. PARSEVAL'S THEOREM
# =============================================================================
# EXAMPLE: Verify Parseval for square wave

print("\n" + "=" * 60)
print("14. PARSEVAL'S THEOREM")
print("=" * 60)
print("Verify: (1/T)∫|f|²dx = (a0/2)² + (1/2)Σ(an² + bn²)")

# Mean square value (LHS)
mean_sq = np.trapezoid(f**2, x) / T
print(f"Mean square value = (1/T)∫f²dx = {mean_sq:.6f}")

# Parseval sum (RHS) - for square wave, an = 0
# bn = 4/(nπ) for odd n
parseval_sum = (a0/2)**2
for n in range(1, 100):  # sum many terms
    bn_theory = 4/(n*np.pi) if n % 2 == 1 else 0
    parseval_sum += 0.5 * bn_theory**2

print(f"Parseval sum (100 terms) = {parseval_sum:.6f}")
print(f"Difference = {abs(mean_sq - parseval_sum):.6f}")

# Famous result: Σ(1/n²) for odd n = π²/8
print(f"\nNote: For square wave, Parseval gives: Σ(1/n²) for odd n = π²/8 = {np.pi**2/8:.6f}")


# =============================================================================
# 15. QR DECOMPOSITION (GRAM-SCHMIDT)
# =============================================================================
# EXAMPLE: Decompose A = [[1, 1], [0, 1], [1, 0]] into QR

print("\n" + "=" * 60)
print("15. QR DECOMPOSITION")
print("=" * 60)
print("Decompose A into Q (orthogonal) and R (upper triangular)")

# From the question:
A = np.array([
    [1, 1],
    [0, 1],
    [1, 0]
], dtype=float)

print(f"A:\n{A}")

# Column 1: a1 = [1, 0, 1]
a1 = A[:, 0]
u1 = a1.copy()
r11 = np.linalg.norm(u1)  # = sqrt(1² + 0² + 1²) = sqrt(2)
q1 = u1 / r11             # = [1/√2, 0, 1/√2]

print(f"\nStep 1: a1 = {a1}")
print(f"  ||a1|| = √(1² + 0² + 1²) = {r11:.4f}")
print(f"  q1 = a1/||a1|| = {q1}")

# Column 2: a2 = [1, 1, 0]
a2 = A[:, 1]
r12 = np.dot(a2, q1)      # projection coefficient
u2 = a2 - r12 * q1        # subtract projection
r22 = np.linalg.norm(u2)
q2 = u2 / r22

print(f"\nStep 2: a2 = {a2}")
print(f"  r12 = a2·q1 = {r12:.4f}")
print(f"  u2 = a2 - r12*q1 = {u2}")
print(f"  r22 = ||u2|| = {r22:.4f}")
print(f"  q2 = u2/||u2|| = {q2}")

Q = np.column_stack([q1, q2])
R = np.array([[r11, r12], [0, r22]])

print(f"\nQ:\n{Q}")
print(f"\nR:\n{R}")
print(f"\nVerify Q'Q = I:\n{Q.T @ Q}")
print(f"\nVerify QR = A:\n{Q @ R}")


# =============================================================================
# 16. QR ALGORITHM FOR EIGENVALUES
# =============================================================================
# EXAMPLE: Find eigenvalues of [[2, 1], [1, 2]]

print("\n" + "=" * 60)
print("16. QR ALGORITHM FOR EIGENVALUES")
print("=" * 60)
print("Find eigenvalues of A = [[2, 1], [1, 2]]")

# From the question:
A = np.array([[2, 1], [1, 2]], dtype=float)
print(f"Original matrix A:\n{A}")

# Iterate: Ak = QkRk, then Ak+1 = RkQk
for i in range(10):
    Q, R = np.linalg.qr(A)
    A_new = R @ Q
    
    if i < 3:
        print(f"\nIteration {i+1}:")
        print(f"  A:\n{A}")
        print(f"  Diagonal: {np.diag(A_new)}")
    A = A_new

print(f"\nFinal matrix (converged):\n{A}")
print(f"Eigenvalues (diagonal): {np.diag(A)}")
print(f"Expected eigenvalues: 1 and 3 (since det(A-λI)=0 gives (2-λ)²-1=0)")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY OF METHODS")
print("=" * 60)
print("""
1. GAUSSIAN ELIMINATION: Forward eliminate, back substitute
2. LINEAR REGRESSION: a1 = (nΣxy - ΣxΣy)/(nΣx² - (Σx)²), a0 = ȳ - a1x̄
3. POLYNOMIAL REGRESSION: Build normal equations matrix from sums
4. GENERAL LS: Design matrix Z, solve Z'Za = Z'y
5. DIVIDED DIFFERENCES: f[xi,xi+1] = (f[xi+1] - f[xi])/(xi+1 - xi)
6. NEWTON INTERP: P(x) = f[x0] + f[x0,x1](x-x0) + ...
7. LAGRANGE INTERP: P(x) = Σ yk * Lk(x), Lk(x) = Π(x-xj)/(xk-xj)
8. BILINEAR: z = (1-t)(1-s)z00 + t(1-s)z10 + (1-t)sz01 + tsz11
9. NEWTON ROOT: xn+1 = xn - f(xn)/f'(xn)
10. SECANT: xn+1 = xn - f(xn)(xn - xn-1)/(f(xn) - f(xn-1))
11. NEWTON OPT: xn+1 = xn - f'(xn)/f''(xn)
12. FINITE DIFF: (yi-1 - 2yi + yi+1)/h² = g(xi)
13. FOURIER: an = (2/T)∫f(x)cos(nωx)dx, bn = (2/T)∫f(x)sin(nωx)dx
14. PARSEVAL: (1/T)∫|f|²dx = (a0/2)² + (1/2)Σ(an² + bn²)
15. QR DECOMP: Gram-Schmidt orthogonalisation
16. QR EIGENVALUES: Repeat A = QR, A = RQ until diagonal
""")
