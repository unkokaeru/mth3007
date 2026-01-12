# Session 4: Questions

> **MTH3007 Numerical Methods**

---

## Task 1: Exponential Basis Functions

In general linear least squares (GLS), we express our model as:

$$\mathbf{y} = \mathbf{Za} + \mathbf{e}$$

where $\mathbf{Z}$ stores each basis function $z_j$ evaluated at each data point $x_i$.

The normal equations are:

$$\mathbf{Z}^T \mathbf{Z} \mathbf{a} = \mathbf{Z}^T \mathbf{y}$$

**Task:** Fit a model of the form $y = a_0 + a_1 e^{-x} + a_2 e^{-2x}$ to the following data:

```python
x = [-3.0, -2.3, -1.6, -0.9, -0.2, 0.5, 1.2, 1.9, 2.6, 3.3, 4.0]
y = [8.26383742, 6.44045188, 4.74903073, 4.565647, 3.61011683, 
     3.32743918, 2.9643915, 1.02239181, 1.09485138, 1.84053372, 1.49110572]
```

**Steps:**

1. Construct the $\mathbf{Z}$ matrix with columns for $z_0 = 1$, $z_1 = e^{-x}$, $z_2 = e^{-2x}$
2. Compute $\mathbf{Z}^T \mathbf{Z}$ and $\mathbf{Z}^T \mathbf{y}$
3. Solve for $\mathbf{a}$ using Gaussian elimination

**Expected result:** $\mathbf{a} \approx (2.138, 0.586, -0.015)^T$

---

## Task 2: Parameter Confidence Intervals

The matrix $(\mathbf{Z}^T \mathbf{Z})^{-1}$ contains statistical information about the fitted parameters:

- **Diagonal entries:** $[(\mathbf{Z}^T \mathbf{Z})^{-1}]_{ii}$ relate to the variance of $a_i$
- **Off-diagonal entries:** relate to covariances between parameters

The **standard error** of the model is:

$$s_{y/x}^2 = \frac{1}{n-m} \sum_{i=0}^{n-1} (y_i - \hat{y}_i)^2$$

where $\hat{y}_i = \sum_{j=0}^{m-1} a_j z_j(x_i)$ is the fitted value.

The **parameter variance** is:

$$\text{Var}(a_i) = s_{y/x}^2 \cdot [(\mathbf{Z}^T \mathbf{Z})^{-1}]_{ii}$$

**Data:**

```python
x = [10.0, 16.3, 23.0, 27.5, 31.0, 35.6, 39.0, 41.5, 42.9, 45.0, 
     46.0, 45.5, 46.0, 49.0, 50.0]
y = [8.953, 16.405, 22.607, 27.769, 32.065, 35.641, 38.617, 41.095,
     43.156, 44.872, 46.301, 47.490, 48.479, 49.303, 49.988]
```

**Tasks:**

**i)** Fit a linear model $y = a_0 + a_1 x$ to the data.

**ii)** Calculate and output the variance of each parameter.

**iii)** Construct 95% confidence intervals using:

$$a_i \pm t_{\alpha/2, n-m} \cdot \sqrt{\text{Var}(a_i)}$$

where $t_{\alpha/2, n-m}$ is the critical value from the t-distribution with $n - m$ degrees of freedom and $\alpha = 0.05$.

---

## Further Reading

- Chapra and Canale, Chapter 17
- Numerical Recipes, Chapters 14–15
