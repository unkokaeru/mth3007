# Session 1: Questions

> **MTH3007 Numerical Methods**

---

## Task 1: Computing Sums

Compute the following sums using your programming language of choice:

1. $\sum_{n=0}^{99} 2n^2$
2. $\sum_{n=1}^{100} n$
3. $\sum_{n=2}^{200} 2n$
4. $\sum_{n=1}^{200} 2n^2$

---

## Task 2: Linear Regression

Use the following formulae to find the y-intercept ($a_0$) and slope ($a_1$) of the least squares best fit for the given data.

**Formulae:**

$$a_1 = \frac{n \sum x_i y_i - \sum x_i \sum y_i}{n \sum x_i^2 - (\sum x_i)^2}$$

$$a_0 = \frac{1}{n} \sum y_i - \frac{a_1}{n} \sum x_i = \bar{y} - a_1 \bar{x}$$

**Data:**

```python
x = [
    0.526993994, 0.691126852, 0.745407955, 0.669344512, 0.518168748,
    0.291558862, 0.010870453, 0.718185730, 0.897190954, 0.476789102,
]

y = [
    3.477982975, 4.197925374, 4.127080815, 3.365719179, 3.387060084,
    1.829099436, 0.658137249, 4.023164612, 5.074088869, 2.752890033,
]
```

**Tasks:**

1. Calculate $\sum x_i$, $\sum y_i$, $\sum x_i y_i$, and $\sum x_i^2$.
2. Calculate $\bar{x}$ and $\bar{y}$.
3. Compute $a_0$ and $a_1$.
4. Verify your results using a plotting library or spreadsheet.
