# Session 1: Scientific Computing Refresher

> Matt Watkins · <mwatkins@lincoln.ac.uk> (edited by William Fayers)

---

## Overview

This session introduces the ideas of curve fitting — specifically the simplest case of fitting a line to data we expect to be linearly related.

## Learning Outcomes

- Revise material from Scientific Computing.
- Code a working version of linear regression.
- Verify your code works correctly via an external reference.

## Recommended Reading

- Chapra and Canale, Chapters 14–15

---

## What is Numerical Methods?

Using computers to solve numerical problems in applied mathematics and physics.

**What it is not:** More programming training.

**Why learn it?** Numerical competency is one of the major skills you can bring to the marketplace alongside soft and professional skills.

---

## Philosophy

- Break down problems into small chunks.
- Use pen and paper and plan your work before attacking the keyboard.
- Test, test, and test again.

> **Important:** Try to test after every single line you add. Save frequently — on OneDrive it will keep backups too.

---

## Least Squares Regression

Suppose that you think a set of paired observations $(x_0, y_0), (x_1, y_1), \ldots, (x_{n-1}, y_{n-1})$ are related as:

$$y_i = a_0 + a_1 x_i + e_i$$

where $e_i$ is the error (or residual) between the model and the observations.

We assume there is a linear relationship between $x$ and $y$, but there is some error in the measurements.

---

## Best Fit

Given our assumption of a straight line:

$$y_i = a_0 + a_1 x_i + e_i$$

The error at each point is given by:

$$e_i = y_i - a_0 - a_1 x_i$$

We take the **sum of the squares of the errors** as our error criterion:

$$S_r = \sum_{i=0}^{n-1} e_i^2 = \sum_{i=0}^{n-1}(y_i - a_0 - a_1 x_i)^2$$

---

## Optimal Parameters

Looking at our model:

$$S_r = \sum_{i=0}^{n-1}(y_i - a_0 - a_1 x_i)^2$$

We see that there are two parameters, $a_0$ and $a_1$, that control the intercept and slope of our model.

To minimise $S_r$, we differentiate with respect to our parameters:

$$\frac{\partial S_r}{\partial a_0} = -2 \sum_{i=0}^{n-1}(y_i - a_0 - a_1 x_i)$$

$$\frac{\partial S_r}{\partial a_1} = -2 \sum_{i=0}^{n-1}(y_i - a_0 - a_1 x_i)x_i$$

> **Note:** The points $(x_i, y_i)$ are not variables — they are measured values. What we can vary are the parameters of our model. So $S_r$ is a function of the two parameters $a_0$ and $a_1$.

---

## Solving the Normal Equations

Setting the partial derivatives to zero:

$$\frac{\partial S_r}{\partial a_0} = -2 \sum_{i=0}^{n-1}(y_i - a_0 - a_1 x_i) = 0$$

$$\frac{\partial S_r}{\partial a_1} = -2 \sum_{i=0}^{n-1}(y_i - a_0 - a_1 x_i)x_i = 0$$

This gives us a pair of simultaneous linear equations called the **normal equations**.

Solving for $a_1$ and $a_0$:

$$a_1 = \frac{n \sum x_i y_i - \sum x_i \sum y_i}{n \sum x_i^2 - (\sum x_i)^2}$$

$$a_0 = \frac{\sum y_i}{n} - a_1 \frac{\sum x_i}{n} = \bar{y} - a_1 \bar{x}$$

where $\bar{y}$ and $\bar{x}$ are the means of the $y$ and $x$ values respectively.

---

## Implementation

### C++ Example

```cpp
#include <iostream>

using std::cout;

int main() {
    double values[100];
    for (int index = 0; index < 100; index++) {
        values[index] = static_cast<double>(index);
    }

    double sum_of_values = 0.0;
    for (int index = 0; index < 100; index++) {
        sum_of_values += values[index];
    }
    
    cout << "The sum of numbers 0 to 99 is " << sum_of_values << "\n";
    return 0;
}
```

### Python Example

```python
"""Calculate the sum of numbers from 0 to 99."""


def main() -> None:
    """Calculate and print the sum of integers from 0 to 99."""
    values = list(range(100))
    sum_of_values = sum(values)
    print(f"The sum of numbers 0 to 99 is {sum_of_values}")


if __name__ == "__main__":
    main()
```

---

## Summary

| Concept | Formula |
|---------|---------|
| Linear model | $y_i = a_0 + a_1 x_i + e_i$ |
| Sum of squared errors | $S_r = \sum_{i=0}^{n-1}(y_i - a_0 - a_1 x_i)^2$ |
| Slope | $a_1 = \frac{n \sum x_i y_i - \sum x_i \sum y_i}{n \sum x_i^2 - (\sum x_i)^2}$ |
| Intercept | $a_0 = \bar{y} - a_1 \bar{x}$ |
