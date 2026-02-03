1. What is meant by the following?
   - local truncation error
   - global truncation error
   - order of an algorithm
   - finite difference method

2. What is the difference between an implicit and explicit relation?

3. Implement the explicit Euler method in a programming language (preferably Python), and with it solve the ODE

$$
\frac{dy(t)}{dt} = bt - ay(t)
$$

The coefficients should be implemented as variables, but take $b = 1$, $a = 22$, $y(0) = 1$, $t_{\text{max}} = 1$, and integrate using both $\Delta t = 0.01$ and $\Delta t = 0.1$

4. Implement the implicit Euler method and solve the same ODE.

5. The analytical solution is

$$
y(t) = e^{-at}\left(y_0 + \frac{b}{a^2}\right) + \frac{bt}{a} - \frac{b}{a^2}
$$

Compare the errors for the two algorithms and plot the solution for each of them and the analytical solution vs time, $y(t)$.
