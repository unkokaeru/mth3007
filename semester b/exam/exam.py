"""Final exam for semester B."""
from collections.abc import Callable

import numpy as np
import numpy.typing as npt


"""
# Markdown supporting solutions:

## Question One

0.000206(4198202345484...), 3 s.f.

## Question Two

Derivation:
q_{n+1} = q_n + h * f(t_{n+1}, q_{n+1})
q_{n+1} = q_n + h * (q_{n+1} - t_{n+1}^3)
q_{n+1} - h * q_{n+1} = q_n - h * t_{n+1}^3
q_{n+1} * (1 - h) = q_n - h * t_{n+1}^3
q_{n+1} = (q_n - h * t_{n+1}^3) / (1 - h)

The equivalent notation is then D.

## Question Three

Ralston's method is a type of RK to minimise numerical error; it uses a different set of coefficients for the intermediate stages, so for certain problems has smaller error compared to RK4.

## Question Four

Implicit is unconditionally stable, so it handles larger time steps without becoming unstable - useful for stiff equations, where explicit methods would require very small timesteps for stability.
The drawback is that it requires solving a nonlinear equation each time step, which is computationally expensive.

## Question Five

Unconditionally stable: if a method remains stable for any size of time step, e.g. implicit Euler method.
Conditionally stable: if a method is stable for certain sizes of time step, e.g. explicit Euler method (with small time steps).
Unconditionally unstable: if a method is unstable for any size of time step, e.g. the explicit central-time central-space (leapfrog-in-time) scheme applied to the diffusion equation - it amplifies errors regardless of how small dt is chosen.

## Question Six

Truncation error: from approximating something analytical, like differentiation/integration, with a finite number of steps - the difference between analytical and numerical solutions.
Round-off error: from finite precision during numerical computation - the difference between a handwritten computation and one done on a computer (which suffers with this finite precision internally, rounding).

## Question Seven

For a PDE, if the spatial step is not small enough then decreasing the time step might not make the solution more accurate due to instability.
For a stiff ODE, if the method isn't used to stiff equations then decreasing the time step might not make the solution more accurate, like using the explicit Euler method for a stiff equation could lead to instability.

## Question Eight

The error of the RK4 method scales with the fourth power of the time step, so to reduce error from £192 -> £1, we can write:
192 * (dt_new / 10)^4 = 1
(dt_new / 10)^4 = 1 / 192
dt_new / 10 = (1 / 192)^(1/4)
dt_new = 10 * (1 / 192)^(1/4)
dt_new = 0.4777213961021834
So, the new time step should be around 0.4777...
"""


def explicit_euler_method(
    derivative_function: Callable[[float, float], float],
    initial_value: float,
    time_start: float,
    time_end: float,
    time_step: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve an ODE using the explicit Euler method.

    The explicit Euler method uses the approximation:
    y_{n+1} = y_n + h * f(t_n, y_n)

    Args:
        derivative_function: Function f(t, y) that computes dy/dt.
        initial_value: Initial condition y(t_0).
        time_start: Starting time t_0.
        time_end: Ending time t_max.
        time_step: Time step size h (Delta t).

    Returns:
        Tuple of (time_values, solution_values) arrays.
    """
    number_of_steps = int((time_end - time_start) / time_step)
    time_values = np.linspace(time_start, time_end, number_of_steps + 1)
    solution_values = np.zeros(number_of_steps + 1)
    solution_values[0] = initial_value

    for step_index in range(number_of_steps):
        current_time = time_values[step_index]
        current_value = solution_values[step_index]
        derivative = derivative_function(current_time, current_value)
        solution_values[step_index + 1] = current_value + time_step * derivative

    return time_values, solution_values


def main():
    # Q1: Explicit Euler to integrate dx(t)/dt = (-4x(t))/(1+t), from t=0..8, where x(0) = 4
    def function_one(t: float, x: float) -> float:
        return (-4 * x) / (1 + t)

    _, question_one = explicit_euler_method(
        derivative_function=function_one,
        initial_value=4.0,
        time_start=0.0,
        time_end=8.0,
        time_step=0.1,
    )

    print(f"Q1: {question_one[-1]}")

    # Q2: Derive the implicit Euler algorithm for dq/dt = q-t^3
    # (indicated in the markdown document)
    # 
    # Q3: What is the main difference between RK4 and Ralston's method?
    # (indicated in the markdown document)
    # 
    # Q4: What is the benefit of using implicit Euler over explicit Euler? Drawback, too?
    # (indicated in the markdown document)
    # 
    # Q5: What is meant by unconditionally stable, conditionally stale, and unconditionally unstable? Examples for each.
    # (indicated in the markdown document)
    # 
    # Q6: What is the difference between a truncation error and a round-off error when discussing a numerical method in solving ODEs?
    # (indicated in the markdown document)
    # 
    # Q7: Typically decreasing the integration step leads to a more accurate solution. Give two examples (one related to a PDE) on why this isn't always the case.
    # (indicated in the markdown document)
    # 
    # Q8: A person programmed the RK4 method for solving the ODE M'(t) = g(t,M(t)) with M(0) = 784115 (£) and used an integration time step of dt = 10 (days).
    #     The resulting error in the amount of money at time t = 1 (year) is 192 (£) - calculated with abs(M_RK4 - M_exact) both at t = 1 (year).
    #     What is the value of dt (days) so that the error is 1 (£)? Neglect round-off errors and assume theoretical error scaling.
    #     (indicated in the markdown document, but final calculation below)
    question_eight = 10 * pow((1 / 192),(1/4))
    print(f"Q8: {question_eight}")

    # Q9: Consider the PDE, (partial^2 G(t,x)/partial x^2) = k^-1 * (partial G(t,x))/(partial t),
    #     where G is the amount of green ink at position x and time t, and k=0.6.
    #     Assume that at t=0, the initial condition is that G(t=0,x)=100 (%), and
    #     the boundary conditions are for the left G(t,x=0)=0 (%) and for the right G(t,x=8)=0 (%).
    #     What is the amount of ink left in the middle at t=7, in %?
    #     Solve with an explicit Euler method with a sufficiently small time step and spatial step such that the answer is correct to 2 s.f.
    # PDE rearranged: dG/dt = k * d^2G/dx^2, with k = 0.6.
    # Stability for explicit Euler: k * dt / dx^2 <= 0.5.
    k_constant = 0.6

    def function_nine(t: float, G: npt.NDArray[np.float64], dx: float) -> npt.NDArray[np.float64]:
        d2G_dx2 = np.zeros_like(G)
        d2G_dx2[1:-1] = (G[2:] - 2 * G[1:-1] + G[:-2]) / (dx * dx)
        return k_constant * d2G_dx2

    # Set up spatial and temporal parameters
    spatial_start = 0.0
    spatial_end = 8.0
    spatial_step = 0.1
    time_start = 0.0
    time_end = 7.0
    # k * dt / dx^2 = 0.6 * 0.005 / 0.01 = 0.3 -> stable
    time_step = 0.005
    spatial_points = int((spatial_end - spatial_start) / spatial_step) + 1

    # Initialise G with initial and boundary conditions
    G = np.zeros(spatial_points)
    G[:] = 100.0  # Initial condition
    G[0] = 0.0  # Left boundary condition
    G[-1] = 0.0  # Right boundary condition

    # Time-stepping loop
    for _ in range(int((time_end - time_start) / time_step)):
        G += time_step * function_nine(time_start, G, spatial_step)
        time_start += time_step

    # Final answer for the ink (G) in the middle at t=7 in %
    middle_index = spatial_points // 2
    print(f"Q9: {G[middle_index]} % (at x = {middle_index * spatial_step})")


if __name__ == "__main__":
    main()