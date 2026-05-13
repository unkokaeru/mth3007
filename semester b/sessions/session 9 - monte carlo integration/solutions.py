"""Solutions for Session 9 - Monte Carlo Integration."""

import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"


def monte_carlo_integrate(
    integrand: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    lower_bound: float,
    upper_bound: float,
    number_of_samples: int,
    random_seed: int | None = None,
) -> tuple[float, float]:
    """Estimate a definite integral using Monte Carlo sampling.

    The estimator is (lecture section 32.3):
        F_N = (b - a) * mean(f(x_i))
    where x_i are uniformly sampled from [a, b].

    The one-standard-deviation error estimate is (lecture section 32.4):
        sigma_F = (b - a) * sqrt((mean(f^2) - mean(f)^2) / N)

    Args:
        integrand: Function f(x) to integrate.
        lower_bound: Lower limit of integration a.
        upper_bound: Upper limit of integration b.
        number_of_samples: Number of random sample points N.
        random_seed: Optional seed for reproducibility.

    Returns:
        Tuple of (estimate, one_sigma_error) for the integral.
    """
    rng = np.random.default_rng(random_seed)
    interval_length = upper_bound - lower_bound
    sample_points = rng.uniform(lower_bound, upper_bound, number_of_samples)
    function_values = integrand(sample_points)

    mean_f = float(np.mean(function_values))
    mean_f_squared = float(np.mean(function_values**2))

    estimate = interval_length * mean_f
    one_sigma_error = interval_length * np.sqrt(
        (mean_f_squared - mean_f**2) / number_of_samples
    )

    return estimate, float(one_sigma_error)


def main() -> None:
    """Entry point for Session 9 solutions."""
    # Q9.1: Estimate integral of sin(x) from 0 to pi (exact = 2)
    def integrand_sin(sample_points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.sin(sample_points)

    lower_bound = 0.0
    upper_bound = np.pi
    exact_value = 2.0

    print("=" * 70)
    print("MC integration of sin(x) from 0 to pi  (exact = 2)")
    print("=" * 70)

    for number_of_samples in [100, 1000, 10000, 100000]:
        estimate, one_sigma_error = monte_carlo_integrate(
            integrand_sin, lower_bound, upper_bound, number_of_samples, random_seed=42
        )
        absolute_error = abs(estimate - exact_value)
        print(
            f"  N = {number_of_samples:7d}:  F_N = {estimate:.6f},"
            f"  1-sigma = {one_sigma_error:.4e},"
            f"  |error| = {absolute_error:.4e}"
        )

    print("=" * 70)

    # Q9.2: Demonstrate 1-sigma bounds — F lies within F_N +/- sigma_F most of the time
    number_of_samples = 10000
    estimate, one_sigma_error = monte_carlo_integrate(
        integrand_sin, lower_bound, upper_bound, number_of_samples, random_seed=42
    )
    print(f"\nFor N = {number_of_samples}:")
    print(f"  Estimate  F_N  = {estimate:.6f}")
    print(f"  1-sigma error  = {one_sigma_error:.6f}")
    print(f"  Interval       = [{estimate - one_sigma_error:.6f}, {estimate + one_sigma_error:.6f}]")
    print(f"  Exact value    = {exact_value:.6f}")


if __name__ == "__main__":
    main()
