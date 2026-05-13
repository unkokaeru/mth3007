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
) -> float:
    """Estimate a definite integral using Monte Carlo sampling.

    The estimator is:
    integral ~ (b - a) * mean(f(x_i))
    where x_i are uniformly sampled from [a, b].

    Args:
        integrand: Function f(x) to integrate.
        lower_bound: Lower limit of integration a.
        upper_bound: Upper limit of integration b.
        number_of_samples: Number of random sample points.
        random_seed: Optional seed for reproducibility.

    Returns:
        Monte Carlo estimate of the integral.
    """
    rng = np.random.default_rng(random_seed)
    sample_points = rng.uniform(lower_bound, upper_bound, number_of_samples)
    function_values = integrand(sample_points)
    return float((upper_bound - lower_bound) * np.mean(function_values))


def convergence_study(
    integrand: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    lower_bound: float,
    upper_bound: float,
    exact_value: float,
    sample_counts: list[int],
    number_of_repeats: int = 20,
    random_seed: int | None = 42,
) -> tuple[list[int], list[float]]:
    """Study how Monte Carlo error decreases as number of samples increases.

    Runs multiple independent estimates for each sample count and computes
    the mean absolute error.

    Args:
        integrand: Function f(x) to integrate.
        lower_bound: Lower limit of integration a.
        upper_bound: Upper limit of integration b.
        exact_value: Known exact value of the integral (for error computation).
        sample_counts: List of sample sizes to test.
        number_of_repeats: Number of independent estimates per sample count.
        random_seed: Base seed for reproducibility.

    Returns:
        Tuple of (sample_counts, mean_absolute_errors).
    """
    mean_absolute_errors = []
    rng = np.random.default_rng(random_seed)

    for count in sample_counts:
        errors = []
        for _ in range(number_of_repeats):
            seed = int(rng.integers(0, 10**9))
            estimate = monte_carlo_integrate(integrand, lower_bound, upper_bound, count, seed)
            errors.append(abs(estimate - exact_value))
        mean_absolute_errors.append(float(np.mean(errors)))

    return sample_counts, mean_absolute_errors


def plot_convergence(
    sample_counts: list[int],
    mean_absolute_errors: list[float],
) -> None:
    """Plot Monte Carlo error convergence on a log-log scale.

    Args:
        sample_counts: List of sample sizes.
        mean_absolute_errors: Mean absolute error at each sample count.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    reference_x = np.array([sample_counts[0], sample_counts[-1]], dtype=float)
    reference_y = mean_absolute_errors[0] * np.sqrt(reference_x[0] / reference_x)

    figure, axes = plt.subplots(figsize=(8, 6))
    axes.loglog(sample_counts, mean_absolute_errors, "b-o", linewidth=1.5, label="MC error")
    axes.loglog(reference_x, reference_y, "k--", linewidth=1.0, label="O(N^{-1/2}) reference")
    axes.set_xlabel("Number of samples N")
    axes.set_ylabel("Mean absolute error")
    axes.set_title("Monte Carlo Integration: Error Convergence")
    axes.legend()
    axes.grid(True, alpha=0.3, which="both")

    figure.savefig(FIGURES_DIR / "monte_carlo_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figures/monte_carlo_convergence.png")


def main() -> None:
    """Entry point for Session 9 solutions."""
    # Q9.1: Estimate integral of sin(x) from 0 to pi (exact = 2)
    def integrand_sin(sample_points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.sin(sample_points)

    lower_bound = 0.0
    upper_bound = np.pi
    exact_value = 2.0
    number_of_samples = 10000

    estimate = monte_carlo_integrate(
        integrand_sin, lower_bound, upper_bound, number_of_samples, random_seed=42
    )
    print(f"MC estimate of integral of sin(x) from 0 to pi: {estimate:.6f}")
    print(f"Exact value: {exact_value:.6f}")
    print(f"Absolute error: {abs(estimate - exact_value):.6e}")

    # Q9.2: Convergence study
    sample_counts = [10, 100, 1000, 10000, 100000]
    counts, errors = convergence_study(
        integrand_sin, lower_bound, upper_bound, exact_value, sample_counts
    )

    print("\nConvergence study:")
    for count, error in zip(counts, errors):
        print(f"  N = {count:7d}:  mean error = {error:.4e}")

    plot_convergence(counts, errors)


if __name__ == "__main__":
    main()
