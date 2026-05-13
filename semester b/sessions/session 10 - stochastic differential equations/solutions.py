"""Solutions for Session 10 - Stochastic Differential Equations."""

import logging
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


def euler_maruyama(
    drift_coefficient: float,
    diffusion_coefficient: float,
    initial_value: float,
    time_start: float,
    time_end: float,
    time_step: float,
    random_seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Solve a linear SDE using the Euler-Maruyama method.

    Solves:  dX = -drift_coefficient * X dt + diffusion_coefficient * dW

    The update rule is:
    X_{n+1} = X_n - drift_coefficient * X_n * dt + diffusion_coefficient * sqrt(dt) * Z_n
    where Z_n ~ N(0, 1) are independent standard normal random variables.

    Args:
        drift_coefficient: Mean-reversion rate (kappa in OU process).
        diffusion_coefficient: Noise amplitude (sigma).
        initial_value: Initial condition X(t_0).
        time_start: Starting time t_0.
        time_end: Ending time t_max.
        time_step: Time step size Delta t.
        random_seed: Optional seed for reproducibility.

    Returns:
        Tuple of (time_values, path_values) arrays.
    """
    rng = np.random.default_rng(random_seed)
    number_of_steps = int((time_end - time_start) / time_step)
    time_values = np.linspace(time_start, time_end, number_of_steps + 1)
    path_values = np.zeros(number_of_steps + 1)
    path_values[0] = initial_value

    noise_increments = rng.standard_normal(number_of_steps) * np.sqrt(time_step)

    for step_index in range(number_of_steps):
        current_value = path_values[step_index]
        path_values[step_index + 1] = (
            current_value
            - drift_coefficient * current_value * time_step
            + diffusion_coefficient * noise_increments[step_index]
        )

    return time_values, path_values


def ornstein_uhlenbeck_process(
    mean_reversion_rate: float,
    diffusion_coefficient: float,
    initial_value: float,
    time_start: float,
    time_end: float,
    time_step: float,
    random_seed: int | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simulate an Ornstein-Uhlenbeck process using the Euler-Maruyama scheme.

    Solves: dV = -mean_reversion_rate * V dt + dW

    The numerical scheme (from lecture section 38.6.1) is:
    V(t + dt) = (1 - mean_reversion_rate * dt) * V(t) + sqrt(dt) * Z(t)

    where Z(t) ~ N(0, 1) is a standard normal random variable at each step.
    Note that for mean_reversion_rate = 0 this reduces to the Wiener process.

    Args:
        mean_reversion_rate: Rate k controlling reversion towards zero.
        diffusion_coefficient: Noise amplitude (coefficient of dW).
        initial_value: Initial condition V(t_0).
        time_start: Starting time t_0.
        time_end: Ending time t_max.
        time_step: Time step size Delta t.
        random_seed: Optional seed for reproducibility.

    Returns:
        Tuple of (time_values, path_values) arrays.
    """
    rng = np.random.default_rng(random_seed)
    number_of_steps = int((time_end - time_start) / time_step)
    time_values = np.linspace(time_start, time_end, number_of_steps + 1)
    path_values = np.zeros(number_of_steps + 1)
    path_values[0] = initial_value

    noise_increments = rng.standard_normal(number_of_steps) * np.sqrt(time_step)

    for step_index in range(number_of_steps):
        current_value = path_values[step_index]
        path_values[step_index + 1] = (
            (1.0 - mean_reversion_rate * time_step) * current_value
            + diffusion_coefficient * noise_increments[step_index]
        )

    return time_values, path_values

def simulate_first_passage_time(
    drift_coefficient: float,
    diffusion_coefficient: float,
    initial_value: float,
    threshold: float,
    time_step: float,
    max_time: float,
    number_of_walkers: int,
    random_seed: int | None = None,
) -> float:
    """Estimate the mean first-passage time by averaging over many realisations.

    Simulates the SDE dX = -drift_coefficient * X dt + diffusion_coefficient * dW
    for multiple independent paths and records the first time each path reaches
    the given threshold. Returns the mean over all walkers that crossed.

    Args:
        drift_coefficient: Drift coefficient (mean-reversion rate).
        diffusion_coefficient: Noise amplitude.
        initial_value: Initial value for all paths.
        threshold: Threshold value (first-passage target).
        time_step: Time step size Delta t.
        max_time: Maximum simulation time per walker.
        number_of_walkers: Number of independent realisations.
        random_seed: Optional seed for reproducibility.

    Returns:
        Mean first-passage time, or nan if no walkers crossed the threshold.
    """
    rng = np.random.default_rng(random_seed)
    number_of_steps = int(max_time / time_step)
    passage_times = []

    for walker_index in range(number_of_walkers):
        current_value = initial_value
        noise_increments = rng.standard_normal(number_of_steps) * np.sqrt(time_step)

        for step_index in range(number_of_steps):
            current_value = (
                current_value
                - drift_coefficient * current_value * time_step
                + diffusion_coefficient * noise_increments[step_index]
            )
            if current_value >= threshold:
                passage_times.append((step_index + 1) * time_step)
                break

    if not passage_times:
        return float("nan")
    return float(np.mean(passage_times))


def plot_sample_paths(
    time_values: npt.NDArray[np.float64],
    paths: list[npt.NDArray[np.float64]],
    title: str,
    filename: str,
) -> None:
    """Plot multiple sample paths of a stochastic process.

    Args:
        time_values: Array of time values.
        paths: List of path arrays to plot.
        title: Plot title.
        filename: Output filename (saved to FIGURES_DIR).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(figsize=(10, 6))
    for path in paths:
        axes.plot(time_values, path, linewidth=0.8, alpha=0.7)
    axes.set_xlabel("Time t")
    axes.set_ylabel("X(t)")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)

    figure.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved figures/%s", filename)


def main() -> None:
    """Entry point for Session 10 solutions."""
    # Q10.1: Simulate Ornstein-Uhlenbeck process
    mean_reversion_rate = 1.0
    diffusion_coefficient = 0.5
    initial_value = 2.0
    time_start = 0.0
    time_end = 10.0
    time_step = 0.01

    paths = []
    for walker_index in range(5):
        time_values, path = ornstein_uhlenbeck_process(
            mean_reversion_rate,
            diffusion_coefficient,
            initial_value,
            time_start,
            time_end,
            time_step,
            random_seed=walker_index,
        )
        paths.append(path)

    plot_sample_paths(
        time_values,
        paths,
        title="Ornstein-Uhlenbeck Process (5 sample paths)",
        filename="ou_sample_paths.png",
    )
    print(f"OU process: final values = {[f'{p[-1]:.4f}' for p in paths]}")

    # Q10.2: First-passage time
    threshold = 3.0
    mean_fpt = simulate_first_passage_time(
        drift_coefficient=0.5,
        diffusion_coefficient=1.0,
        initial_value=0.0,
        threshold=threshold,
        time_step=0.001,
        max_time=50.0,
        number_of_walkers=1000,
        random_seed=42,
    )
    print(f"Mean first-passage time to threshold {threshold}: {mean_fpt:.4f}")


if __name__ == "__main__":
    main()
