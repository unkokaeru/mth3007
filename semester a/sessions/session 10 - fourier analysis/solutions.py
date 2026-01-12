"""Session 10: Fourier Analysis Solutions.

This module provides solutions to the Session 10 exercises covering:
- Fourier series coefficients
- Square, triangle, and sawtooth waves
- Gibbs phenomenon
- Parseval's theorem

Author: William Fayers
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Create figures directory in the same folder as this script
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def compute_fourier_coefficients_numerical(
    func: callable,
    period: float,
    num_terms: int,
    num_integration_points: int = 1000,
) -> dict:
    """Compute Fourier coefficients numerically using trapezoidal integration.
    
    Parameters
    ----------
    func : callable
        The periodic function f(x).
    period : float
        The period T of the function.
    num_terms : int
        Number of Fourier terms to compute (n = 0, 1, ..., N).
    num_integration_points : int, optional
        Number of points for numerical integration.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'a0': the constant term (a₀)
        - 'a_n': array of cosine coefficients [a₁, a₂, ..., aₙ]
        - 'b_n': array of sine coefficients [b₁, b₂, ..., bₙ]
    
    Notes
    -----
    The Fourier series is:
        f(x) = a₀/2 + Σₙ(aₙcos(nωx) + bₙsin(nωx))
    
    where ω = 2π/T and:
        a₀ = (2/T) ∫₀ᵀ f(x) dx
        aₙ = (2/T) ∫₀ᵀ f(x)cos(nωx) dx
        bₙ = (2/T) ∫₀ᵀ f(x)sin(nωx) dx
    """
    omega = 2 * np.pi / period
    x_integration = np.linspace(0, period, num_integration_points)
    dx = period / (num_integration_points - 1)
    f_values = func(x_integration)
    
    # Compute a₀ using trapezoidal rule
    a0 = (2 / period) * np.trapezoid(f_values, x_integration)
    
    # Compute aₙ and bₙ for n = 1, 2, ..., N
    a_coefficients = np.zeros(num_terms)
    b_coefficients = np.zeros(num_terms)
    
    for harmonic_index in range(1, num_terms + 1):
        cos_values = f_values * np.cos(harmonic_index * omega * x_integration)
        sin_values = f_values * np.sin(harmonic_index * omega * x_integration)
        
        a_coefficients[harmonic_index - 1] = (
            (2 / period) * np.trapezoid(cos_values, x_integration)
        )
        b_coefficients[harmonic_index - 1] = (
            (2 / period) * np.trapezoid(sin_values, x_integration)
        )
    
    return {
        'a0': a0,
        'a_n': a_coefficients,
        'b_n': b_coefficients,
        'period': period,
        'omega': omega,
    }


def evaluate_fourier_series(
    x_values: np.ndarray,
    coefficients: dict,
    num_terms: int = None,
) -> np.ndarray:
    """Evaluate the Fourier series at given x values.
    
    Parameters
    ----------
    x_values : np.ndarray
        Points at which to evaluate the series.
    coefficients : dict
        Fourier coefficients from compute_fourier_coefficients_numerical.
    num_terms : int, optional
        Number of terms to use. If None, uses all available.
    
    Returns
    -------
    np.ndarray
        Values of the Fourier series at x_values.
    """
    omega = coefficients['omega']
    a0 = coefficients['a0']
    a_n = coefficients['a_n']
    b_n = coefficients['b_n']
    
    if num_terms is None:
        num_terms = len(a_n)
    
    # Start with constant term
    result = np.ones_like(x_values) * a0 / 2
    
    # Add harmonic terms
    for harmonic_index in range(1, min(num_terms + 1, len(a_n) + 1)):
        result += (
            a_n[harmonic_index - 1] * np.cos(harmonic_index * omega * x_values) +
            b_n[harmonic_index - 1] * np.sin(harmonic_index * omega * x_values)
        )
    
    return result


def square_wave(x: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    """Generate a square wave.
    
    Parameters
    ----------
    x : np.ndarray
        Input values.
    period : float, optional
        Period of the wave (default 2π).
    
    Returns
    -------
    np.ndarray
        Square wave values (+1 or -1).
    """
    normalised_x = (x % period) / period
    return np.where(normalised_x < 0.5, 1.0, -1.0)


def triangle_wave(x: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    """Generate a triangle wave.
    
    Parameters
    ----------
    x : np.ndarray
        Input values.
    period : float, optional
        Period of the wave (default 2π).
    
    Returns
    -------
    np.ndarray
        Triangle wave values (ranging from -1 to 1).
    """
    normalised_x = (x % period) / period
    return np.where(
        normalised_x < 0.5,
        4 * normalised_x - 1,
        3 - 4 * normalised_x
    )


def sawtooth_wave(x: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    """Generate a sawtooth wave.
    
    Parameters
    ----------
    x : np.ndarray
        Input values.
    period : float, optional
        Period of the wave (default 2π).
    
    Returns
    -------
    np.ndarray
        Sawtooth wave values (ranging from -1 to 1).
    """
    normalised_x = (x % period) / period
    return 2 * normalised_x - 1


def analytical_square_wave_coefficients(num_terms: int) -> dict:
    """Get analytical Fourier coefficients for the square wave.
    
    Parameters
    ----------
    num_terms : int
        Number of terms.
    
    Returns
    -------
    dict
        Analytical Fourier coefficients.
    
    Notes
    -----
    For a square wave of amplitude 1 and period 2π:
        a₀ = 0
        aₙ = 0 (all n)
        bₙ = 4/(nπ) for odd n, 0 for even n
    """
    a_coefficients = np.zeros(num_terms)
    b_coefficients = np.zeros(num_terms)
    
    for harmonic_number in range(1, num_terms + 1):
        if harmonic_number % 2 == 1:  # Odd harmonics only
            b_coefficients[harmonic_number - 1] = 4 / (harmonic_number * np.pi)
    
    return {
        'a0': 0.0,
        'a_n': a_coefficients,
        'b_n': b_coefficients,
        'period': 2 * np.pi,
        'omega': 1.0,
    }


def analytical_triangle_wave_coefficients(num_terms: int) -> dict:
    """Get analytical Fourier coefficients for the triangle wave.
    
    Parameters
    ----------
    num_terms : int
        Number of terms.
    
    Returns
    -------
    dict
        Analytical Fourier coefficients.
    
    Notes
    -----
    For a triangle wave of amplitude 1 and period 2π:
        a₀ = 0
        aₙ = 0 (all n)
        bₙ = 8/(n²π²) × (-1)^((n-1)/2) for odd n, 0 for even n
    """
    a_coefficients = np.zeros(num_terms)
    b_coefficients = np.zeros(num_terms)
    
    for harmonic_number in range(1, num_terms + 1):
        if harmonic_number % 2 == 1:  # Odd harmonics only
            sign = (-1)**((harmonic_number - 1) // 2)
            b_coefficients[harmonic_number - 1] = 8 * sign / (harmonic_number**2 * np.pi**2)
    
    return {
        'a0': 0.0,
        'a_n': a_coefficients,
        'b_n': b_coefficients,
        'period': 2 * np.pi,
        'omega': 1.0,
    }


def analytical_sawtooth_wave_coefficients(num_terms: int) -> dict:
    """Get analytical Fourier coefficients for the sawtooth wave.
    
    Parameters
    ----------
    num_terms : int
        Number of terms.
    
    Returns
    -------
    dict
        Analytical Fourier coefficients.
    
    Notes
    -----
    For a sawtooth wave of amplitude 1 and period 2π:
        a₀ = 0
        aₙ = 0 (all n)
        bₙ = -2/(nπ) × (-1)^n = 2/(nπ) × (-1)^(n+1)
    """
    a_coefficients = np.zeros(num_terms)
    b_coefficients = np.zeros(num_terms)
    
    for harmonic_number in range(1, num_terms + 1):
        b_coefficients[harmonic_number - 1] = 2 * (-1)**(harmonic_number + 1) / (harmonic_number * np.pi)
    
    return {
        'a0': 0.0,
        'a_n': a_coefficients,
        'b_n': b_coefficients,
        'period': 2 * np.pi,
        'omega': 1.0,
    }


def compute_parseval_sum(coefficients: dict, num_terms: int = None) -> float:
    """Compute the Parseval sum: (a₀/2)² + (1/2)Σ(aₙ² + bₙ²).
    
    Parameters
    ----------
    coefficients : dict
        Fourier coefficients.
    num_terms : int, optional
        Number of terms to include.
    
    Returns
    -------
    float
        The Parseval sum.
    
    Notes
    -----
    Parseval's theorem states:
        (1/T) ∫₀ᵀ |f(x)|² dx = (a₀/2)² + (1/2)Σₙ(aₙ² + bₙ²)
    """
    a0 = coefficients['a0']
    a_n = coefficients['a_n']
    b_n = coefficients['b_n']
    
    if num_terms is None:
        num_terms = len(a_n)
    
    parseval_sum = (a0 / 2)**2
    
    for term_index in range(min(num_terms, len(a_n))):
        parseval_sum += 0.5 * (a_n[term_index]**2 + b_n[term_index]**2)
    
    return parseval_sum


def compute_mean_square_numerical(
    func: callable,
    period: float,
    num_integration_points: int = 1000,
) -> float:
    """Compute the mean square value of a function.
    
    Parameters
    ----------
    func : callable
        The function f(x).
    period : float
        The period T.
    num_integration_points : int, optional
        Number of integration points.
    
    Returns
    -------
    float
        The mean square value: (1/T) ∫₀ᵀ |f(x)|² dx.
    """
    x_integration = np.linspace(0, period, num_integration_points)
    f_squared = func(x_integration)**2
    
    return np.trapezoid(f_squared, x_integration) / period


def task1_fourier_coefficients() -> dict:
    """Compute and compare Fourier coefficients for standard waveforms.
    
    Returns
    -------
    dict
        Results for square, triangle, and sawtooth waves.
    """
    num_terms = 20
    period = 2 * np.pi
    
    results = {}
    
    # Square wave
    square_numerical = compute_fourier_coefficients_numerical(
        lambda x: square_wave(x, period), period, num_terms
    )
    square_analytical = analytical_square_wave_coefficients(num_terms)
    results['square_wave'] = {
        'numerical': square_numerical,
        'analytical': square_analytical,
    }
    
    # Triangle wave
    triangle_numerical = compute_fourier_coefficients_numerical(
        lambda x: triangle_wave(x, period), period, num_terms
    )
    triangle_analytical = analytical_triangle_wave_coefficients(num_terms)
    results['triangle_wave'] = {
        'numerical': triangle_numerical,
        'analytical': triangle_analytical,
    }
    
    # Sawtooth wave
    sawtooth_numerical = compute_fourier_coefficients_numerical(
        lambda x: sawtooth_wave(x, period), period, num_terms
    )
    sawtooth_analytical = analytical_sawtooth_wave_coefficients(num_terms)
    results['sawtooth_wave'] = {
        'numerical': sawtooth_numerical,
        'analytical': sawtooth_analytical,
    }
    
    return results


def task2_gibbs_phenomenon() -> dict:
    """Demonstrate the Gibbs phenomenon.
    
    Returns
    -------
    dict
        Results showing Gibbs overshoot for different numbers of terms.
    """
    period = 2 * np.pi
    
    # Analytical coefficients for square wave
    num_terms_list = [1, 3, 5, 11, 21, 51, 101]
    
    x_fine = np.linspace(-0.1, np.pi + 0.1, 2000)
    
    results = {}
    
    for num_terms in num_terms_list:
        coefficients = analytical_square_wave_coefficients(num_terms)
        y_fourier = evaluate_fourier_series(x_fine, coefficients, num_terms)
        
        # Find overshoot near discontinuity at x = π
        discontinuity_region = (x_fine > np.pi - 0.3) & (x_fine < np.pi + 0.3)
        overshoot = np.max(y_fourier[discontinuity_region])
        
        results[num_terms] = {
            'x_values': x_fine,
            'y_values': y_fourier,
            'overshoot': overshoot,
            'overshoot_percentage': (overshoot - 1) * 100,
        }
    
    return results


def task3_parseval_verification() -> dict:
    """Verify Parseval's theorem for standard waveforms.
    
    Returns
    -------
    dict
        Parseval verification results.
    """
    period = 2 * np.pi
    num_terms = 100
    
    results = {}
    
    # Square wave
    square_coeffs = analytical_square_wave_coefficients(num_terms)
    square_parseval = compute_parseval_sum(square_coeffs)
    square_mean_sq = compute_mean_square_numerical(
        lambda x: square_wave(x, period), period
    )
    results['square_wave'] = {
        'parseval_sum': square_parseval,
        'mean_square': square_mean_sq,
        'exact_mean_square': 1.0,  # For ±1 square wave
    }
    
    # Triangle wave
    triangle_coeffs = analytical_triangle_wave_coefficients(num_terms)
    triangle_parseval = compute_parseval_sum(triangle_coeffs)
    triangle_mean_sq = compute_mean_square_numerical(
        lambda x: triangle_wave(x, period), period
    )
    results['triangle_wave'] = {
        'parseval_sum': triangle_parseval,
        'mean_square': triangle_mean_sq,
        'exact_mean_square': 1/3,  # For triangle wave
    }
    
    # Sawtooth wave
    sawtooth_coeffs = analytical_sawtooth_wave_coefficients(num_terms)
    sawtooth_parseval = compute_parseval_sum(sawtooth_coeffs)
    sawtooth_mean_sq = compute_mean_square_numerical(
        lambda x: sawtooth_wave(x, period), period
    )
    results['sawtooth_wave'] = {
        'parseval_sum': sawtooth_parseval,
        'mean_square': sawtooth_mean_sq,
        'exact_mean_square': 1/3,  # For sawtooth wave
    }
    
    return results


def plot_waveform_and_fourier(
    wave_func: callable,
    coefficients: dict,
    wave_name: str,
    num_terms_list: list = [1, 3, 5, 11],
) -> None:
    """Plot a waveform and its Fourier approximations.
    
    Parameters
    ----------
    wave_func : callable
        The waveform function.
    coefficients : dict
        Fourier coefficients.
    wave_name : str
        Name of the waveform.
    num_terms_list : list
        Numbers of terms to show.
    """
    period = 2 * np.pi
    x_values = np.linspace(0, 2 * period, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Original function
    y_original = wave_func(x_values, period)
    
    for i, num_terms in enumerate(num_terms_list):
        axes[i].plot(x_values, y_original, 'k-', linewidth=1, alpha=0.5, label='Original')
        
        y_fourier = evaluate_fourier_series(x_values, coefficients, num_terms)
        axes[i].plot(x_values, y_fourier, 'b-', linewidth=2,
                     label=f'Fourier ({num_terms} terms)')
        
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('f(x)')
        axes[i].set_title(f'{wave_name}: {num_terms} term(s)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim([0, 2 * period])
        axes[i].set_ylim([-1.5, 1.5])
    
    plt.suptitle(f'Fourier Series Approximation: {wave_name}', fontsize=14)
    plt.tight_layout()
    # Generate filename from wave name
    filename = wave_name.lower().replace(' ', '_') + '_fourier_approximation.png'
    save_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_gibbs_phenomenon(gibbs_results: dict) -> None:
    """Plot the Gibbs phenomenon.
    
    Parameters
    ----------
    gibbs_results : dict
        Results from task2_gibbs_phenomenon.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Different numbers of terms near discontinuity
    ax1 = axes[0, 0]
    for num_terms in [5, 11, 21, 51]:
        data = gibbs_results[num_terms]
        mask = (data['x_values'] > 2.5) & (data['x_values'] < 4)
        ax1.plot(data['x_values'][mask], data['y_values'][mask],
                 label=f'N = {num_terms}', linewidth=1.5)
    
    ax1.axhline(y=1, color='k', linestyle='--', linewidth=1)
    ax1.axhline(y=-1, color='k', linestyle='--', linewidth=1)
    ax1.axvline(x=np.pi, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Gibbs Phenomenon Near Discontinuity (x = π)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Overshoot convergence
    ax2 = axes[0, 1]
    num_terms_vals = list(gibbs_results.keys())
    overshoots = [gibbs_results[num_terms]['overshoot'] for num_terms in num_terms_vals]
    ax2.semilogx(num_terms_vals, overshoots, 'bo-', markersize=8)
    ax2.axhline(y=1.0898, color='r', linestyle='--', label='Gibbs limit ≈ 1.0898')
    ax2.set_xlabel('Number of terms')
    ax2.set_ylabel('Maximum overshoot')
    ax2.set_title('Overshoot vs Number of Terms')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Zoomed view of overshoot
    ax3 = axes[1, 0]
    data_101 = gibbs_results[101]
    mask = (data_101['x_values'] > 3.0) & (data_101['x_values'] < 3.3)
    ax3.plot(data_101['x_values'][mask], data_101['y_values'][mask], 'b-', linewidth=2)
    ax3.axhline(y=1, color='k', linestyle='--')
    ax3.axhline(y=1.0898, color='r', linestyle='--', label='Gibbs limit')
    ax3.set_xlabel('x')
    ax3.set_ylabel('f(x)')
    ax3.set_title('Zoomed View: 101 Terms')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Full waveform with 51 terms
    ax4 = axes[1, 1]
    data_51 = gibbs_results[51]
    ax4.plot(data_51['x_values'], data_51['y_values'], 'b-', linewidth=1.5)
    ax4.axhline(y=1, color='k', linestyle='--', linewidth=0.5)
    ax4.axhline(y=-1, color='k', linestyle='--', linewidth=0.5)
    ax4.set_xlabel('x')
    ax4.set_ylabel('f(x)')
    ax4.set_title('Square Wave Approximation: 51 Terms')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'gibbs_phenomenon.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def display_coefficient_table(coefficients: dict, wave_name: str, num_display: int = 10) -> None:
    """Display Fourier coefficients in a table.
    
    Parameters
    ----------
    coefficients : dict
        Fourier coefficients.
    wave_name : str
        Name of the waveform.
    num_display : int
        Number of coefficients to display.
    """
    print(f"\n{wave_name} Fourier Coefficients:")
    print("-" * 50)
    print(f"a₀ = {coefficients['a0']:.6f}")
    print()
    print("  n  |       aₙ       |       bₙ")
    print("-----|----------------|----------------")
    
    for term_index in range(num_display):
        print(f"  {term_index+1:2d} | {coefficients['a_n'][term_index]:14.6f} | {coefficients['b_n'][term_index]:14.6f}")


def main() -> None:
    """Execute all Session 10 tasks and display results."""
    print("=" * 60)
    print("SESSION 10: FOURIER ANALYSIS")
    print("=" * 60)
    
    # Task 1: Fourier Coefficients
    print("\n--- Task 1: Fourier Coefficients for Standard Waveforms ---\n")
    
    task1_results = task1_fourier_coefficients()
    
    for wave_name, data in task1_results.items():
        display_name = wave_name.replace('_', ' ').title()
        display_coefficient_table(data['analytical'], display_name)
        
        # Compare numerical and analytical
        print(f"\n  Comparison (first 5 non-zero bₙ):")
        numerical = data['numerical']['b_n']
        analytical = data['analytical']['b_n']
        print("    n  | Numerical     | Analytical    | Difference")
        print("   ----|---------------|---------------|------------")
        count = 0
        for term_index in range(len(numerical)):
            if abs(analytical[term_index]) > 1e-10:
                diff = abs(numerical[term_index] - analytical[term_index])
                print(f"   {term_index+1:3d} | {numerical[term_index]:13.6f} | {analytical[term_index]:13.6f} | {diff:.2e}")
                count += 1
                if count >= 5:
                    break
    
    # Plot waveforms
    print("\nGenerating waveform plots...")
    period = 2 * np.pi
    
    for wave_name, wave_func in [
        ('Square Wave', square_wave),
        ('Triangle Wave', triangle_wave),
        ('Sawtooth Wave', sawtooth_wave)
    ]:
        key = wave_name.lower().replace(' ', '_')
        coeffs = task1_results[key]['analytical']
        plot_waveform_and_fourier(wave_func, coeffs, wave_name)
    
    # Task 2: Gibbs Phenomenon
    print("\n" + "=" * 60)
    print("Task 2: Gibbs Phenomenon")
    print("=" * 60)
    
    gibbs_results = task2_gibbs_phenomenon()
    
    print("\nGibbs Phenomenon Analysis:")
    print("The overshoot at discontinuities does not vanish as N → ∞")
    print("Instead, it converges to approximately 9% of the jump.")
    print()
    print("  N terms | Overshoot | Overshoot %")
    print("  --------|-----------|------------")
    for num_terms in sorted(gibbs_results.keys()):
        data = gibbs_results[num_terms]
        print(f"    {num_terms:4d}  |   {data['overshoot']:.4f}  |   {data['overshoot_percentage']:.2f}%")
    
    print(f"\n  Theoretical Gibbs limit: ≈ 1.0898 (about 9% overshoot)")
    
    # Plot Gibbs phenomenon
    print("\nGenerating Gibbs phenomenon plots...")
    plot_gibbs_phenomenon(gibbs_results)
    
    # Task 3: Parseval's Theorem
    print("\n" + "=" * 60)
    print("Task 3: Parseval's Theorem Verification")
    print("=" * 60)
    
    parseval_results = task3_parseval_verification()
    
    print("\nParseval's Theorem: (1/T)∫|f(x)|²dx = (a₀/2)² + (1/2)Σ(aₙ² + bₙ²)")
    print()
    print("  Waveform       | Mean Square | Parseval Sum | Exact Value")
    print("  ----------------|-------------|--------------|------------")
    for wave_name, data in parseval_results.items():
        display_name = wave_name.replace('_', ' ').title()
        print(f"  {display_name:15s} |   {data['mean_square']:.6f} |    {data['parseval_sum']:.6f} |   {data['exact_mean_square']:.6f}")
    
    # Analytical derivation
    print("\n" + "=" * 60)
    print("ANALYTICAL DERIVATION: Square Wave Coefficients")
    print("=" * 60)
    print("""
    For a square wave with period 2π:
    
        f(x) = +1  for 0 < x < π
        f(x) = -1  for π < x < 2π
    
    The coefficients are:
    
        a₀ = (1/π) ∫₀²π f(x) dx = (1/π)[∫₀π 1 dx + ∫π²π (-1) dx]
           = (1/π)[π - π] = 0
    
        aₙ = (1/π) ∫₀²π f(x)cos(nx) dx
           = (1/π)[∫₀π cos(nx) dx - ∫π²π cos(nx) dx]
           = (1/π)[(1/n)sin(nx)|₀π - (1/n)sin(nx)|π²π]
           = (1/nπ)[sin(nπ) - 0 - sin(2nπ) + sin(nπ)]
           = (2/nπ)sin(nπ) = 0  (for all integer n)
    
        bₙ = (1/π) ∫₀²π f(x)sin(nx) dx
           = (1/π)[∫₀π sin(nx) dx - ∫π²π sin(nx) dx]
           = (1/π)[(-1/n)cos(nx)|₀π + (1/n)cos(nx)|π²π]
           = (1/nπ)[-cos(nπ) + 1 + cos(2nπ) - cos(nπ)]
           = (1/nπ)[2 - 2cos(nπ)]
           = (2/nπ)[1 - cos(nπ)]
    
        For odd n: cos(nπ) = -1, so bₙ = (2/nπ)(1 - (-1)) = 4/(nπ)
        For even n: cos(nπ) = +1, so bₙ = (2/nπ)(1 - 1) = 0
    
    Therefore, the Fourier series is:
        f(x) = (4/π)[sin(x) + (1/3)sin(3x) + (1/5)sin(5x) + ...]
             = (4/π) Σₖ (1/(2k-1))sin((2k-1)x)  for k = 1, 2, 3, ...
    """)
    
    print("\n" + "=" * 60)
    print("SESSION 10 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
