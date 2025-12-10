# Wasserstein Distance

import numpy as np
from scipy import stats
from scipy import integrate

"""
Wasserstein Distance (Earth Mover's Distance)

Definition:
For probability distributions P and Q on a metric space (X, d), the Wasserstein distance of order p is:
W_p(P, Q) = (inf_{γ ∈ Π(P, Q)} ∫_{X×X} d(x, y)^p dγ(x, y))^{1/p}

where Π(P, Q) is the set of all couplings of P and Q (joint distributions with marginals P and Q),
and d(x, y) is the metric on X.

For p=1 (Earth Mover's Distance):
W_1(P, Q) = inf_{γ ∈ Π(P, Q)} ∫_{X×X} |x - y| dγ(x, y)

For continuous distributions with CDF F and G:
W_1(P, Q) = ∫_{-∞}^{∞} |F(x) - G(x)| dx

Properties:
1. W_p(P, Q) ≥ 0
2. W_p(P, Q) = 0 if and only if P = Q (almost everywhere)
3. W_p(P, Q) = W_p(Q, P) (symmetry)
4. W_p(P, R) ≤ W_p(P, Q) + W_p(Q, R) (triangle inequality)
5. Unlike KL divergence, TV distance, and Hellinger distance, Wasserstein distance is sensitive to the geometry of the underlying space.
6. It can be used to compare distributions that have disjoint supports.
"""

def wasserstein_distance_cdf(p_cdf, q_cdf, p=1, x_min=-10, x_max=10, n_points=1000):
    """
    Compute the Wasserstein distance of order p between two distributions using their CDFs.
    This method is particularly efficient for 1-dimensional distributions.
    
    Parameters:
    p_cdf: callable, CDF of distribution P
    q_cdf: callable, CDF of distribution Q
    p: int, order of the Wasserstein distance (default: 1)
    x_min: float, lower bound of the integration domain
    x_max: float, upper bound of the integration domain
    n_points: int, number of points to use for numerical integration
    
    Returns:
    float: Wasserstein distance of order p
    """
    def integrand(x):
        return abs(p_cdf(x) - q_cdf(x))**p
    
    distance, error = integrate.quad(integrand, x_min, x_max)
    return distance**(1/p)

def wasserstein_distance_empirical(p_samples, q_samples, p=1, n_bins=100):
    """
    Compute the Wasserstein distance of order p between two empirical distributions.
    This method uses histogram binning and optimal transport.
    
    Parameters:
    p_samples: array-like, samples from distribution P
    q_samples: array-like, samples from distribution Q
    p: int, order of the Wasserstein distance (default: 1)
    n_bins: int, number of bins to use for histogram estimation
    
    Returns:
    float: Wasserstein distance of order p
    """
    # Convert samples to numpy arrays
    p_samples = np.asarray(p_samples)
    q_samples = np.asarray(q_samples)
    
    # Create histogram bins
    min_val = min(np.min(p_samples), np.min(q_samples))
    max_val = max(np.max(p_samples), np.max(q_samples))
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    # Compute bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Compute histograms (probability masses)
    p_hist, _ = np.histogram(p_samples, bins=bins, density=False)
    q_hist, _ = np.histogram(q_samples, bins=bins, density=False)
    
    # Normalize histograms to get probability masses
    p_mass = p_hist / np.sum(p_hist)
    q_mass = q_hist / np.sum(q_hist)
    
    # Compute transport cost matrix (Euclidean distance raised to the p-th power)
    cost_matrix = np.abs(bin_centers[:, np.newaxis] - bin_centers[np.newaxis, :])**p
    
    # Compute the Wasserstein distance using optimal transport
    # For 1D distributions, we can use a more efficient approach
    return wasserstein_distance_1d_hist(p_mass, q_mass, bin_centers, p)

def wasserstein_distance_1d_hist(p_mass, q_mass, bin_centers, p=1):
    """
    Compute the Wasserstein distance of order p between two 1D histograms.
    This is an efficient implementation for 1D distributions.
    
    Parameters:
    p_mass: array-like, probability masses for distribution P
    q_mass: array-like, probability masses for distribution Q
    bin_centers: array-like, bin centers for both histograms
    p: int, order of the Wasserstein distance (default: 1)
    
    Returns:
    float: Wasserstein distance of order p
    """
    # Compute cumulative distributions
    p_cdf = np.cumsum(p_mass)
    q_cdf = np.cumsum(q_mass)
    
    # For p=1, we can compute it directly using the formula for 1D CDFs
    if p == 1:
        # Use linear interpolation to compute the integral
        return integrate.simps(np.abs(p_cdf - q_cdf), bin_centers)
    else:
        # For higher p, we need to use optimal transport
        # Here we use a simplified version based on cumulative distributions
        # This is not exact but works well for histograms
        return np.sum(np.abs(p_cdf - q_cdf) * np.abs(bin_centers[1] - bin_centers[0]))**(1/p)

def example_normal_distributions(mu1=0, sigma1=1, mu2=1, sigma2=1):
    """
    Example: Compare Wasserstein distance between normal distributions
    """
    print(f"Example: Wasserstein Distance between Normal Distributions")
    print(f"Distribution P ~ N({mu1}, {sigma1}^2)")
    print(f"Distribution Q ~ N({mu2}, {sigma2}^2)")
    
    # Define the CDFs
    def p_cdf(x):
        return stats.norm.cdf(x, mu1, sigma1)
    
    def q_cdf(x):
        return stats.norm.cdf(x, mu2, sigma2)
    
    # Compute Wasserstein distances for different orders
    w1 = wasserstein_distance_cdf(p_cdf, q_cdf, p=1)
    w2 = wasserstein_distance_cdf(p_cdf, q_cdf, p=2)
    
    print(f"Wasserstein Distance (p=1): {w1:.6f}")
    print(f"Wasserstein Distance (p=2): {w2:.6f}")
    
    return w1, w2

def example_empirical_distributions():
    """
    Example: Compare Wasserstein distance between empirical distributions
    """
    print(f"\nExample: Wasserstein Distance between Empirical Distributions")
    
    # Generate samples from two different distributions
    np.random.seed(42)
    p_samples = np.random.normal(0, 1, 1000)
    q_samples = np.random.normal(1, 1, 1000)
    
    # Compute Wasserstein distances
    w1 = wasserstein_distance_empirical(p_samples, q_samples, p=1)
    w2 = wasserstein_distance_empirical(p_samples, q_samples, p=2)
    
    print(f"Wasserstein Distance (p=1): {w1:.6f}")
    print(f"Wasserstein Distance (p=2): {w2:.6f}")
    
    return w1, w2

def example_disjoint_supports():
    """
    Example: Wasserstein distance for distributions with disjoint supports
    """
    print(f"\nExample: Wasserstein Distance for Distributions with Disjoint Supports")
    
    # Generate samples from two distributions with disjoint supports
    np.random.seed(42)
    p_samples = np.random.normal(0, 0.5, 1000)
    q_samples = np.random.normal(5, 0.5, 1000)
    
    # Compute Wasserstein distances
    w1 = wasserstein_distance_empirical(p_samples, q_samples, p=1)
    w2 = wasserstein_distance_empirical(p_samples, q_samples, p=2)
    
    print(f"Wasserstein Distance (p=1): {w1:.6f}")
    print(f"Wasserstein Distance (p=2): {w2:.6f}")
    
    # Compare with other distances (conceptually)
    print("\nNote: Unlike KL divergence and TV distance, which would be maximum or undefined")
    print("for distributions with disjoint supports, Wasserstein distance provides a")
    print("meaningful measure based on the geometric distance between the supports.")
    
    return w1, w2

if __name__ == "__main__":
    # Run example 1: Normal distributions
    example_normal_distributions()
    print("\n" + "="*50 + "\n")
    
    # Run example 2: Empirical distributions
    example_empirical_distributions()
    print("\n" + "="*50 + "\n")
    
    # Run example 3: Distributions with disjoint supports
    example_disjoint_supports()
