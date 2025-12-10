# f-divergence

import numpy as np
from scipy.integrate import quad

"""
f-divergence (Ali-Silvey divergence) between two distributions P and Q.

Definition:
D_f(P||Q) = D_f(p||q) = ∫ q(x)f(p(x)/q(x)) dx
where f : R₊ → R is convex and f(1) = 0.

Examples:
- KL-divergence: f(t) = t log t
- TV-distance: f(t) = (1/2)|t - 1|
- Hellinger distance: f(t) = (√t - 1)^2 = t + 1 - 2√t
- χ²-divergence: f(t) = (1/2)(t - 1)^2
"""

def f_divergence_density(p, q, f, x_min=-np.inf, x_max=np.inf, eps=1e-10):
    """
    Compute f-divergence between two distributions using their probability density functions.
    
    Args:
        p: Function representing the PDF of distribution P
        q: Function representing the PDF of distribution Q
        f: Convex function with f(1) = 0
        x_min: Lower bound of the integration
        x_max: Upper bound of the integration
        eps: Small epsilon to avoid division by zero
        
    Returns:
        float: f-divergence between P and Q
    """
    # Define the integrand
    def integrand(x):
        q_x = q(x)
        if q_x < eps:
            return 0.0  # Avoid division by zero
        p_x = p(x)
        t = p_x / q_x
        return q_x * f(t)
    
    result, _ = quad(integrand, x_min, x_max)
    return result

def f_divergence_histogram(p_hist, q_hist, f, eps=1e-10):
    """
    Compute f-divergence between two distributions using their histograms.
    
    Args:
        p_hist: Normalized histogram of distribution P (array-like)
        q_hist: Normalized histogram of distribution Q (array-like)
        f: Convex function with f(1) = 0
        eps: Small epsilon to avoid division by zero
        
    Returns:
        float: f-divergence between P and Q
    """
    p = np.asarray(p_hist)
    q = np.asarray(q_hist)
    
    if p.shape != q.shape:
        raise ValueError("Histograms must have the same shape")
    
    # Ensure histograms are normalized
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Compute f-divergence
    divergence = 0.0
    for p_i, q_i in zip(p, q):
        if q_i < eps:
            continue  # Avoid division by zero
        t = p_i / q_i
        divergence += q_i * f(t)
    
    return divergence

# Specific examples of f-divergence

def kl_divergence_f(t):
    """f(t) = t log t for KL divergence"""
    return t * np.log(t)

def tv_distance_f(t):
    """f(t) = (1/2)|t - 1| for Total Variation distance"""
    return 0.5 * np.abs(t - 1)

def hellinger_distance_f(t):
    """f(t) = (√t - 1)^2 for Hellinger distance"""
    return (np.sqrt(t) - 1)**2

def chi_squared_divergence_f(t):
    """f(t) = (1/2)(t - 1)^2 for χ²-divergence"""
    return 0.5 * (t - 1)**2

# Example usage
def example_f_divergences():
    """
    Example: Compute different f-divergences between two distributions
    """
    print("Example: Different f-divergences between two normal distributions")
    
    # Define two normal distributions
    mu1, sigma1 = 0, 1
    mu2, sigma2 = 1, 1
    
    p = lambda x: (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu1)/sigma1)**2)
    q = lambda x: (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu2)/sigma2)**2)
    
    # Compute different f-divergences
    kl_div = f_divergence_density(p, q, kl_divergence_f, x_min=-10, x_max=10)
    tv_div = f_divergence_density(p, q, tv_distance_f, x_min=-10, x_max=10)
    hellinger_div = f_divergence_density(p, q, hellinger_distance_f, x_min=-10, x_max=10)
    chi2_div = f_divergence_density(p, q, chi_squared_divergence_f, x_min=-10, x_max=10)
    
    print(f"Distribution P ~ N({mu1}, {sigma1}^2)")
    print(f"Distribution Q ~ N({mu2}, {sigma2}^2)")
    print(f"KL divergence: {kl_div:.6f}")
    print(f"Total Variation distance: {tv_div:.6f}")
    print(f"Hellinger distance squared: {hellinger_div:.6f}")
    print(f"χ²-divergence: {chi2_div:.6f}")
    
    return kl_div, tv_div, hellinger_div, chi2_div

if __name__ == "__main__":
    example_f_divergences()
