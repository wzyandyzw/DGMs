# Total Variation Distance

import numpy as np
from scipy.integrate import quad

"""
Total Variation Distance (TV) between two distributions P and Q.

Definition:
TV(P, Q) = sup_{A ⊂ X} |P(A) - Q(A)|

Properties:
- 0 ≤ TV(P, Q) ≤ 1
- TV(P, Q) = 0 iff P = Q
- TV is a metric
- TV(P, Q) = 1 iff there exists A ⊂ X with P(A) = 1 and Q(A) = 0

Alternative forms (Scheffe's Theorem):
1. TV(P, Q) = (1/2) ∫ |p(x) - q(x)| dx
2. TV(P, Q) = 1 - ∫ min{p(x), q(x)} dx
3. TV(P, Q) = P(B) - Q(B) for B := {x : p(x) ≥ q(x)}
"""

def total_variation_distance_density(p, q, x_min=-np.inf, x_max=np.inf):
    """
    Compute Total Variation Distance between two distributions using their probability density functions.
    
    Args:
        p: Function representing the PDF of distribution P
        q: Function representing the PDF of distribution Q
        x_min: Lower bound of the integration
        x_max: Upper bound of the integration
        
    Returns:
        float: Total Variation Distance between P and Q
    """
    # Using the alternative form from Scheffe's Theorem
    integrand = lambda x: abs(p(x) - q(x))
    result, _ = quad(integrand, x_min, x_max)
    return 0.5 * result

def total_variation_distance_histogram(p_hist, q_hist):
    """
    Compute Total Variation Distance between two distributions using their histograms.
    
    Args:
        p_hist: Normalized histogram of distribution P (array-like)
        q_hist: Normalized histogram of distribution Q (array-like)
        
    Returns:
        float: Total Variation Distance between P and Q
    """
    p = np.asarray(p_hist)
    q = np.asarray(q_hist)
    
    if p.shape != q.shape:
        raise ValueError("Histograms must have the same shape")
    
    # Ensure histograms are normalized
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return 0.5 * np.sum(np.abs(p - q))

# Example usage
def example_normal_distributions():
    """
    Example: Compute TV distance between two normal distributions
    """
    print("Example: TV Distance between two normal distributions")
    
    # Define two normal distributions
    mu1, sigma1 = 0, 1
    mu2, sigma2 = 1, 1
    
    p = lambda x: (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu1)/sigma1)**2)
    q = lambda x: (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu2)/sigma2)**2)
    
    # Compute TV distance
    tv_distance = total_variation_distance_density(p, q, x_min=-10, x_max=10)
    print(f"TV distance between N({mu1}, {sigma1}^2) and N({mu2}, {sigma2}^2): {tv_distance:.6f}")
    
    return tv_distance

if __name__ == "__main__":
    example_normal_distributions()
