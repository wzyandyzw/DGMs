# Hellinger Distance

import numpy as np
from scipy.integrate import quad

"""
Hellinger Distance (H) between two distributions P and Q.

Definition:
H(P, Q) = (∫ (√p(x) - √q(x))² dx)^(1/2)

Properties:
- H(P, Q) is the L₂ distance between √p and √q
- 0 ≤ H(P, Q)² ≤ 2
- H(P, Q) = 0 iff P = Q
- H is a metric
- H(P, Q)² = 2 iff there exists A ⊂ X with P(A) = 1 and Q(A) = 0
"""

def hellinger_distance_density(p, q, x_min=-np.inf, x_max=np.inf):
    """
    Compute Hellinger Distance between two distributions using their probability density functions.
    
    Args:
        p: Function representing the PDF of distribution P
        q: Function representing the PDF of distribution Q
        x_min: Lower bound of the integration
        x_max: Upper bound of the integration
        
    Returns:
        float: Hellinger Distance between P and Q
    """
    # Using the definition
    integrand = lambda x: (np.sqrt(p(x)) - np.sqrt(q(x)))**2
    result, _ = quad(integrand, x_min, x_max)
    return np.sqrt(result)

def hellinger_distance_histogram(p_hist, q_hist):
    """
    Compute Hellinger Distance between two distributions using their histograms.
    
    Args:
        p_hist: Normalized histogram of distribution P (array-like)
        q_hist: Normalized histogram of distribution Q (array-like)
        
    Returns:
        float: Hellinger Distance between P and Q
    """
    p = np.asarray(p_hist)
    q = np.asarray(q_hist)
    
    if p.shape != q.shape:
        raise ValueError("Histograms must have the same shape")
    
    # Ensure histograms are normalized
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Compute Hellinger distance
    result = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))
    return result

def hellinger_squared_distance_density(p, q, x_min=-np.inf, x_max=np.inf):
    """
    Compute squared Hellinger Distance between two distributions using their PDFs.
    This is sometimes more convenient for calculations.
    
    Args:
        p: Function representing the PDF of distribution P
        q: Function representing the PDF of distribution Q
        x_min: Lower bound of the integration
        x_max: Upper bound of the integration
        
    Returns:
        float: Squared Hellinger Distance between P and Q
    """
    integrand = lambda x: (np.sqrt(p(x)) - np.sqrt(q(x)))**2
    result, _ = quad(integrand, x_min, x_max)
    return result

# Example usage
def example_normal_distributions():
    """
    Example: Compute Hellinger distance between two normal distributions
    """
    print("Example: Hellinger Distance between two normal distributions")
    
    # Define two normal distributions
    mu1, sigma1 = 0, 1
    mu2, sigma2 = 1, 1
    
    p = lambda x: (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu1)/sigma1)**2)
    q = lambda x: (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu2)/sigma2)**2)
    
    # Compute Hellinger distance
    h_distance = hellinger_distance_density(p, q, x_min=-10, x_max=10)
    h_squared_distance = hellinger_squared_distance_density(p, q, x_min=-10, x_max=10)
    
    print(f"Hellinger distance between N({mu1}, {sigma1}^2) and N({mu2}, {sigma2}^2): {h_distance:.6f}")
    print(f"Squared Hellinger distance: {h_squared_distance:.6f}")
    
    return h_distance, h_squared_distance

if __name__ == "__main__":
    example_normal_distributions()
