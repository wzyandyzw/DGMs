# Jensen-Shannon (JS) Divergence

import numpy as np
from scipy.integrate import quad

"""
Jensen-Shannon Divergence (JS) between two distributions P and Q.

Definition:
JS(P||Q) = JS(p||q) = (1/2)KL(p||(p+q)/2) + (1/2)KL(q||(p+q)/2)

Properties:
- JS-divergence is symmetric
- JS-divergence is bounded: [0, log 2]
"""

# Import KL divergence functions
from kl_divergence import kl_divergence_density, kl_divergence_histogram, kl_divergence_discrete

def js_divergence_density(p, q, x_min=-np.inf, x_max=np.inf, eps=1e-10):
    """
    Compute JS Divergence between two distributions using their probability density functions.
    
    Args:
        p: Function representing the PDF of distribution P
        q: Function representing the PDF of distribution Q
        x_min: Lower bound of the integration
        x_max: Upper bound of the integration
        eps: Small epsilon to avoid division by zero or log(0)
        
    Returns:
        float: JS Divergence between P and Q
    """
    # Define the average distribution
    def m(x):
        return 0.5 * (p(x) + q(x))
    
    # Compute JS divergence using the definition
    kl_pm = kl_divergence_density(p, m, x_min, x_max, eps)
    kl_qm = kl_divergence_density(q, m, x_min, x_max, eps)
    
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return js

def js_divergence_histogram(p_hist, q_hist, eps=1e-10):
    """
    Compute JS Divergence between two distributions using their histograms.
    
    Args:
        p_hist: Normalized histogram of distribution P (array-like)
        q_hist: Normalized histogram of distribution Q (array-like)
        eps: Small epsilon to avoid division by zero or log(0)
        
    Returns:
        float: JS Divergence between P and Q
    """
    # Compute JS divergence using the definition
    kl_pq = kl_divergence_histogram(p_hist, q_hist, eps)
    kl_qp = kl_divergence_histogram(q_hist, p_hist, eps)
    
    js = 0.5 * kl_pq + 0.5 * kl_qp
    return js

def js_divergence_discrete(p_pmf, q_pmf, eps=1e-10):
    """
    Compute JS Divergence between two discrete distributions using their PMFs.
    
    Args:
        p_pmf: Probability mass function of distribution P (array-like)
        q_pmf: Probability mass function of distribution Q (array-like)
        eps: Small epsilon to avoid division by zero or log(0)
        
    Returns:
        float: JS Divergence between P and Q
    """
    # Define the average distribution
    m_pmf = 0.5 * (np.asarray(p_pmf) + np.asarray(q_pmf))
    
    # Compute JS divergence using the definition
    kl_pm = kl_divergence_discrete(p_pmf, m_pmf, eps)
    kl_qm = kl_divergence_discrete(q_pmf, m_pmf, eps)
    
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return js

def js_divergence_discrete_simple(p_pmf, q_pmf, eps=1e-10):
    """
    A simpler implementation of JS divergence for discrete distributions.
    Since JS divergence is symmetric, we can compute it as the average of KL divergences.
    
    Args:
        p_pmf: Probability mass function of distribution P (array-like)
        q_pmf: Probability mass function of distribution Q (array-like)
        eps: Small epsilon to avoid division by zero or log(0)
        
    Returns:
        float: JS Divergence between P and Q
    """
    # Compute JS divergence as the average of KL divergences
    kl_pq = kl_divergence_discrete(p_pmf, q_pmf, eps)
    kl_qp = kl_divergence_discrete(q_pmf, p_pmf, eps)
    
    js = 0.5 * kl_pq + 0.5 * kl_qp
    return js

# Example usage
def example_discrete_distributions():
    """
    Example: Compute JS divergence between two discrete distributions
    """
    print("Example: JS Divergence between two discrete distributions")
    
    # Define two discrete distributions
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.3, 0.3, 0.4])
    
    # Compute JS divergence using both implementations
    js1 = js_divergence_discrete(p, q)
    js2 = js_divergence_discrete_simple(p, q)
    
    print(f"Distribution P: {p}")
    print(f"Distribution Q: {q}")
    print(f"JS(P||Q) (using average distribution): {js1:.6f}")
    print(f"JS(P||Q) (using symmetric KL): {js2:.6f}")
    print(f"JS divergence is symmetric: JS(P||Q) = JS(Q||P)")
    
    return js1

if __name__ == "__main__":
    example_discrete_distributions()
