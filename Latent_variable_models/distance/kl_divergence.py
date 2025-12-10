# Kullback-Liebler (KL) Divergence

import numpy as np
from scipy.integrate import quad

"""
Kullback-Liebler Divergence (KL) between two distributions P and Q.

Definition:
KL(P||Q) = KL(p||q) = ∫ p(x) log(p(x)/q(x)) dx
(Analogous for discrete distributions.)

Properties:
- KL-divergence is not symmetric
- KL-divergence is not a metric
- KL(P||Q) ≥ 0, with equality iff P = Q

Alternative interpretation:
KL(P||Q) = E_p [log(p(X)/q(X))]
"""

def kl_divergence_density(p, q, x_min=-np.inf, x_max=np.inf, eps=1e-10):
    """
    Compute KL Divergence between two distributions using their probability density functions.
    
    Args:
        p: Function representing the PDF of distribution P
        q: Function representing the PDF of distribution Q
        x_min: Lower bound of the integration
        x_max: Upper bound of the integration
        eps: Small epsilon to avoid division by zero or log(0)
        
    Returns:
        float: KL Divergence between P and Q
    """
    # Using the definition with numerical stability considerations
    integrand = lambda x: p(x) * np.log( (p(x) + eps) / (q(x) + eps) )
    result, _ = quad(integrand, x_min, x_max)
    return result

def kl_divergence_histogram(p_hist, q_hist, eps=1e-10):
    """
    Compute KL Divergence between two distributions using their histograms.
    
    Args:
        p_hist: Normalized histogram of distribution P (array-like)
        q_hist: Normalized histogram of distribution Q (array-like)
        eps: Small epsilon to avoid division by zero or log(0)
        
    Returns:
        float: KL Divergence between P and Q
    """
    p = np.asarray(p_hist)
    q = np.asarray(q_hist)
    
    if p.shape != q.shape:
        raise ValueError("Histograms must have the same shape")
    
    # Ensure histograms are normalized
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Compute KL divergence with numerical stability
    kl = np.sum(p * np.log( (p + eps) / (q + eps) ))
    return kl

def kl_divergence_discrete(p_pmf, q_pmf, eps=1e-10):
    """
    Compute KL Divergence between two discrete distributions using their PMFs.
    
    Args:
        p_pmf: Probability mass function of distribution P (array-like)
        q_pmf: Probability mass function of distribution Q (array-like)
        eps: Small epsilon to avoid division by zero or log(0)
        
    Returns:
        float: KL Divergence between P and Q
    """
    p = np.asarray(p_pmf)
    q = np.asarray(q_pmf)
    
    if p.shape != q.shape:
        raise ValueError("PMFs must have the same shape")
    
    # Ensure PMFs sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Compute KL divergence with numerical stability
    kl = np.sum(p * np.log( (p + eps) / (q + eps) ))
    return kl

# Example usage
def example_discrete_distributions():
    """
    Example: Compute KL divergence between two discrete distributions
    """
    print("Example: KL Divergence between two discrete distributions")
    
    # Define two discrete distributions
    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.3, 0.3, 0.4])
    
    # Compute KL divergence
    kl_pq = kl_divergence_discrete(p, q)
    kl_qp = kl_divergence_discrete(q, p)
    
    print(f"Distribution P: {p}")
    print(f"Distribution Q: {q}")
    print(f"KL(P||Q): {kl_pq:.6f}")
    print(f"KL(Q||P): {kl_qp:.6f}")
    print(f"Note: KL divergence is not symmetric (KL(P||Q) != KL(Q||P))")
    
    return kl_pq, kl_qp

if __name__ == "__main__":
    example_discrete_distributions()
