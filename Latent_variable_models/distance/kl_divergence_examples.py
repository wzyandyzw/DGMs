# KL Divergence Examples

import numpy as np
from scipy.stats import norm
from numpy.linalg import inv, det

"""
KL Divergence Examples

Example 1: Bernoulli distributions
p(0) = p(1) = 1/2, q(0) = (1-ε)/2, q(1) = (1+ε)/2
Exact expressions:
- KL(p||q) = -1/2 log(1 - ε²)
- KL(q||p) = 1/2 log(1 - ε²) + ε/2 log((1+ε)/(1-ε))

Example 2: Multivariate normal distributions
P ~ N(μ₀, Σ₀), Q ~ N(μ₁, Σ₁)
Exact expression:
2KL(P||Q) = tr(Σ₁⁻¹Σ₀) + (μ₁ - μ₀)ᵀΣ₁⁻¹(μ₁ - μ₀) + ln(det(Σ₁)/det(Σ₀)) - d
"""

def kl_bernoulli_example(epsilon=0.1):
    """
    Example: Compute KL divergence between two Bernoulli distributions
    as described in the knowledge.md file.
    
    Args:
        epsilon: Parameter for the Bernoulli distributions
        
    Returns:
        tuple: (kl_pq, kl_qp)
    """
    print(f"Example 1: KL Divergence between Bernoulli distributions with ε = {epsilon}")
    
    # Define the distributions
    p = np.array([0.5, 0.5])
    q = np.array([(1-epsilon)/2, (1+epsilon)/2])
    
    # Compute KL divergence using the exact expressions
    kl_pq_exact = -0.5 * np.log(1 - epsilon**2)
    kl_qp_exact = 0.5 * np.log(1 - epsilon**2) + (epsilon/2) * np.log((1+epsilon)/(1-epsilon))
    
    # Compute KL divergence using numerical methods
    def compute_kl(p, q):
        """Helper function to compute KL divergence numerically"""
        return np.sum(p * np.log(p / q))
    
    kl_pq_numeric = compute_kl(p, q)
    kl_qp_numeric = compute_kl(q, p)
    
    print(f"p: {p}")
    print(f"q: {q}")
    print(f"KL(p||q) (exact): {kl_pq_exact:.6f}")
    print(f"KL(p||q) (numeric): {kl_pq_numeric:.6f}")
    print(f"KL(q||p) (exact): {kl_qp_exact:.6f}")
    print(f"KL(q||p) (numeric): {kl_qp_numeric:.6f}")
    
    return kl_pq_exact, kl_qp_exact

def kl_multivariate_normal_example():
    """
    Example: Compute KL divergence between two multivariate normal distributions
    as described in the knowledge.md file.
    
    Returns:
        float: KL divergence between the two distributions
    """
    print("Example 2: KL Divergence between multivariate normal distributions")
    
    # Define the distributions
    # 2-dimensional example
    d = 2
    
    mu0 = np.array([0, 0])
    sigma0 = np.array([[1, 0.5], [0.5, 2]])
    
    mu1 = np.array([1, 2])
    sigma1 = np.array([[2, 0], [0, 3]])
    
    # Compute KL divergence using the exact expression
    sigma1_inv = inv(sigma1)
    
    term1 = np.trace(sigma1_inv @ sigma0)
    term2 = (mu1 - mu0).T @ sigma1_inv @ (mu1 - mu0)
    term3 = np.log(det(sigma1) / det(sigma0))
    
    kl_pq_exact = 0.5 * (term1 + term2 + term3 - d)
    
    print(f"Distribution P ~ N(μ0={mu0}, Σ0={sigma0})")
    print(f"Distribution Q ~ N(μ1={mu1}, Σ1={sigma1})")
    print(f"KL(P||Q) (exact): {kl_pq_exact:.6f}")
    
    return kl_pq_exact

def kl_univariate_normal_example():
    """
    Example: Compute KL divergence between two univariate normal distributions
    (a special case of the multivariate normal example)
    
    Returns:
        float: KL divergence between the two distributions
    """
    print("Example 3: KL Divergence between univariate normal distributions")
    
    # Define the distributions
    mu0, sigma0 = 0, 1
    mu1, sigma1 = 2, 3
    
    # Compute KL divergence using the exact expression for univariate case
    kl_pq_exact = np.log(sigma1/sigma0) + (sigma0**2 + (mu0 - mu1)**2)/(2*sigma1**2) - 0.5
    
    # Compute using scipy's norm.entropy
    # Note: norm.entropy gives H(p), and KL(p||q) = H(p, q) - H(p)
    # where H(p, q) is the cross-entropy
    h_p = norm.entropy(loc=mu0, scale=sigma0)
    cross_entropy_pq = -np.mean(norm.logpdf(np.random.normal(mu0, sigma0, 10000), loc=mu1, scale=sigma1))
    kl_pq_numeric = cross_entropy_pq - h_p
    
    print(f"Distribution P ~ N(μ0={mu0}, σ0²={sigma0**2})")
    print(f"Distribution Q ~ N(μ1={mu1}, σ1²={sigma1**2})")
    print(f"KL(P||Q) (exact): {kl_pq_exact:.6f}")
    print(f"KL(P||Q) (numeric): {kl_pq_numeric:.6f}")
    
    return kl_pq_exact

if __name__ == "__main__":
    # Run all examples
    kl_bernoulli_example(epsilon=0.1)
    print("\n" + "="*50 + "\n")
    kl_multivariate_normal_example()
    print("\n" + "="*50 + "\n")
    kl_univariate_normal_example()
