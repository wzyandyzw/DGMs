# KL Divergence vs. Total Variation and Hellinger Distances

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

"""
KL Divergence vs. Total Variation and Hellinger Distances

Inequalities:
1. For any P, Q:
   H(P, Q)² ≤ KL(P||Q)

2. Pinsker's inequality:
   TV(P, Q)² ≤ KL(P||Q)/2

Product Distributions (Tensorization):
For P = ⊗_{i=1}^n P_i (densities p(x₁,…,xₙ)=p₁(x₁)…pₙ(xₙ)):
- TV(⊗₁ⁿ P_i, ⊗₁ⁿ Q_i) ≤ Σ₁ⁿ TV(P_i, Q_i)
- H(⊗₁ⁿ P_i, ⊗₁ⁿ Q_i)² ≤ Σ₁ⁿ H(P_i, Q_i)²
- KL(⊗₁ⁿ P_i, ⊗₁ⁿ Q_i) = Σ₁ⁿ KL(P_i, Q_i)
"""

# Import distance functions
from kl_divergence import kl_divergence_density
from total_variation_distance import total_variation_distance_density
from hellinger_distance import hellinger_distance_density, hellinger_squared_distance_density

def verify_inequalities(p, q, x_min=-np.inf, x_max=np.inf):
    """
    Verify the inequalities between KL divergence, Hellinger distance, and TV distance
    """
    # Compute the distances
    kl = kl_divergence_density(p, q, x_min, x_max)
    tv = total_variation_distance_density(p, q, x_min, x_max)
    h = hellinger_distance_density(p, q, x_min, x_max)
    h_squared = h**2
    
    # Verify the inequalities
    inequality1 = h_squared <= kl
    pinsker_inequality = tv**2 <= kl / 2
    
    print(f"KL Divergence (KL): {kl:.6f}")
    print(f"Total Variation Distance (TV): {tv:.6f}")
    print(f"Hellinger Distance (H): {h:.6f}")
    print(f"Hellinger Distance Squared (H²): {h_squared:.6f}")
    print(f"Inequality 1 (H² ≤ KL): {inequality1}")
    print(f"Pinsker's inequality (TV² ≤ KL/2): {pinsker_inequality}")
    print(f"Both inequalities hold: {inequality1 and pinsker_inequality}")
    
    return kl, tv, h, h_squared

def example_normal_distributions(mu1=0, sigma1=1, mu2=1, sigma2=1):
    """
    Example: Compare KL divergence, TV distance, and Hellinger distance for normal distributions
    """
    print("Example: KL Divergence vs. TV Distance vs. Hellinger Distance")
    print(f"Distribution P ~ N({mu1}, {sigma1}^2)")
    print(f"Distribution Q ~ N({mu2}, {sigma2}^2)")
    
    # Define the normal distributions
    def p(x):
        return (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu1)/sigma1)**2)
    
    def q(x):
        return (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu2)/sigma2)**2)
    
    # Verify inequalities
    kl, tv, h, h_squared = verify_inequalities(p, q, x_min=-10, x_max=10)
    
    return kl, tv, h, h_squared

def example_tensorization():
    """
    Example: Demonstrate tensorization properties for product distributions
    """
    print("\nExample: Tensorization Properties for Product Distributions")
    
    # Define two pairs of normal distributions
    mu1_1, sigma1_1 = 0, 1
    mu1_2, sigma1_2 = 1, 1
    
    mu2_1, sigma2_1 = 0, 1
    mu2_2, sigma2_2 = 2, 1
    
    # Define individual distributions
    def p1(x):
        return (1/(sigma1_1*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu1_1)/sigma1_1)**2)
    
    def q1(x):
        return (1/(sigma1_2*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu1_2)/sigma1_2)**2)
    
    def p2(x):
        return (1/(sigma2_1*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu2_1)/sigma2_1)**2)
    
    def q2(x):
        return (1/(sigma2_2*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu2_2)/sigma2_2)**2)
    
    # Define product distributions (2D normal distributions)
    def p_product(x1, x2):
        return p1(x1) * p2(x2)
    
    def q_product(x1, x2):
        return q1(x1) * q2(x2)
    
    # Compute distances for individual distributions
    kl1 = kl_divergence_density(p1, q1, x_min=-10, x_max=10)
    kl2 = kl_divergence_density(p2, q2, x_min=-10, x_max=10)
    
    tv1 = total_variation_distance_density(p1, q1, x_min=-10, x_max=10)
    tv2 = total_variation_distance_density(p2, q2, x_min=-10, x_max=10)
    
    h_squared1 = hellinger_squared_distance_density(p1, q1, x_min=-10, x_max=10)
    h_squared2 = hellinger_squared_distance_density(p2, q2, x_min=-10, x_max=10)
    
    # For product distributions, we'll use the tensorization properties directly
    kl_product = kl1 + kl2  # Exact tensorization
    tv_product_upper = tv1 + tv2  # Upper bound
    h_squared_product_upper = h_squared1 + h_squared2  # Upper bound
    
    print(f"Distribution P1 ~ N({mu1_1}, {sigma1_1}^2)")
    print(f"Distribution Q1 ~ N({mu1_2}, {sigma1_2}^2)")
    print(f"Distribution P2 ~ N({mu2_1}, {sigma2_1}^2)")
    print(f"Distribution Q2 ~ N({mu2_2}, {sigma2_2}^2)")
    print(f"Product Distribution P = P1 × P2")
    print(f"Product Distribution Q = Q1 × Q2")
    print(f"\nIndividual Distances:")
    print(f"KL(P1||Q1): {kl1:.6f}")
    print(f"KL(P2||Q2): {kl2:.6f}")
    print(f"TV(P1, Q1): {tv1:.6f}")
    print(f"TV(P2, Q2): {tv2:.6f}")
    print(f"H²(P1, Q1): {h_squared1:.6f}")
    print(f"H²(P2, Q2): {h_squared2:.6f}")
    print(f"\nProduct Distributions (Tensorization):")
    print(f"KL(P||Q) = KL(P1||Q1) + KL(P2||Q2): {kl_product:.6f}")
    print(f"TV(P, Q) ≤ TV(P1, Q1) + TV(P2, Q2): TV(P, Q) ≤ {tv_product_upper:.6f}")
    print(f"H²(P, Q) ≤ H²(P1, Q1) + H²(P2, Q2): H²(P, Q) ≤ {h_squared_product_upper:.6f}")
    
    return kl_product, tv_product_upper, h_squared_product_upper

def visualize_pinsker_inequality():
    """
    Visualize Pinsker's inequality by plotting TV² vs KL/2
    """
    print("\nVisualizing Pinsker's Inequality...")
    
    # Generate KL values
    kl_values = np.linspace(0, 5, 100)
    
    # Compute TV² upper bound from KL/2
    tv_squared_upper = kl_values / 2
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(kl_values, tv_squared_upper, label="Pinsker's bound: TV² ≤ KL/2")
    plt.fill_between(kl_values, 0, tv_squared_upper, alpha=0.2)
    plt.xlabel("KL Divergence (KL)")
    plt.ylabel("Total Variation Distance Squared (TV²)")
    plt.title("Pinsker's Inequality: TV² ≤ KL/2")
    plt.legend()
    plt.grid(True)
    plt.savefig("pinsker_inequality.png")
    print("Plot saved as 'pinsker_inequality.png'")

if __name__ == "__main__":
    # Run example 1: Inequalities
    example_normal_distributions()
    print("\n" + "="*50 + "\n")
    
    # Run example 2: Tensorization
    example_tensorization()
    print("\n" + "="*50 + "\n")
    
    # Visualize Pinsker's inequality
    visualize_pinsker_inequality()
