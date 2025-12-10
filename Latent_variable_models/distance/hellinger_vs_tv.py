# Hellinger Distance vs. Total Variation Distance

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

"""
Hellinger Distance vs. Total Variation Distance

Inequalities:
1. For densities p, q:
   ∫ min{p(x), q(x)} dx ≥ (1/2)(∫ √(p(x)q(x)) dx)^2 = (1/2)(1 - (1/2)H(p, q)^2)^2

2. For distributions P, Q:
   (1/2)H(P, Q)^2 ≤ TV(P, Q) ≤ H(P, Q)√(1 - (H(P, Q)^2)/4)

3. H(P, Q)^2 = 0 iff TV(P, Q) = 0
   H(P, Q)^2 = 2 iff TV(P, Q) = 1

4. H(P_n, Q_n) → 0 iff TV(P_n, Q_n) → 0
"""

# Import distance functions
from total_variation_distance import total_variation_distance_density
from hellinger_distance import hellinger_distance_density, hellinger_squared_distance_density

def compute_min_integral(p, q, x_min=-np.inf, x_max=np.inf):
    """
    Compute ∫ min{p(x), q(x)} dx
    """
    integrand = lambda x: min(p(x), q(x))
    result, _ = quad(integrand, x_min, x_max)
    return result

def compute_sqrt_product_integral(p, q, x_min=-np.inf, x_max=np.inf):
    """
    Compute ∫ √(p(x)q(x)) dx
    """
    integrand = lambda x: np.sqrt(p(x) * q(x))
    result, _ = quad(integrand, x_min, x_max)
    return result

def verify_inequalities(p, q, x_min=-10, x_max=10):
    """
    Verify the inequalities between Hellinger distance and Total Variation distance
    """
    # Compute the distances
    tv = total_variation_distance_density(p, q, x_min, x_max)
    h_squared = hellinger_squared_distance_density(p, q, x_min, x_max)
    h = np.sqrt(h_squared)
    
    # Compute the integral terms
    min_integral = compute_min_integral(p, q, x_min, x_max)
    sqrt_product_integral = compute_sqrt_product_integral(p, q, x_min, x_max)
    
    # Verify the inequalities
    inequality1 = min_integral >= 0.5 * (sqrt_product_integral)**2
    inequality2_lower = 0.5 * h_squared <= tv
    inequality2_upper = tv <= h * np.sqrt(1 - h_squared / 4)
    
    print(f"Total Variation Distance (TV): {tv:.6f}")
    print(f"Hellinger Distance (H): {h:.6f}")
    print(f"Hellinger Distance Squared (H²): {h_squared:.6f}")
    print(f"∫ min{p(x), q(x)} dx: {min_integral:.6f}")
    print(f"∫ √(p(x)q(x)) dx: {sqrt_product_integral:.6f}")
    print(f"Inequality 1 (min_integral ≥ 0.5*(sqrt_product_integral)^2): {inequality1}")
    print(f"Inequality 2 lower bound (0.5*H² ≤ TV): {inequality2_lower}")
    print(f"Inequality 2 upper bound (TV ≤ H*sqrt(1 - H²/4)): {inequality2_upper}")
    print(f"Both inequalities hold: {inequality1 and inequality2_lower and inequality2_upper}")
    
    return tv, h, h_squared

def example_normal_distributions(mu1=0, sigma1=1, mu2=1, sigma2=1):
    """
    Example: Compare Hellinger distance and TV distance for normal distributions
    """
    print("Example: Hellinger Distance vs. Total Variation Distance")
    print(f"Distribution P ~ N({mu1}, {sigma1}^2)")
    print(f"Distribution Q ~ N({mu2}, {sigma2}^2)")
    
    # Define the normal distributions
    def p(x):
        return (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu1)/sigma1)**2)
    
    def q(x):
        return (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu2)/sigma2)**2)
    
    # Verify inequalities
    tv, h, h_squared = verify_inequalities(p, q, x_min=-10, x_max=10)
    
    return tv, h, h_squared

def visualize_inequalities():
    """
    Visualize the relationship between Hellinger distance and TV distance
    by plotting the inequalities for a range of H values
    """
    print("\nVisualizing the inequalities...")
    
    # Generate H values between 0 and sqrt(2)
    h = np.linspace(0, np.sqrt(2), 100)
    h_squared = h**2
    
    # Compute the bounds
    lower_bound = 0.5 * h_squared
    upper_bound = h * np.sqrt(1 - h_squared / 4)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(h, lower_bound, label="Lower bound: 0.5*H²")
    plt.plot(h, upper_bound, label="Upper bound: H*sqrt(1 - H²/4)")
    plt.fill_between(h, lower_bound, upper_bound, alpha=0.2)
    plt.xlabel("Hellinger Distance (H)")
    plt.ylabel("Total Variation Distance (TV)")
    plt.title("Inequalities between Hellinger Distance and Total Variation Distance")
    plt.legend()
    plt.grid(True)
    plt.savefig("hellinger_vs_tv.png")
    print("Plot saved as 'hellinger_vs_tv.png'")

if __name__ == "__main__":
    # Run example
    example_normal_distributions()
    print("\n" + "="*50 + "\n")
    
    # Visualize the inequalities
    visualize_inequalities()
