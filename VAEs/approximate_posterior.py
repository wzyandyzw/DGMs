# 知识点2：近似后验推断 (Approximate Posterior Inference) 示例代码
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# 定义真实的后验分布（这里用一个复杂的分布来模拟）
def true_posterior(z, x=2.0):
    """模拟一个复杂的真实后验分布"""
    # 构造一个双峰分布作为复杂后验的示例
    peak1 = multivariate_normal.pdf(z, mean=-2.0, cov=0.5)
    peak2 = multivariate_normal.pdf(z, mean=2.0, cov=0.5)
    return 0.3 * peak1 + 0.7 * peak2

# 定义近似后验分布族（这里使用高斯分布）
def approximate_posterior(z, mu, sigma):
    """高斯近似后验"""
    return multivariate_normal.pdf(z, mean=mu, cov=sigma)

# 使用KL散度来评估近似质量
def kl_divergence(mu, sigma, x=2.0, num_samples=1000):
    """计算近似后验与真实后验之间的KL散度"""
    # 从近似后验中采样
    z_samples = np.random.normal(mu, np.sqrt(sigma), size=num_samples)
    
    # 计算每个样本的对数比率
    log_true = np.log(true_posterior(z_samples, x) + 1e-10)  # 添加小常数避免log(0)
    log_approx = np.log(approximate_posterior(z_samples, mu, sigma) + 1e-10)
    
    # KL散度的蒙特卡洛估计
    kl = np.mean(log_approx - log_true)
    return kl

# 简单的优化：找到最佳的高斯近似参数
# 这里使用网格搜索作为简单的优化方法
mus = np.linspace(-3, 3, 100)
sigmas = np.linspace(0.1, 3, 100)

best_kl = np.inf
best_mu = 0
best_sigma = 1

for mu in mus:
    for sigma in sigmas:
        current_kl = kl_divergence(mu, sigma)
        if current_kl < best_kl:
            best_kl = current_kl
            best_mu = mu
            best_sigma = sigma

print(f"最佳近似参数: mu={best_mu:.2f}, sigma={best_sigma:.2f}")
print(f"最小KL散度: {best_kl:.4f}")

# 可视化结果
z = np.linspace(-5, 5, 1000)
true_p = true_posterior(z)
approx_p = approximate_posterior(z, best_mu, best_sigma)

plt.figure(figsize=(10, 6))
plt.plot(z, true_p, label='真实后验分布', linewidth=2)
plt.plot(z, approx_p, label=f'高斯近似后验 (mu={best_mu:.2f}, sigma={best_sigma:.2f})', linestyle='--', linewidth=2)
plt.title('后验分布及其高斯近似')
plt.xlabel('z')
plt.ylabel('概率密度')
plt.legend()
plt.grid(True)
plt.show()

print("\n注意：这里展示了使用高斯分布近似复杂后验分布的基本思想")
print("在实际应用中，我们会使用更灵活的分布族和更高效的优化算法")