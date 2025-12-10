# 知识点1：后验推断 (Posterior Inference) 示例代码
import numpy as np
from scipy.stats import multivariate_normal

# 定义一个简单的GMM模型参数
means = np.array([[0, 0], [5, 5]])  # 两个高斯分量的均值
covs = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])  # 两个高斯分量的协方差矩阵
weights = np.array([0.6, 0.4])  # 混合权重

# 计算后验概率 p(z|x)
def compute_posterior(x, means, covs, weights):
    """计算GMM模型中每个分量的后验概率"""
    n_components = len(means)
    n_samples = x.shape[0]
    
    # 计算每个样本在每个分量下的似然
    likelihoods = np.zeros((n_samples, n_components))
    for i in range(n_components):
        likelihoods[:, i] = multivariate_normal.pdf(x, mean=means[i], cov=covs[i])
    
    # 计算每个样本的总似然（混合似然）
    total_likelihood = np.sum(likelihoods * weights, axis=1, keepdims=True)
    
    # 计算后验概率（责任）
    posterior = (likelihoods * weights) / total_likelihood
    
    return posterior

# 生成一些样本
np.random.seed(42)
x = np.random.randn(10, 2)  # 10个2维样本

# 计算后验概率
posterior = compute_posterior(x, means, covs, weights)

print("样本后验概率:")
print(posterior)
print("\n每个样本的责任总和:")
print(np.sum(posterior, axis=1))  # 应该都等于1