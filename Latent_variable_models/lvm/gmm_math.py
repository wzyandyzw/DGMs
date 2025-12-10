# 知识点4：高斯混合模型的数学表达
"""
- 高斯密度：N(x; μ, Σ) = 1/√((2π)^d · det(Σ)) · exp(-1/2(x - μ)⊤Σ⁻¹(x - μ))
- GMM联合分布：p(x) = Σ_z p(x, z) = Σ_z p(z) p(x|z) = Σ_{k=1}^K p(z = k) N(x; μ_k*, Σ_k*) = Σ_{k=1}^K π_k* N(x; μ_k*, Σ_k*)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 计算高斯密度函数
def gaussian_density(x, mu, sigma):
    """计算多元高斯分布的密度函数
    
    参数:
        x: 数据点，形状为(n_samples, n_features)
        mu: 均值向量，形状为(n_features,)
        sigma: 协方差矩阵，形状为(n_features, n_features)
        
    返回:
        密度值，形状为(n_samples,)
    """
    d = len(mu)
    det_sigma = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)
    
    # 计算指数部分
    exponent = -0.5 * np.sum((x - mu) @ inv_sigma * (x - mu), axis=1)
    
    # 计算归一化常数
    const = 1 / np.sqrt((2 * np.pi)**d * det_sigma)
    
    # 计算密度
    density = const * np.exp(exponent)
    
    return density

# 计算GMM密度函数
def gmm_density(x, weights, means, covariances):
    """计算高斯混合模型的密度函数
    
    参数:
        x: 数据点，形状为(n_samples, n_features)
        weights: 混合权重，形状为(n_components,)
        means: 均值向量列表，形状为(n_components, n_features)
        covariances: 协方差矩阵列表，形状为(n_components, n_features, n_features)
        
    返回:
        GMM密度值，形状为(n_samples,)
    """
    n_samples, n_features = x.shape
    n_components = len(weights)
    
    # 初始化密度数组
    density = np.zeros(n_samples)
    
    # 对每个混合成分计算密度并加权求和
    for k in range(n_components):
        density += weights[k] * gaussian_density(x, means[k], covariances[k])
    
    return density

# 可视化高斯混合模型
def visualize_gmm(weights, means, covariances):
    """可视化二维高斯混合模型
    
    参数:
        weights: 混合权重，形状为(n_components,)
        means: 均值向量列表，形状为(n_components, 2)
        covariances: 协方差矩阵列表，形状为(n_components, 2, 2)
    """
    # 创建网格点
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 计算GMM密度
    density = gmm_density(grid_points, weights, means, covariances)
    density = density.reshape(xx.shape)
    
    # 绘制密度等高线
    plt.figure(figsize=(12, 5))
    
    # 1. GMM密度等高线
    plt.subplot(1, 2, 1)
    contour = plt.contourf(xx, yy, density, levels=20, cmap='viridis')
    plt.colorbar(contour)
    plt.title('GMM密度等高线')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    
    # 2. 各个混合成分的密度
    plt.subplot(1, 2, 2)
    for k in range(len(weights)):
        component_density = multivariate_normal.pdf(grid_points, mean=means[k], cov=covariances[k])
        component_density = component_density.reshape(xx.shape)
        plt.contour(xx, yy, component_density, levels=5, linestyles='dashed', alpha=0.8)
        plt.scatter(means[k][0], means[k][1], marker='o', s=100, label=f'成分 {k+1}')
    
    plt.title('各混合成分的密度（虚线）')
    plt.xlabel('特征1')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../figure/gmm_math_visualization.png')
    plt.close()

# 主函数
if __name__ == "__main__":
    # 定义GMM参数
    weights = [0.3, 0.5, 0.2]  # 混合权重
    means = [[-2, -2], [0, 0], [2, 2]]  # 均值向量
    covariances = [
        [[1, 0.5], [0.5, 1]],    # 协方差矩阵1
        [[1.5, -0.5], [-0.5, 1]], # 协方差矩阵2
        [[0.5, 0], [0, 0.5]]      # 协方差矩阵3
    ]
    
    # 打印GMM参数
    print("高斯混合模型参数:")
    print(f"混合权重: {weights}")
    print(f"均值向量: {means}")
    print(f"协方差矩阵: {covariances}")
    
    # 可视化GMM
    visualize_gmm(weights, means, covariances)
    
    # 验证与scipy的一致性
    print("\n验证与scipy.stats.multivariate_normal的一致性:")
    test_points = np.array([[-2, -2], [0, 0], [2, 2]])
    
    # 使用自定义函数计算
    custom_density = gmm_density(test_points, weights, means, covariances)
    
    # 使用scipy计算
    scipy_density = np.zeros(len(test_points))
    for k in range(len(weights)):
        scipy_density += weights[k] * multivariate_normal.pdf(test_points, mean=means[k], cov=covariances[k])
    
    print(f"自定义函数结果: {custom_density}")
    print(f"scipy结果: {scipy_density}")
    print(f"差异: {np.max(np.abs(custom_density - scipy_density))}")
    
    print("\n数学表达示例完成！")
