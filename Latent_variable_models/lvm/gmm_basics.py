# 知识点3：高斯混合模型（GMM）的基本原理
"""
- 贝叶斯网络结构：z -> x
  - z ~ Categorical_π*(1, ..., K)
  - p(x|z = k) = N(x; μ_k*, Σ_k*)
- 生成过程：
  - 通过采样z选择混合成分k
  - 从该高斯分布采样生成数据点
- 聚类：后验概率p(z|x)标识混合成分
- 无监督学习：从无标注数据中学习（不适定问题）
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# 高斯混合模型示例
def gmm_example():
    """创建高斯混合模型示例"""
    # 生成模拟数据
    n_samples = 1000
    n_components = 3
    X, y = make_blobs(n_samples=n_samples, centers=n_components, cluster_std=0.8, random_state=42)
    
    # 初始化并训练GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    
    # 预测聚类结果
    y_pred = gmm.predict(X)
    
    # 计算后验概率
    posterior_probs = gmm.predict_proba(X)
    
    # 可视化结果
    plt.figure(figsize=(15, 5))
    
    # 1. 原始数据
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    plt.title('原始数据（真实标签）')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.grid(True)
    
    # 2. GMM聚类结果
    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.8)
    plt.title('GMM聚类结果')
    plt.xlabel('特征1')
    plt.grid(True)
    
    # 3. 后验概率可视化（使用透明度表示）
    plt.subplot(1, 3, 3)
    for i in range(n_components):
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=posterior_probs[:, i])
    plt.title('后验概率可视化（透明度）')
    plt.xlabel('特征1')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../figure/gmm_example.png')
    plt.close()
    
    # 打印GMM参数
    print("GMM参数:")
    print(f"权重（π）: {gmm.weights_}")
    print(f"均值（μ_k）: {gmm.means_}")
    print(f"协方差（Σ_k）: {gmm.covariances_}")
    
    return X, y, y_pred, posterior_probs, gmm

# 生成过程演示
def gmm_generative_process(gmm, n_samples=500):
    """演示GMM的生成过程"""
    # 从GMM生成新样本
    X_gen, z_gen = gmm.sample(n_samples=n_samples)
    
    # 可视化生成的数据
    plt.figure(figsize=(8, 6))
    plt.scatter(X_gen[:, 0], X_gen[:, 1], c=z_gen, cmap='viridis', s=50, alpha=0.8)
    plt.title('GMM生成过程示例')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.grid(True)
    plt.savefig('../figure/gmm_generative_process.png')
    plt.close()
    
    print(f"生成了 {n_samples} 个样本")
    return X_gen, z_gen

# 主函数
if __name__ == "__main__":
    print("高斯混合模型示例")
    X, y, y_pred, posterior_probs, gmm = gmm_example()
    
    print("\nGMM生成过程")
    X_gen, z_gen = gmm_generative_process(gmm)
    
    print("\n示例完成！")
