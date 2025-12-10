# 知识点6：高斯混合模型的学习
"""
- GMM的底层参数：θ* = {π*, {μ_k*}_{k=1}^K, {Σ_k*}_{k=1}^K}
- 最大似然估计（MLE）：
  θ̂ = arg max_θ (1/N) Σ_{i=1}^N log p_θ(x_i) = arg max_θ (1/N) Σ_{i=1}^N log (Σ_{k=1}^K π_k N(x; μ_k, Σ_k))
- 无解析解，是非凸优化问题
- 期望最大化（EM）算法是学习GMM的经典方法
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# GMM学习示例
def gmm_learning_example():
    """演示GMM的学习过程"""
    print("=== GMM学习示例 ===")
    
    # 生成模拟数据
    n_samples = 1000
    n_components = 3
    X, y = make_blobs(n_samples=n_samples, centers=n_components, cluster_std=1.0, random_state=42)
    
    # 打印数据基本信息
    print(f"1. 生成 {n_samples} 个样本，包含 {n_components} 个真实聚类")
    print(f"   数据形状: {X.shape}")
    
    # 使用不同数量的混合成分训练GMM
    n_components_list = [2, 3, 4, 5]
    silhouette_scores = []
    models = []
    
    print("\n2. 使用不同数量的混合成分训练GMM:")
    for n_comp in n_components_list:
        # 训练GMM
        gmm = GaussianMixture(n_components=n_comp, random_state=42)
        gmm.fit(X)
        models.append(gmm)
        
        # 预测聚类
        y_pred = gmm.predict(X)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(X, y_pred)
        silhouette_scores.append(silhouette_avg)
        
        # 计算对数似然
        log_likelihood = gmm.score(X)
        
        print(f"   - {n_comp} 个混合成分:")
        print(f"     轮廓系数: {silhouette_avg:.4f}")
        print(f"     对数似然: {log_likelihood:.4f}")
    
    # 选择最优的模型（轮廓系数最高）
    best_idx = np.argmax(silhouette_scores)
    best_n_components = n_components_list[best_idx]
    best_model = models[best_idx]
    
    print(f"\n3. 最优模型: {best_n_components} 个混合成分")
    
    # 可视化不同数量混合成分的效果
    plt.figure(figsize=(15, 10))
    
    for i, (n_comp, model) in enumerate(zip(n_components_list, models)):
        # 预测聚类
        y_pred = model.predict(X)
        
        # 绘制结果
        plt.subplot(2, 2, i+1)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, alpha=0.8)
        plt.scatter(model.means_[:, 0], model.means_[:, 1], marker='x', c='red', s=200, linewidths=3)
        plt.title(f'{n_comp} 个混合成分 (轮廓系数: {silhouette_scores[i]:.4f})')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../figure/gmm_learning_comparison.png')
    plt.close()
    
    # 可视化对数似然和轮廓系数
    plt.figure(figsize=(12, 5))
    
    # 1. 对数似然
    plt.subplot(1, 2, 1)
    log_likelihoods = [model.score(X) for model in models]
    plt.plot(n_components_list, log_likelihoods, 'o-', linewidth=2)
    plt.title('对数似然 vs 混合成分数量')
    plt.xlabel('混合成分数量')
    plt.ylabel('对数似然')
    plt.grid(True)
    
    # 2. 轮廓系数
    plt.subplot(1, 2, 2)
    plt.plot(n_components_list, silhouette_scores, 'o-', linewidth=2, color='green')
    plt.title('轮廓系数 vs 混合成分数量')
    plt.xlabel('混合成分数量')
    plt.ylabel('轮廓系数')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../figure/gmm_model_selection.png')
    plt.close()
    
    return X, y, best_model, best_n_components

# 演示GMM参数
def gmm_parameters_demo(gmm, X):
    """演示GMM的学习参数"""
    print("\n=== GMM学习参数演示 ===")
    
    # 获取学习到的参数
    weights = gmm.weights_
    means = gmm.means_
    covariances = gmm.covariances_
    
    print(f"1. 学习到的参数:")
    print(f"   - 混合权重 (π): {weights}")
    print(f"   - 均值 (μ_k): {means}")
    print(f"   - 协方差 (Σ_k): 形状为 {covariances.shape}")
    
    # 可视化学习到的高斯分布
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c='gray', s=50, alpha=0.5, label='数据点')
    
    # 绘制每个高斯分布
    from scipy.stats import multivariate_normal
    x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    for i in range(gmm.n_components):
        # 计算当前高斯分布的密度
        z = multivariate_normal.pdf(grid_points, mean=means[i], cov=covariances[i])
        z = z.reshape(xx.shape)
        
        # 绘制等高线
        plt.contour(xx, yy, z, levels=5, linestyles='dashed', alpha=0.8, label=f'成分 {i+1}')
        
        # 绘制均值
        plt.scatter(means[i, 0], means[i, 1], marker='x', c='red', s=200, linewidths=3)
    
    plt.title('学习到的高斯混合模型')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.grid(True)
    plt.savefig('../figure/gmm_learned_parameters.png')
    plt.close()

# 主函数
if __name__ == "__main__":
    X, y, best_model, best_n_components = gmm_learning_example()
    gmm_parameters_demo(best_model, X)
    
    print("\nGMM学习示例完成！")
    print("要点总结:")
    print("- GMM的参数包括混合权重、均值向量和协方差矩阵")
    print("- MLE是学习GMM参数的常用方法，但无解析解")
    print("- EM算法是学习GMM的经典方法")
    print("- 可以使用交叉验证、轮廓系数等方法选择最优的混合成分数量")
