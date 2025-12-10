# 知识点10：GMM的EM算法实现
"""
- E-step：计算后验概率
  \[ q^{(\ell)}(z|x_i) = p_{\theta^{(\ell)}}(z|x_i) = \frac{p_{\theta^{(\ell)}}(x_i|z)p_{\theta^{(\ell)}}(z)}{p_{\theta^{(\ell)}}(x_i)} = \frac{\pi_z^{(\ell)} \mathcal{N}(x_i; \mu_z^{(\ell)}, \Sigma_z^{(\ell)})}{\sum_{k=1}^K \pi_k^{(\ell)} \mathcal{N}(x_i; \mu_k^{(\ell)}, \Sigma_k^{(\ell)})} \]
- M-step：更新参数
  \[ \theta^{(\ell+1)} = \arg \max_\theta \sum_{i=1}^N \sum_{k=1}^K q^{(\ell)}(k|x_i) \log(p_\theta(k)p_\theta(x_i|k)) \]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GMMWithEM:
    """使用EM算法实现的高斯混合模型"""
    
    def __init__(self, n_components=2, max_iter=100, tolerance=1e-4):
        """初始化GMM模型
        
        参数:
        n_components: 混合成分的数量
        max_iter: 最大迭代次数
        tolerance: 收敛阈值
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        # 模型参数
        self.weights = None  # 混合权重，shape=(n_components,)
        self.means = None  # 均值，shape=(n_components, n_features)
        self.covariances = None  # 协方差矩阵，shape=(n_components, n_features, n_features)
    
    def initialize_parameters(self, X):
        """初始化模型参数
        
        参数:
        X: 训练数据，shape=(n_samples, n_features)
        """
        n_samples, n_features = X.shape
        
        # 1. 初始化混合权重
        self.weights = np.ones(self.n_components) / self.n_components
        
        # 2. 初始化均值
        # 从数据中随机选择n_components个样本作为初始均值
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices]
        
        # 3. 初始化协方差矩阵
        # 初始化为数据的协方差矩阵乘以一个小的缩放因子
        data_cov = np.cov(X.T)
        self.covariances = np.array([data_cov * 0.5 for _ in range(self.n_components)])
        
        print(f"初始化参数：")
        print(f"   混合权重：{self.weights}")
        print(f"   均值：{self.means}")
        print(f"   协方差矩阵：")
        for i in range(self.n_components):
            print(f"     成分 {i+1}: {self.covariances[i]}")
    
    def e_step(self, X):
        """E-step：计算后验概率
        
        参数:
        X: 训练数据，shape=(n_samples, n_features)
        
        返回:
        responsibilities: 后验概率，shape=(n_samples, n_components)
        """
        n_samples, _ = X.shape
        
        # 计算每个样本对每个成分的责任
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # 计算成分k的高斯概率密度
            prob = multivariate_normal.pdf(X, mean=self.means[k], cov=self.covariances[k])
            # 乘以混合权重
            responsibilities[:, k] = self.weights[k] * prob
        
        # 归一化，使得每个样本的责任总和为1
        total_prob = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= total_prob
        
        return responsibilities
    
    def m_step(self, X, responsibilities):
        """M-step：更新模型参数
        
        参数:
        X: 训练数据，shape=(n_samples, n_features)
        responsibilities: 后验概率，shape=(n_samples, n_components)
        """
        n_samples, n_features = X.shape
        
        # 1. 更新混合权重
        # 每个成分的权重是该成分的总责任除以样本总数
        self.weights = responsibilities.sum(axis=0) / n_samples
        
        # 2. 更新均值
        # 每个成分的均值是数据的加权平均，权重是责任
        for k in range(self.n_components):
            weight_sum = responsibilities[:, k].sum()
            # 加权平均
            self.means[k] = (responsibilities[:, k][:, np.newaxis] * X).sum(axis=0) / weight_sum
        
        # 3. 更新协方差矩阵
        for k in range(self.n_components):
            weight_sum = responsibilities[:, k].sum()
            # 计算每个样本与均值的偏差
            diff = X - self.means[k]
            # 加权协方差矩阵
            cov = (responsibilities[:, k][:, np.newaxis, np.newaxis] * 
                   np.einsum('ij,ik->ijk', diff, diff)).sum(axis=0) / weight_sum
            self.covariances[k] = cov
    
    def compute_log_likelihood(self, X):
        """计算数据的对数似然
        
        参数:
        X: 训练数据，shape=(n_samples, n_features)
        
        返回:
        log_likelihood: 对数似然值
        """
        n_samples, _ = X.shape
        log_likelihood = 0.0
        
        for i in range(n_samples):
            # 计算每个样本的概率密度
            prob = 0.0
            for k in range(self.n_components):
                prob += self.weights[k] * multivariate_normal.pdf(X[i], 
                                                                   mean=self.means[k], 
                                                                   cov=self.covariances[k])
            # 累加对数似然
            log_likelihood += np.log(prob)
        
        return log_likelihood
    
    def fit(self, X):
        """训练GMM模型
        
        参数:
        X: 训练数据，shape=(n_samples, n_features)
        
        返回:
        self: 训练好的模型
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.initialize_parameters(X)
        
        # 记录对数似然历史
        log_likelihood_history = []
        
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self.e_step(X)
            
            # 计算当前对数似然
            current_log_likelihood = self.compute_log_likelihood(X)
            log_likelihood_history.append(current_log_likelihood)
            
            # M-step
            self.m_step(X, responsibilities)
            
            # 检查收敛性
            if iteration > 0:
                delta = np.abs(current_log_likelihood - log_likelihood_history[-2])
                if delta < self.tolerance:
                    print(f"EM算法在第{iteration+1}次迭代收敛")
                    break
            
            if (iteration + 1) % 10 == 0:
                print(f"迭代 {iteration+1}: 对数似然 = {current_log_likelihood:.4f}")
        
        self.log_likelihood_history = log_likelihood_history
        self.responsibilities = responsibilities
        
        print(f"\n最终参数：")
        print(f"   混合权重：{self.weights}")
        print(f"   均值：{self.means}")
        print(f"   协方差矩阵：")
        for i in range(self.n_components):
            print(f"     成分 {i+1}: {self.covariances[i]}")
        
        return self
    
    def predict(self, X):
        """预测每个样本的成分
        
        参数:
        X: 测试数据，shape=(n_samples, n_features)
        
        返回:
        labels: 每个样本的成分标签，shape=(n_samples,)
        """
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X):
        """预测每个样本属于每个成分的概率
        
        参数:
        X: 测试数据，shape=(n_samples, n_features)
        
        返回:
        probabilities: 每个样本属于每个成分的概率，shape=(n_samples, n_components)
        """
        return self.e_step(X)
    
    def plot_results(self, X):
        """绘制结果（仅适用于二维数据）"""
        if X.shape[1] != 2:
            print("只能绘制二维数据的结果")
            return
        
        plt.figure(figsize=(12, 5))
        
        # 1. 数据分布和聚类结果
        plt.subplot(1, 2, 1)
        
        # 预测每个样本的成分
        labels = self.predict(X)
        
        # 绘制数据点
        for k in range(self.n_components):
            plt.scatter(X[labels == k, 0], X[labels == k, 1], s=50, alpha=0.6, label=f'成分 {k+1}')
        
        # 绘制均值
        plt.scatter(self.means[:, 0], self.means[:, 1], s=200, c='black', marker='*', label='均值')
        
        # 绘制协方差椭圆
        for k in range(self.n_components):
            self._plot_cov_ellipse(self.means[k], self.covariances[k], color='black', alpha=0.3)
        
        plt.title('GMM聚类结果')
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.legend()
        plt.grid(True)
        
        # 2. 对数似然随迭代次数变化
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(self.log_likelihood_history) + 1), self.log_likelihood_history, 'b-')
        plt.xlabel('迭代次数')
        plt.ylabel('对数似然')
        plt.title('对数似然随迭代次数变化')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../figure/gmm_em_results.png')
        plt.close()
    
    def _plot_cov_ellipse(self, mean, cov, color='black', alpha=0.3):
        """绘制协方差矩阵对应的椭圆
        
        参数:
        mean: 均值，shape=(2,)
        cov: 协方差矩阵，shape=(2, 2)
        color: 颜色
        alpha: 透明度
        """
        # 计算协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # 确保特征值为正
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        
        # 计算椭圆的角度
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # 创建椭圆
        from matplotlib.patches import Ellipse
        
        # 椭圆的半长轴和半短轴为特征值的平方根乘以2
        width, height = 2 * np.sqrt(eigenvalues)
        
        ellipse = Ellipse(mean, width, height, angle=angle, fill=False, 
                          color=color, alpha=alpha, linewidth=2)
        
        plt.gca().add_patch(ellipse)

# 生成模拟数据
def generate_gmm_data(n_samples=300, n_components=2):
    """生成高斯混合模型数据
    
    参数:
    n_samples: 样本数量
    n_components: 混合成分的数量
    
    返回:
    X: 生成的数据，shape=(n_samples, 2)
    labels: 每个样本的真实成分，shape=(n_samples,)
    """
    # 真实参数
    true_means = np.array([[-2, -2], [2, 2]])
    true_covs = np.array([[[1, 0.5], [0.5, 1]], [[1.5, -0.5], [-0.5, 1.5]]])
    true_weights = np.array([0.4, 0.6])
    
    X = []
    labels = []
    
    for i in range(n_samples):
        # 选择成分
        component = np.random.choice(n_components, p=true_weights)
        
        # 生成样本
        sample = np.random.multivariate_normal(true_means[component], true_covs[component])
        
        X.append(sample)
        labels.append(component)
    
    return np.array(X), np.array(labels), true_means, true_covs, true_weights

# 主函数
def main():
    print("=== GMM的EM算法实现演示 ===")
    
    # 生成数据
    n_samples = 300
    n_components = 2
    X, true_labels, true_means, true_covs, true_weights = generate_gmm_data(n_samples, n_components)
    
    print(f"\n生成数据：")
    print(f"   样本数：{n_samples}")
    print(f"   混合成分数：{n_components}")
    print(f"   真实均值：{true_means}")
    print(f"   真实混合权重：{true_weights}")
    
    # 创建GMM模型
    gmm = GMMWithEM(n_components=n_components, max_iter=100, tolerance=1e-4)
    
    # 训练模型
    gmm.fit(X)
    
    # 绘制结果
    gmm.plot_results(X)
    
    print("\nGMM的EM算法实现演示完成！")
    print("\n要点总结:")
    print("- GMM的EM算法实现包括E-step和M-step两个交替执行的步骤")
    print("- E-step：计算每个样本属于每个成分的后验概率（软分配）")
    print("- M-step：基于软分配更新模型参数（混合权重、均值、协方差矩阵）")
    print("- EM算法保证对数似然单调非递减")
    print("- GMM可以处理数据的复杂分布，适用于各种聚类任务")

if __name__ == "__main__":
    main()
