# 知识点9：期望最大化（EM）算法
"""
- 新机器学习目标：\( \max_	heta \sum_{i=1}^N \max_{q(\cdot|x_i)} \int q(z|x_i) \log \left( \frac{p_	heta(x_i,z)}{q(z|x_i)} ight) dz \)
- EM交替执行两个步骤：
  - E-step：\( q^{(\ell)}(\cdot|x_i) = p_{	heta^{(\ell)}}(\cdot|x_i) \)
  - M-step：\( 	heta^{(\ell+1)} = \arg \max_	heta \sum_{i=1}^N \int q^{(\ell)}(z|x_i) \log(p_	heta(x_i,z))dz \)
- 与k-means算法的过程类似
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

class EMAlgorithmDemo:
    """期望最大化（EM）算法的基本演示"""
    
    def __init__(self, num_components=2):
        self.num_components = num_components
        
    def generate_data(self, n_samples=200):
        """生成高斯混合模型数据"""
        self.n_samples = n_samples
        
        # 真实参数
        self.true_means = np.array([2.0, 7.0])
        self.true_stds = np.array([1.0, 1.5])
        self.true_weights = np.array([0.4, 0.6])
        
        # 生成数据
        self.data = []
        self.labels = []
        
        for i in range(n_samples):
            # 选择成分
            component = np.random.choice(self.num_components, p=self.true_weights)
            # 生成样本
            sample = np.random.normal(loc=self.true_means[component], scale=self.true_stds[component])
            self.data.append(sample)
            self.labels.append(component)
        
        self.data = np.array(self.data).reshape(-1, 1)
        self.labels = np.array(self.labels)
        
        print(f"生成数据：")
        print(f"   成分数：{self.num_components}")
        print(f"   样本数：{n_samples}")
        print(f"   真实均值：{self.true_means}")
        print(f"   真实标准差：{self.true_stds}")
        print(f"   真实权重：{self.true_weights}")
    
    def initialize_parameters(self):
        """初始化模型参数"""
        # 随机初始化均值
        self.means = np.random.uniform(low=np.min(self.data), high=np.max(self.data), size=self.num_components)
        
        # 初始化标准差
        self.stds = np.random.uniform(low=0.5, high=2.0, size=self.num_components)
        
        # 初始化权重
        self.weights = np.ones(self.num_components) / self.num_components
        
        print(f"\n初始参数：")
        print(f"   均值：{self.means}")
        print(f"   标准差：{self.stds}")
        print(f"   权重：{self.weights}")
    
    def e_step(self):
        """E-step：计算后验概率"""
        # 计算每个样本属于每个成分的概率
        n_samples, _ = self.data.shape
        responsibilities = np.zeros((n_samples, self.num_components))
        
        for i in range(self.num_components):
            responsibilities[:, i] = self.weights[i] * norm.pdf(self.data[:, 0], loc=self.means[i], scale=self.stds[i])
        
        # 归一化
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        
        return responsibilities
    
    def m_step(self, responsibilities):
        """M-step：更新模型参数"""
        n_samples, _ = self.data.shape
        
        # 更新权重
        self.weights = np.sum(responsibilities, axis=0) / n_samples
        
        # 更新均值
        self.means = np.sum(responsibilities * self.data, axis=0) / np.sum(responsibilities, axis=0)
        
        # 更新标准差
        for i in range(self.num_components):
            diff = self.data[:, 0] - self.means[i]
            weighted_var = np.sum(responsibilities[:, i] * diff**2)
            self.stds[i] = np.sqrt(weighted_var / np.sum(responsibilities[:, i]))
    
    def compute_log_likelihood(self):
        """计算对数似然"""
        n_samples, _ = self.data.shape
        log_likelihood = 0.0
        
        for i in range(n_samples):
            prob = 0.0
            for j in range(self.num_components):
                prob += self.weights[j] * norm.pdf(self.data[i, 0], loc=self.means[j], scale=self.stds[j])
            log_likelihood += np.log(prob)
        
        return log_likelihood
    
    def run_em(self, max_iterations=100, tolerance=1e-4):
        """运行EM算法"""
        log_likelihood_history = []
        
        for iteration in range(max_iterations):
            # E-step
            responsibilities = self.e_step()
            
            # 计算当前对数似然
            current_log_likelihood = self.compute_log_likelihood()
            log_likelihood_history.append(current_log_likelihood)
            
            # M-step
            self.m_step(responsibilities)
            
            # 检查收敛性
            if iteration > 0:
                delta = np.abs(current_log_likelihood - log_likelihood_history[-2])
                if delta < tolerance:
                    print(f"\nEM算法在第{iteration+1}次迭代收敛")
                    break
            
            if (iteration + 1) % 10 == 0:
                print(f"迭代 {iteration+1}: 对数似然 = {current_log_likelihood:.4f}")
        
        self.log_likelihood_history = log_likelihood_history
        self.responsibilities = responsibilities
        
        print(f"\n最终参数：")
        print(f"   均值：{self.means}")
        print(f"   标准差：{self.stds}")
        print(f"   权重：{self.weights}")
        print(f"   真实均值：{self.true_means}")
        print(f"   真实标准差：{self.true_stds}")
        print(f"   真实权重：{self.true_weights}")
    
    def plot_results(self):
        """绘制结果"""
        plt.figure(figsize=(12, 5))
        
        # 1. 数据分布和拟合曲线
        plt.subplot(1, 2, 1)
        
        # 绘制数据直方图
        plt.hist(self.data[:, 0], bins=30, density=True, alpha=0.6, color='gray', label='数据分布')
        
        # 绘制拟合曲线
        x = np.linspace(np.min(self.data) - 2, np.max(self.data) + 2, 1000)
        
        for i in range(self.num_components):
            y = self.weights[i] * norm.pdf(x, loc=self.means[i], scale=self.stds[i])
            plt.plot(x, y, label=f'成分 {i+1} (μ={self.means[i]:.2f}, σ={self.stds[i]:.2f})')
        
        # 绘制混合曲线
        mixture_y = np.zeros_like(x)
        for i in range(self.num_components):
            mixture_y += self.weights[i] * norm.pdf(x, loc=self.means[i], scale=self.stds[i])
        plt.plot(x, mixture_y, 'k-', linewidth=2, label='混合分布')
        
        plt.xlabel('x')
        plt.ylabel('概率密度')
        plt.title('数据分布和EM拟合结果')
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
        plt.savefig('../figure/em_algorithm_demo.png')
        plt.close()

# 与k-means算法的比较
def compare_with_kmeans():
    """比较EM算法与k-means算法的过程"""
    print("\n=== EM算法与k-means算法的比较 ===")
    
    print("1. 相似之处：")
    print("   - 都是迭代优化算法")
    print("   - 都交替执行分配和更新步骤")
    print("   - 都可能陷入局部最优")
    print("   - 都需要初始化参数")
    
    print("\n2. 不同之处：")
    print("   - k-means算法处理硬分配（每个点属于一个簇），EM算法处理软分配（每个点有属于每个簇的概率）")
    print("   - k-means算法使用欧氏距离，EM算法使用概率模型")
    print("   - k-means算法通常用于聚类，EM算法更通用，可以用于各种概率模型")
    print("   - EM算法可以处理缺失数据，k-means算法通常不能")
    
    print("\n3. 过程类比：")
    print("   - k-means的分配步骤 ≈ EM算法的E-step")
    print("   - k-means的更新步骤 ≈ EM算法的M-step")
    print("   - 两者都试图最大化某种目标函数")

# 主函数
def main():
    print("=== 期望最大化（EM）算法演示 ===")
    
    # 创建EM算法演示对象
    em_demo = EMAlgorithmDemo(num_components=2)
    
    # 生成数据
    em_demo.generate_data(n_samples=300)
    
    # 初始化参数
    em_demo.initialize_parameters()
    
    # 运行EM算法
    em_demo.run_em(max_iterations=100, tolerance=1e-4)
    
    # 绘制结果
    em_demo.plot_results()
    
    # 与k-means算法比较
    compare_with_kmeans()
    
    print("\nEM算法演示完成！")
    print("\n要点总结:")
    print("- EM算法是一种交替优化算法，用于估计含有隐变量的概率模型的参数")
    print("- E-step：计算隐变量的后验概率（软分配）")
    print("- M-step：基于软分配更新模型参数")
    print("- EM算法保证对数似然单调非递减")
    print("- 与k-means算法类似，但处理软分配而不是硬分配")

if __name__ == "__main__":
    main()
