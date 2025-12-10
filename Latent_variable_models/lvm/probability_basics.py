# 知识点5：概率基本概念澄清
"""
- 概率课程中会仔细区分：
  - 随机变量，如X : Ω → ℝ^d
  - 样本，如x₁, x₂, …, x_N ∈ ℝ^d
- 实际应用中会模糊这些对象的区分
- 课程中不会讨论形式化构造Ω
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

# 随机变量和样本的示例
def random_variable_example():
    """演示随机变量和样本的概念"""
    print("=== 随机变量和样本示例 ===")
    
    # 定义随机变量（高斯分布）
    print("1. 定义随机变量 X ~ N(μ=0, σ²=1)")
    mu = 0
    sigma = 1
    
    # 生成样本
    print("2. 从随机变量 X 生成 N=1000 个样本")
    N = 1000
    samples = np.random.normal(mu, sigma, N)
    
    # 打印样本的基本统计信息
    print(f"3. 样本的基本统计信息:")
    print(f"   样本均值: {np.mean(samples):.4f}")
    print(f"   样本方差: {np.var(samples):.4f}")
    print(f"   样本标准差: {np.std(samples):.4f}")
    print(f"   样本最小值: {np.min(samples):.4f}")
    print(f"   样本最大值: {np.max(samples):.4f}")
    
    # 可视化随机变量的概率密度函数和样本分布
    plt.figure(figsize=(12, 5))
    
    # 1. 概率密度函数（PDF）
    plt.subplot(1, 2, 1)
    x = np.linspace(-4, 4, 100)
    pdf = norm.pdf(x, mu, sigma)
    plt.plot(x, pdf, 'b-', label=f'N(μ={mu}, σ²={sigma²})')
    plt.title('随机变量 X 的概率密度函数 (PDF)')
    plt.xlabel('X 的取值')
    plt.ylabel('概率密度')
    plt.legend()
    plt.grid(True)
    
    # 2. 样本直方图
    plt.subplot(1, 2, 2)
    plt.hist(samples, bins=30, density=True, alpha=0.7, color='blue', label='样本直方图')
    plt.plot(x, pdf, 'r--', label=f'N(μ={mu}, σ²={sigma²})')
    plt.title(f'{N} 个样本的分布')
    plt.xlabel('样本值')
    plt.ylabel('频率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../figure/probability_basics.png')
    plt.close()
    
    return samples

# 多个随机变量的示例
def multiple_random_variables():
    """演示多个随机变量的概念"""
    print("\n=== 多个随机变量示例 ===")
    
    # 定义两个随机变量
    print("1. 定义两个随机变量:")
    print("   X ~ Uniform(a=0, b=2)")
    print("   Y ~ N(μ=1, σ²=0.5)")
    
    # 生成样本
    N = 500
    samples_x = np.random.uniform(0, 2, N)
    samples_y = np.random.normal(1, np.sqrt(0.5), N)
    
    # 可视化两个随机变量的样本
    plt.figure(figsize=(12, 5))
    
    # 1. 第一个随机变量的样本
    plt.subplot(1, 2, 1)
    plt.hist(samples_x, bins=20, density=True, alpha=0.7, color='green')
    x = np.linspace(-0.5, 2.5, 100)
    pdf_x = uniform.pdf(x, 0, 2)
    plt.plot(x, pdf_x, 'r--', label='Uniform(0, 2)')
    plt.title('随机变量 X 的样本分布')
    plt.xlabel('X 的样本值')
    plt.ylabel('频率')
    plt.legend()
    plt.grid(True)
    
    # 2. 第二个随机变量的样本
    plt.subplot(1, 2, 2)
    plt.hist(samples_y, bins=20, density=True, alpha=0.7, color='purple')
    y = np.linspace(-1, 3, 100)
    pdf_y = norm.pdf(y, 1, np.sqrt(0.5))
    plt.plot(y, pdf_y, 'r--', label='N(1, 0.5)')
    plt.title('随机变量 Y 的样本分布')
    plt.xlabel('Y 的样本值')
    plt.ylabel('频率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../figure/multiple_random_variables.png')
    plt.close()
    
    return samples_x, samples_y

# 主函数
if __name__ == "__main__":
    # 单随机变量示例
    samples = random_variable_example()
    
    # 多个随机变量示例
    samples_x, samples_y = multiple_random_variables()
    
    print("\n概率基本概念澄清示例完成！")
    print("\n总结:")
    print("- 随机变量 X 是一个数学概念，表示从样本空间 Ω 到实数的映射")
    print("- 样本 x₁, x₂, …, x_N 是从随机变量 X 中实际抽取的具体数值")
    print("- 实际应用中，我们经常通过样本数据来推断随机变量的性质")
