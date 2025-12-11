#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识点2：两样本测试与判别器

本文件包含两样本测试与判别器的相关代码实现，包括：
1. 两样本测试的基本原理
2. 生成建模与两样本测试
3. 通过判别器进行两样本测试
4. 判别器的训练目标与最优解
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def two_sample_testing_basics():
    """
    两样本测试的基本原理
    
    给定S1 = {x ~ P}和S2 = {x ~ Q}，两样本测试考虑以下假设：
    - 零假设H0: P = Q
    - 备择假设H1: P ≠ Q
    
    测试统计量T比较S1和S2，例如均值差：
    T(S1, S2) = ||(1/|S1|)Σ_{x∈S1} x - (1/|S2|)Σ_{x∈S2} x||_2
    
    如果T大于阈值α，则拒绝H0，否则接受H0
    """
    print("=== 两样本测试的基本原理 ===")
    
    np.random.seed(42)
    sample_size = 500
    
    # 情况1：P = Q（零假设为真）
    p_samples = np.random.normal(0, 1, sample_size)
    q_samples_same = np.random.normal(0, 1, sample_size)
    
    # 情况2：P ≠ Q（备择假设为真）
    q_samples_diff = np.random.normal(2, 1, sample_size)
    
    # 计算均值差测试统计量
    def mean_diff_test(s1, s2):
        """计算均值差测试统计量"""
        mean_s1 = np.mean(s1)
        mean_s2 = np.mean(s2)
        return np.abs(mean_s1 - mean_s2)
    
    # 计算测试统计量
    t_same = mean_diff_test(p_samples, q_samples_same)
    t_diff = mean_diff_test(p_samples, q_samples_diff)
    
    print(f"情况1（P=Q）的均值差测试统计量: {t_same:.4f}")
    print(f"情况2（P≠Q）的均值差测试统计量: {t_diff:.4f}")
    
    # 可视化两个情况
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(p_samples, bins=50, density=True, alpha=0.5, label='P样本')
    plt.hist(q_samples_same, bins=50, density=True, alpha=0.5, label='Q样本（相同分布）')
    plt.title(f'P=Q时的样本分布\n均值差 = {t_same:.4f}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(p_samples, bins=50, density=True, alpha=0.5, label='P样本')
    plt.hist(q_samples_diff, bins=50, density=True, alpha=0.5, label='Q样本（不同分布）')
    plt.title(f'P≠Q时的样本分布\n均值差 = {t_diff:.4f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('figure/two_sample_testing_basics.png')
    plt.close()
    
    print("两样本测试基本原理已可视化并保存到figure/two_sample_testing_basics.png")


def generative_modeling_and_two_sample_testing():
    """
    生成建模与两样本测试
    
    假设我们可以直接访问S1 = D = {x ~ p_data}
    此外，我们有一个模型分布p_θ
    假设模型分布允许高效采样，令S2 = {x ~ p_θ}
    
    分布之间的距离的替代概念：训练生成模型以最小化S1和S2之间的两样本测试目标
    """
    print("\n=== 生成建模与两样本测试 ===")
    
    np.random.seed(42)
    
    # 真实数据分布p_data
    p_data = np.concatenate([
        np.random.normal(-2, 0.5, 500),
        np.random.normal(2, 0.5, 500)
    ])
    
    # 不同质量的生成模型
    
    # 模型1：质量较差的生成模型
    generator1 = lambda size: np.random.uniform(-5, 5, size)
    p_theta1 = generator1(1000)
    
    # 模型2：质量较好的生成模型
    generator2 = lambda size: np.concatenate([
        np.random.normal(-2, 0.8, size // 2),
        np.random.normal(2, 0.8, size // 2)
    ])
    p_theta2 = generator2(1000)
    
    # 使用两样本测试评估生成模型质量
    def evaluate_generator(real_samples, generated_samples):
        """使用两样本测试评估生成模型质量"""
        # 创建标签：real=1, generated=0
        X = np.concatenate([real_samples, generated_samples]).reshape(-1, 1)
        y = np.concatenate([np.ones(len(real_samples)), np.zeros(len(generated_samples))])
        
        # 训练逻辑回归模型作为两样本测试分类器
        clf = LogisticRegression(random_state=42)
        clf.fit(X, y)
        
        # 计算分类准确率作为测试统计量
        accuracy = accuracy_score(y, clf.predict(X))
        
        return accuracy
    
    # 评估两个生成模型
    acc1 = evaluate_generator(p_data, p_theta1)
    acc2 = evaluate_generator(p_data, p_theta2)
    
    print(f"模型1（质量较差）的两样本测试准确率: {acc1:.4f}")
    print(f"模型2（质量较好）的两样本测试准确率: {acc2:.4f}")
    print("\n说明：准确率越高，说明生成样本与真实样本越容易区分，模型质量越差")
    print("准确率接近0.5，说明生成样本与真实样本难以区分，模型质量越好")


def discriminator_two_sample_test():
    """
    通过判别器进行两样本测试
    
    寻找高维数据中好的两样本测试目标很困难
    在生成模型设置中，我们知道S1和S2来自不同的分布p_data和p_theta
    
    解决方法：训练一个分类器（称为判别器）来自动识别两组样本S1和S2之间的差异
    """
    print("\n=== 通过判别器进行两样本测试 ===")
    
    np.random.seed(42)
    
    # 生成高维数据
    dim = 10
    sample_size = 1000
    
    # 真实数据分布：多变量高斯分布
    p_data = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), sample_size)
    
    # 生成模型分布：不同均值的多变量高斯分布
    p_theta = np.random.multivariate_normal(np.ones(dim)*0.5, np.eye(dim), sample_size)
    
    # 创建判别器（简单的逻辑回归）
    def train_discriminator(real_samples, generated_samples):
        """训练判别器区分真实样本和生成样本"""
        # 创建标签：real=1, generated=0
        X = np.concatenate([real_samples, generated_samples])
        y = np.concatenate([np.ones(len(real_samples)), np.zeros(len(generated_samples))])
        
        # 训练判别器
        discriminator = LogisticRegression(random_state=42)
        discriminator.fit(X, y)
        
        return discriminator
    
    # 训练判别器
    discriminator = train_discriminator(p_data, p_theta)
    
    # 计算判别器性能
    X_test = np.concatenate([p_data, p_theta])
    y_test = np.concatenate([np.ones(len(p_data)), np.zeros(len(p_theta))])
    accuracy = accuracy_score(y_test, discriminator.predict(X_test))
    
    print(f"高维数据中判别器的两样本测试准确率: {accuracy:.4f}")
    print("说明：即使在高维空间中，判别器也能有效地区分两个分布")


def discriminator_training_objective():
    """
    判别器的训练目标与最优解
    
    判别器的训练目标：
    E_{x~p_data}[log(D_φ(x))] + E_{x~p_θ}[log(1 - D_φ(x))]
    
    最优判别器（输出[0,1]之间的连续概率）：
    D^*_θ(x) = p_data(x) / (p_data(x) + p_θ(x))
    """
    print("\n=== 判别器的训练目标与最优解 ===")
    
    np.random.seed(42)
    
    # 创建一维分布示例
    x = np.linspace(-5, 5, 1000)
    
    # 真实数据分布 p_data
    p_data = norm.pdf(x, 0, 1)
    
    # 生成模型分布 p_theta
    p_theta = norm.pdf(x, 1, 1)
    
    # 计算最优判别器
    def optimal_discriminator(p_data, p_theta):
        """计算最优判别器"""
        return p_data / (p_data + p_theta + 1e-10)  # 添加小值避免除零
    
    d_star = optimal_discriminator(p_data, p_theta)
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    
    plt.plot(x, p_data, label='p_data(x)')
    plt.plot(x, p_theta, label='p_theta(x)')
    plt.plot(x, d_star, label='D^*_θ(x) - 最优判别器')
    plt.axhline(y=0.5, color='r', linestyle='--', label='随机猜测')
    
    plt.title('最优判别器示例')
    plt.xlabel('x')
    plt.ylabel('概率/密度')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('figure/optimal_discriminator.png')
    plt.close()
    
    print("最优判别器已可视化并保存到figure/optimal_discriminator.png")
    print("\n最优判别器性质：")
    print("1. 如果p_theta = p_data，则D^*_θ(x) = 1/2（无法区分，随机猜测）")
    print("2. 对于p_data(x) > p_theta(x)的区域，D^*_θ(x) > 1/2")
    print("3. 对于p_data(x) < p_theta(x)的区域，D^*_θ(x) < 1/2")


if __name__ == "__main__":
    two_sample_testing_basics()
    generative_modeling_and_two_sample_testing()
    discriminator_two_sample_test()
    discriminator_training_objective()
