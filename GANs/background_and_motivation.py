#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识点1：背景介绍与动机

本文件包含GAN背景介绍与动机的相关代码实现，包括：
1. 最大似然估计的局限性
2. 无似然学习的动机
3. 分布比较的基本概念
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def maximum_likelihood_estimation():
    """
    最大似然估计的局限性示例
    
    最大似然估计（MLE）：
    θ̂ = argmax_θ Σ_{i=1}^N log p_θ(x_i), 其中 x_1,...,x_N ~ p_data(x)
    
    本函数通过示例展示MLE的局限性：
    1. 具有良好测试对数似然但生成样本质量差的情况
    2. 具有良好生成样本但测试对数似然差的情况
    """
    print("=== 最大似然估计的局限性 ===")
    
    # 生成数据分布
    np.random.seed(42)
    data_size = 1000
    # 真实数据分布：混合高斯分布
    p_data = np.concatenate([
        np.random.normal(-2, 0.5, data_size // 2),
        np.random.normal(2, 0.5, data_size // 2)
    ])
    
    # 示例1：具有良好测试对数似然但生成样本质量差的情况
    # 噪声混合模型：p_θ(x) = 0.01 p_data(x) + 0.99 p_noise(x)
    p_noise = np.random.uniform(-10, 10, data_size)
    p_theta_1 = np.concatenate([
        p_data[:int(data_size * 0.01)],
        np.random.uniform(-10, 10, int(data_size * 0.99))
    ])
    
    # 计算对数似然
    def log_likelihood(samples, distribution):
        """计算样本在给定分布下的对数似然"""
        # 使用核密度估计来近似分布
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(distribution)
        return np.mean(np.log(kde(samples) + 1e-10))  # 添加小值避免log(0)
    
    # 计算示例1的对数似然
    ll_1 = log_likelihood(p_data, p_theta_1)
    print(f"示例1 - 噪声混合模型对数似然: {ll_1:.4f}")
    
    # 示例2：具有良好生成样本但测试对数似然差的情况
    # 记忆训练集：只生成训练数据中的样本
    p_theta_2 = np.random.choice(p_data, data_size, replace=True)
    
    # 计算示例2的对数似然
    ll_2 = log_likelihood(np.random.normal(0, 1, data_size), p_theta_2)  # 测试集来自不同分布
    print(f"示例2 - 记忆训练集模型对数似然: {ll_2:.4f}")
    
    # 可视化三个分布
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(p_data, bins=50, density=True, alpha=0.7, label='真实数据分布')
    plt.title('真实数据分布')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.hist(p_theta_1, bins=50, density=True, alpha=0.7, label='噪声混合模型')
    plt.title('示例1: 噪声混合模型')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.hist(p_theta_2, bins=50, density=True, alpha=0.7, label='记忆训练集模型')
    plt.title('示例2: 记忆训练集模型')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('figure/mle_limitations.png')
    plt.close()
    
    print("MLE局限性示例已可视化并保存到figure/mle_limitations.png")


def likelihood_free_learning():
    """
    无似然学习的动机
    
    无似然学习考虑不直接依赖似然函数的替代训练目标
    
    本函数通过两样本测试展示无似然学习的动机
    """
    print("\n=== 无似然学习的动机 ===")
    print("无似然学习考虑不直接依赖似然函数的替代训练目标")
    print("")
    print("示例1说明：即使模型生成的大部分样本是噪声，只要少量样本来自真实分布，")
    print("其对数似然也可能接近真实分布的对数似然，特别是在高维情况下。")
    print("")
    print("示例2说明：如果模型只是记忆训练数据，生成的样本看起来会很好，")
    print("但对测试集的对数似然会非常差。")
    print("")
    print("这些例子表明，似然性和样本质量可能需要解耦。")


def distribution_comparison():
    """
    分布比较的基本概念
    
    给定来自两个分布的有限样本集S1 = {x ~ P}和S2 = {x ~ Q}，
    如何判断这些样本是否来自同一分布，即P = Q？
    
    本函数展示基本的分布比较方法
    """
    print("\n=== 分布比较的基本概念 ===")
    
    # 生成两个分布的样本
    np.random.seed(42)
    
    # 分布P：高斯分布 N(0, 1)
    samples_p = np.random.normal(0, 1, 1000)
    
    # 分布Q1：与P相同的分布 N(0, 1)
    samples_q1 = np.random.normal(0, 1, 1000)
    
    # 分布Q2：与P不同的分布 N(2, 1)
    samples_q2 = np.random.normal(2, 1, 1000)
    
    # 基本的分布比较方法：均值差
    def mean_difference(s1, s2):
        """计算两个样本集的均值差"""
        return np.abs(np.mean(s1) - np.mean(s2))
    
    # 计算均值差
    md_p_q1 = mean_difference(samples_p, samples_q1)
    md_p_q2 = mean_difference(samples_p, samples_q2)
    
    print(f"分布P和Q1（相同分布）的均值差: {md_p_q1:.4f}")
    print(f"分布P和Q2（不同分布）的均值差: {md_p_q2:.4f}")
    
    # 可视化分布比较
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(samples_p, bins=50, density=True, alpha=0.5, label='P ~ N(0, 1)')
    plt.hist(samples_q1, bins=50, density=True, alpha=0.5, label='Q1 ~ N(0, 1)')
    plt.title('相同分布比较')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(samples_p, bins=50, density=True, alpha=0.5, label='P ~ N(0, 1)')
    plt.hist(samples_q2, bins=50, density=True, alpha=0.5, label='Q2 ~ N(2, 1)')
    plt.title('不同分布比较')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('figure/distribution_comparison.png')
    plt.close()
    
    print("分布比较示例已可视化并保存到figure/distribution_comparison.png")


if __name__ == "__main__":
    maximum_likelihood_estimation()
    likelihood_free_learning()
    distribution_comparison()
