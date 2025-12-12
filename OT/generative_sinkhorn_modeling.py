#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最优传输（Optimal Transport）知识分类 - 知识点6：生成Sinkhorn建模

本文件对应知识点6的内容，主要介绍生成Sinkhorn建模。
"""

import numpy as np
import matplotlib.pyplot as plt

def generative_sinkhorn_modeling():
    """
    生成Sinkhorn建模
    - 目标：近似最小化 Wasserstein-1 距离 W_1(p_data, p_θ)
    - 采样过程：从数据分布和模型分布中抽取样本
    - 经验分布：
      p̂_data(x) = (1/B) Σ_{i=1}^B I_{x_i = x}
      p̂_θ(y) = (1/B) Σ_{i=1}^B I_{y_i = y}
    - 使用Sinkhorn-Knopp算法估计 W_1(p̂_data, p̂_θ)
    - 梯度更新：θ ← θ - η ∇_θ W_1(p̂_data, p̂_θ)
    """
    print("="*60)
    print("知识点6：生成Sinkhorn建模")
    print("="*60)
    print("\n1. Generative Sinkhorn modeling")
    print("- 目标：近似最小化 Wasserstein-1 距离 W_1(p_data, p_θ)")
    print("- 采样过程：")
    print("  - 从数据分布中抽取样本：x₁, ..., x_B ~ p_data")
    print("  - 从模型分布中抽取样本：y₁, ..., y_B ~ p_θ")
    print("- 经验分布：")
    print("  p̂_data(x) = (1/B) Σ_{i=1}^B I_{x_i = x}")
    print("  p̂_θ(y) = (1/B) Σ_{i=1}^B I_{y_i = y}")
    print("- 使用Sinkhorn-Knopp算法估计 W_1(p̂_data, p̂_θ)")
    print("- 梯度更新：θ ← θ - η ∇_θ W_1(p̂_data, p̂_θ)")
    print()

def wasserstein_distance_estimation():
    """
    Wasserstein距离的估计
    - 使用经验分布和Sinkhorn-Knopp算法估计Wasserstein-1距离
    """
    print("2. Wasserstein距离的估计")
    
    # 生成数据分布和模型分布的样本
    np.random.seed(42)
    B = 10  # 批次大小
    
    # 数据分布：高斯分布 N(0, 1)
    x_samples = np.random.normal(0, 1, B)
    
    # 模型分布：高斯分布 N(μ, 1)，其中μ是模型参数
    μ = 2.0  # 初始模型参数
    y_samples = np.random.normal(μ, 1, B)
    
    print(f"批次大小 B = {B}")
    print(f"数据分布样本 x ~ N(0, 1): {x_samples}")
    print(f"模型分布样本 y ~ N({μ}, 1): {y_samples}")
    
    # 计算经验分布
    def empirical_distribution(samples):
        # 创建经验分布的概率质量函数
        unique, counts = np.unique(samples, return_counts=True)
        prob = counts / len(samples)
        return unique, prob
    
    x_unique, p_data = empirical_distribution(x_samples)
    y_unique, p_model = empirical_distribution(y_samples)
    
    print(f"\n数据分布的经验分布：")
    print(f"  支持点: {x_unique}")
    print(f"  概率: {p_data}")
    print(f"模型分布的经验分布：")
    print(f"  支持点: {y_unique}")
    print(f"  概率: {p_model}")
    
    # 计算成本矩阵（Wasserstein-1使用绝对值距离）
    C = np.abs(x_unique[:, np.newaxis] - y_unique)
    
    print(f"\n成本矩阵 C (绝对值距离):")
    print(C)
    
    # 正则化参数
    λ = 0.5
    
    # 使用Sinkhorn-Knopp算法估计Wasserstein距离
    def sinkhorn_algorithm(p, q, C, ε, max_iter=1000, tol=1e-6):
        m, n = C.shape
        
        # 初始化u和v
        u = np.ones(m)
        v = np.ones(n)
        
        # 计算K = exp(-C / ε)
        K = np.exp(-C / ε)
        
        for i in range(max_iter):
            # 更新u
            u_new = p / (K @ v)
            
            # 更新v
            v_new = q / (K.T @ u_new)
            
            # 检查收敛性
            if np.max(np.abs(u_new - u)) < tol and np.max(np.abs(v_new - v)) < tol:
                u, v = u_new, v_new
                break
            
            u, v = u_new, v_new
        
        # 计算传输计划
        gamma = u[:, np.newaxis] * K * v
        
        # 计算正则化的Wasserstein距离
        wasserstein_reg = np.sum(gamma * C)
        
        return gamma, wasserstein_reg, u, v
    
    gamma, wasserstein_reg, u, v = sinkhorn_algorithm(p_data, p_model, C, λ)
    
    print(f"\n正则化参数 λ = {λ}")
    print(f"估计的传输计划 γ:")
    print(gamma)
    print(f"估计的正则化Wasserstein-1距离: {wasserstein_reg:.4f}")
    
    return x_samples, y_samples, x_unique, p_data, y_unique, p_model, C, gamma, wasserstein_reg, λ, μ

def gradient_update():
    """
    梯度更新示例
    - 展示如何使用估计的Wasserstein距离进行梯度更新
    """
    print("\n3. 梯度更新示例")
    
    # 创建一个简单的生成模型，参数为μ
    def generate_samples(μ, B):
        return np.random.normal(μ, 1, B)
    
    # 目标数据分布：N(0, 1)
    def generate_data(B):
        return np.random.normal(0, 1, B)
    
    # 计算经验分布和成本矩阵
    def compute_cost_matrix(data_samples, model_samples):
        # 使用经验分布的支持点
        x_unique = np.unique(data_samples)
        y_unique = np.unique(model_samples)
        
        # 计算概率质量
        _, p_data = np.unique(data_samples, return_counts=True)
        _, p_model = np.unique(model_samples, return_counts=True)
        p_data = p_data / len(data_samples)
        p_model = p_model / len(model_samples)
        
        # 计算成本矩阵
        C = np.abs(x_unique[:, np.newaxis] - y_unique)
        
        return x_unique, p_data, y_unique, p_model, C
    
    # Sinkhorn-Knopp算法
    def sinkhorn_algorithm(p, q, C, ε, max_iter=1000, tol=1e-6):
        m, n = C.shape
        u = np.ones(m)
        v = np.ones(n)
        K = np.exp(-C / ε)
        
        for i in range(max_iter):
            u_new = p / (K @ v)
            v_new = q / (K.T @ u_new)
            
            if np.max(np.abs(u_new - u)) < tol and np.max(np.abs(v_new - v)) < tol:
                u, v = u_new, v_new
                break
            
            u, v = u_new, v_new
        
        gamma = u[:, np.newaxis] * K * v
        wasserstein_reg = np.sum(gamma * C)
        
        return gamma, wasserstein_reg
    
    # 初始化模型参数
    μ = 2.0  # 初始均值
    B = 50  # 批次大小
    λ = 0.5  # 正则化参数
    η = 0.1  # 学习率
    iterations = 5  # 迭代次数
    
    print(f"初始模型参数 μ = {μ}")
    print(f"批次大小 B = {B}")
    print(f"正则化参数 λ = {λ}")
    print(f"学习率 η = {η}")
    
    # 训练循环
    for iter in range(iterations):
        # 生成样本
        data_samples = generate_data(B)
        model_samples = generate_samples(μ, B)
        
        # 计算成本矩阵
        x_unique, p_data, y_unique, p_model, C = compute_cost_matrix(data_samples, model_samples)
        
        # 估计Wasserstein距离
        gamma, wasserstein_reg = sinkhorn_algorithm(p_data, p_model, C, λ)
        
        # 近似梯度（这里使用有限差分法近似）
        delta = 0.01
        
        # 计算μ+delta的Wasserstein距离
        model_samples_plus = generate_samples(μ + delta, B)
        x_unique_plus, p_data_plus, y_unique_plus, p_model_plus, C_plus = compute_cost_matrix(data_samples, model_samples_plus)
        gamma_plus, wasserstein_reg_plus = sinkhorn_algorithm(p_data_plus, p_model_plus, C_plus, λ)
        
        # 计算梯度
        grad_μ = (wasserstein_reg_plus - wasserstein_reg) / delta
        
        # 更新参数
        μ -= η * grad_μ
        
        print(f"\n迭代 {iter+1}:")
        print(f"  正则化Wasserstein距离: {wasserstein_reg:.4f}")
        print(f"  梯度 ∇_μ W_1: {grad_μ:.4f}")
        print(f"  更新后的参数 μ: {μ:.4f}")
    
    print(f"\n最终模型参数 μ = {μ:.4f}")
    
    return μ

def generative_model_example():
    """
    生成模型示例
    - 完整的生成Sinkhorn建模示例
    """
    print("\n4. 生成模型示例")
    
    # 定义生成模型
    class GenerativeModel:
        def __init__(self, μ):
            self.μ = μ
        
        def sample(self, B):
            return np.random.normal(self.μ, 1, B)
        
        def update(self, grad_μ, η):
            self.μ -= η * grad_μ
    
    # 初始化模型
    model = GenerativeModel(μ=2.0)
    B = 100  # 批次大小
    λ = 0.5  # 正则化参数
    η = 0.05  # 学习率
    iterations = 10  # 迭代次数
    
    print(f"初始模型均值 μ = {model.μ}")
    print(f"批次大小 B = {B}")
    print(f"正则化参数 λ = {λ}")
    print(f"学习率 η = {η}")
    
    # 存储训练过程中的参数和Wasserstein距离
    params_history = [model.μ]
    wasserstein_history = []
    
    for iter in range(iterations):
        # 生成数据样本
        data_samples = np.random.normal(0, 1, B)
        
        # 生成模型样本
        model_samples = model.sample(B)
        
        # 计算经验分布
        x_unique, counts_x = np.unique(data_samples, return_counts=True)
        y_unique, counts_y = np.unique(model_samples, return_counts=True)
        p_data = counts_x / B
        p_model = counts_y / B
        
        # 计算成本矩阵
        C = np.abs(x_unique[:, np.newaxis] - y_unique)
        
        # 使用Sinkhorn-Knopp算法计算正则化的Wasserstein距离
        def sinkhorn(p, q, C, ε):
            m, n = C.shape
            u = np.ones(m)
            v = np.ones(n)
            K = np.exp(-C / ε)
            
            for i in range(1000):
                u_new = p / (K @ v)
                v_new = q / (K.T @ u_new)
                
                if np.max(np.abs(u_new - u)) < 1e-6 and np.max(np.abs(v_new - v)) < 1e-6:
                    u, v = u_new, v_new
                    break
                
                u, v = u_new, v_new
            
            gamma = u[:, np.newaxis] * K * v
            return np.sum(gamma * C)
        
        wasserstein_reg = sinkhorn(p_data, p_model, C, λ)
        
        # 近似梯度
        delta = 0.01
        
        # 计算μ+delta的Wasserstein距离
        model_samples_plus = np.random.normal(model.μ + delta, 1, B)
        x_unique_plus, counts_x_plus = np.unique(data_samples, return_counts=True)
        y_unique_plus, counts_y_plus = np.unique(model_samples_plus, return_counts=True)
        p_data_plus = counts_x_plus / B
        p_model_plus = counts_y_plus / B
        C_plus = np.abs(x_unique_plus[:, np.newaxis] - y_unique_plus)
        wasserstein_reg_plus = sinkhorn(p_data_plus, p_model_plus, C_plus, λ)
        
        # 计算梯度
        grad_μ = (wasserstein_reg_plus - wasserstein_reg) / delta
        
        # 更新模型参数
        model.update(grad_μ, η)
        
        # 存储历史记录
        params_history.append(model.μ)
        wasserstein_history.append(wasserstein_reg)
        
        print(f"\n迭代 {iter+1}:")
        print(f"  正则化Wasserstein距离: {wasserstein_reg:.4f}")
        print(f"  梯度 ∇_μ W_1: {grad_μ:.4f}")
        print(f"  更新后的参数 μ: {model.μ:.4f}")
    
    print(f"\n最终模型均值 μ = {model.μ:.4f}")
    
    # 可视化训练过程
    plt.figure(figsize=(10, 5))
    
    # 绘制参数变化
    plt.subplot(1, 2, 1)
    plt.plot(params_history)
    plt.xlabel('迭代次数')
    plt.ylabel('模型参数 μ')
    plt.title('模型参数变化')
    plt.grid(True)
    
    # 绘制Wasserstein距离变化
    plt.subplot(1, 2, 2)
    plt.plot(wasserstein_history)
    plt.xlabel('迭代次数')
    plt.ylabel('正则化Wasserstein距离')
    plt.title('Wasserstein距离变化')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./figure/generative_sinkhorn_training.png')
    print("\n已生成训练过程可视化图：./figure/generative_sinkhorn_training.png")
    plt.close()
    
    return params_history, wasserstein_history

def visualize_generative_modeling():
    """
    可视化生成Sinkhorn建模
    """
    print("\n5. 可视化生成Sinkhorn建模")
    
    # 生成数据样本和模型样本
    np.random.seed(42)
    B = 100
    
    # 数据分布：N(0, 1)
    data_samples = np.random.normal(0, 1, B)
    
    # 模型分布：N(2, 1)（初始化远离数据分布）
    model_samples_initial = np.random.normal(2, 1, B)
    
    # 训练后的模型分布：N(0.5, 1)（接近数据分布）
    model_samples_trained = np.random.normal(0.5, 1, B)
    
    # 绘制分布图
    plt.figure(figsize=(12, 6))
    
    plt.hist(data_samples, bins=20, alpha=0.5, label='数据分布 p_data')
    plt.hist(model_samples_initial, bins=20, alpha=0.5, label='初始模型分布 p_θ(μ=2)')
    plt.hist(model_samples_trained, bins=20, alpha=0.5, label='训练后模型分布 p_θ(μ=0.5)')
    
    plt.xlabel('值')
    plt.ylabel('频率')
    plt.title('生成Sinkhorn建模 - 模型分布的优化')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figure/generative_modeling_distributions.png')
    print("\n已生成生成建模分布可视化图：./figure/generative_modeling_distributions.png")
    plt.close()

if __name__ == "__main__":
    generative_sinkhorn_modeling()
    wasserstein_distance_estimation()
    gradient_update()
    generative_model_example()
    visualize_generative_modeling()
