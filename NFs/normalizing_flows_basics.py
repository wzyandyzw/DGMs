#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalizing Flows - 知识点3：Normalizing Flows 基础

本文件介绍了Normalizing Flows的基础知识，包括：
1. Normalizing Flows 定义
2. 变换流
3. 学习与推理
4. Flow 模型的设计要点
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


def normalizing_flows_definition():
    """
    Normalizing Flows 定义
    介绍Normalizing Flows的基本定义
    """
    print("=== Normalizing Flows 定义 ===")
    
    print("Normalizing Flows是一种生成模型，通过可逆变换将简单分布映射到复杂分布。")
    print("\n数学定义：")
    print("- 考虑观察变量X和潜在变量Z上的有向潜在变量模型")
    print("- 在NF中，Z和X之间的映射由确定性可逆函数f_theta: R^n → R^n给出")
    print("- 使得X = f_theta(Z)和Z = f_theta^{-1}(X)")
    print("- 使用变量变换，边际似然p_theta(x)由下式给出：")
    print("  p_theta(x) = p_Z(f_theta^{-1}(x)) * |det(∂f_theta^{-1}(x)/∂x)|")
    
    print("\n重要注意事项：")
    print("- x和z需要是连续的且维度相同")
    print("- 变换必须是可逆的，以确保可以进行精确的似然评估和采样")
    
    # 可视化简单的NF示例
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 简单高斯分布作为先验
    z = np.random.normal(0, 1, 10000)
    ax1.hist(z, bins=50, density=True, alpha=0.7, color='blue')
    ax1.set_title('先验分布：Z ~ N(0, 1)')
    ax1.set_xlabel('z')
    ax1.set_ylabel('概率密度')
    
    # 简单的可逆变换
    def f(z):
        """简单的可逆变换函数"""
        return z + 0.5 * np.sin(2 * z)  # 正弦变换
    
    def f_inv(x):
        """逆变换（使用牛顿迭代法近似）"""
        z = x.copy()
        for _ in range(10):
            z = z - (f(z) - x) / (1 + np.cos(2 * z))
        return z
    
    x = f(z)
    ax2.hist(x, bins=50, density=True, alpha=0.7, color='purple')
    ax2.set_title('变换后的分布：X = f(Z)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('概率密度')
    
    plt.tight_layout()
    plt.savefig('./figure/nf_definition.png')
    plt.close()
    
    print("已生成可视化：Normalizing Flows 定义示例")


def transformation_flows():
    """
    变换流
    介绍如何通过组合多个可逆变换构建Flow模型
    """
    print("\n=== 变换流 ===")
    
    print("Normalizing Flows的核心思想是通过组合多个可逆变换来构建复杂的分布：")
    print("- Normalizing：变量变换在应用可逆变换后给出标准化密度")
    print("- Flow：可逆变换可以相互组合")
    print("\n变换组合：")
    print("z_M = f^{(M)} ∘ ... ∘ f^{(1)}(z_0) = f_theta(z_0)")
    
    print("\n边际似然计算：")
    print("p_X(x) = p_Z(z) * |det(∂f_theta(z)/∂z)|⁻¹")
    print("        = p_Z(z) * ∏_{m=1}^M |det(∂f_theta^{(m)}(z_{m-1})/∂z_{m-1})|⁻¹")
    
    print("\n逆变换计算：")
    print("z_0 = (f_theta^{(1)})^{-1}((f_theta^{(2)})^{-1}(...(f_theta^{(M)})^{-1}(z_M)...))")
    print("p_X(x) = p_Z(f_theta^{-1}(x)) * ∏_{m=1}^M |det(∂(f_theta^{(m)})^{-1}(z_m)/∂z_m)|")
    
    # 可视化多级变换流
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 初始分布
    z0 = np.random.normal(0, 1, 10000)
    axes[0].hist(z0, bins=50, density=True, alpha=0.7, color='blue')
    axes[0].set_title('z_0: 初始分布')
    axes[0].set_xlabel('z_0')
    axes[0].set_ylabel('概率密度')
    
    # 变换1：平移和缩放
    z1 = 2 * z0 + 1
    axes[1].hist(z1, bins=50, density=True, alpha=0.7, color='green')
    axes[1].set_title('z_1: 线性变换')
    axes[1].set_xlabel('z_1')
    axes[1].set_ylabel('概率密度')
    
    # 变换2：非线性正弦变换
    z2 = z1 + 0.3 * np.sin(3 * z1)
    axes[2].hist(z2, bins=50, density=True, alpha=0.7, color='orange')
    axes[2].set_title('z_2: 正弦变换')
    axes[2].set_xlabel('z_2')
    axes[2].set_ylabel('概率密度')
    
    # 变换3：指数变换（限制在正数域）
    z3 = np.exp(z2)
    axes[3].hist(z3, bins=50, density=True, alpha=0.7, color='purple')
    axes[3].set_title('z_3: 指数变换')
    axes[3].set_xlabel('z_3')
    axes[3].set_ylabel('概率密度')
    
    plt.tight_layout()
    plt.savefig('./figure/transformation_flows.png')
    plt.close()
    
    print("已生成可视化：多级变换流示例")


def learning_and_inference():
    """
    学习与推理
    介绍Normalizing Flows的学习和推理过程
    """
    print("\n=== 学习与推理 ===")
    
    print("学习方法：")
    print("- 通过数据集D上的最大似然估计进行学习")
    print("- 目标函数：log p_theta(D) = Σ_{x∈D} [log p_Z(f_theta^{-1}(x)) + log |det(∂f_theta^{-1}(x)/∂x)|]")
    print("- 可以使用梯度下降等优化算法进行训练")
    
    print("\n推理过程：")
    print("- 似然评估：通过逆变换x→z和变量变换公式进行精确似然评估")
    print("  p_theta(x) = p_Z(z) * |det(∂f_theta^{-1}(x)/∂x)|")
    
    print("\n采样过程：")
    print("- 通过正向变换z→x进行采样")
    print("  z ~ p_Z(z), x = f_theta(z)")
    
    print("\n潜在表示推断：")
    print("- 通过逆变换推断潜在表示（不需要推理网络！）")
    print("  z = f_theta^{-1}(x)")
    
    # 简单的学习示例
    print("\n简单学习示例：")
    print("假设我们有一个数据集x ~ N(2, 3)")
    print("我们使用NF模型：X = a Z + b，其中Z ~ N(0, 1)")
    
    # 生成模拟数据
    true_mu_x = 2
    true_sigma_x = 3
    x_data = np.random.normal(true_mu_x, true_sigma_x, 1000)
    
    # 理论上最优的参数
    optimal_a = true_sigma_x
    optimal_b = true_mu_x
    
    print(f"理论最优参数：a = {optimal_a}, b = {optimal_b}")
    print(f"雅可比行列式：|det(∂f^{-1}(x)/∂x)| = 1/a = {1/optimal_a:.4f}")
    
    # 可视化学习过程
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(x_data, bins=50, density=True, alpha=0.7, color='blue', label='数据集')
    
    # 绘制真实分布和模型分布
    x_range = np.linspace(-10, 14, 1000)
    ax.plot(x_range, norm.pdf(x_range, true_mu_x, true_sigma_x), 'r-', linewidth=2, label='真实分布')
    ax.plot(x_range, norm.pdf((x_range - optimal_b)/optimal_a, 0, 1) / optimal_a, 'g--', linewidth=2, label='NF模型')
    
    ax.set_title('NF模型学习示例')
    ax.set_xlabel('x')
    ax.set_ylabel('概率密度')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('./figure/learning_example.png')
    plt.close()
    
    print("已生成可视化：NF学习示例")


def flow_model_design_points():
    """
    Flow 模型的设计要点
    介绍设计Flow模型时需要考虑的关键点
    """
    print("\n=== Flow 模型的设计要点 ===")
    
    print("设计Flow模型时需要考虑以下几个关键点：")
    
    print("\n1. 简单先验p_Z(z)")
    print("   - 允许高效采样和易于处理的似然评估")
    print("   - 常用选择：各向同性高斯分布")
    
    print("\n2. 可逆变换")
    print("   - 具有易处理评估的可逆变换")
    print("   - 似然评估需要高效评估x→z映射")
    print("   - 采样需要高效评估z→x映射")
    
    print("\n3. 雅可比行列式计算")
    print("   - 计算似然需要评估n×n雅可比矩阵的行列式")
    print("   - 直接计算行列式的复杂度为O(n³)，在学习循环中成本过高")
    print("   - 核心思想：选择变换使得生成的雅可比矩阵具有特殊结构")
    print("   - 例如，三角矩阵的行列式是对角线元素的乘积，计算复杂度为O(n)")
    
    print("\n4. 计算效率权衡")
    print("   - 变换的表达能力与计算效率之间的权衡")
    print("   - 更复杂的变换可能提供更强的表达能力，但计算成本更高")
    print("   - 常见的高效变换结构：")
    print("     - 三角结构雅可比矩阵")
    print("     - 正交变换")
    print("     - 可分变换")
    
    # 可视化雅可比矩阵结构
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 随机矩阵
    n = 50
    random_mat = np.random.randn(n, n)
    ax1.imshow(np.abs(random_mat) > 0.1, cmap='binary')
    ax1.set_title('随机矩阵结构')
    ax1.set_xlabel('列')
    ax1.set_ylabel('行')
    
    # 下三角矩阵
    lower_tri = np.tril(np.random.randn(n, n))
    ax2.imshow(np.abs(lower_tri) > 0.1, cmap='binary')
    ax2.set_title('下三角矩阵结构')
    ax2.set_xlabel('列')
    ax2.set_ylabel('行')
    
    plt.tight_layout()
    plt.savefig('./figure/jacobian_structures.png')
    plt.close()
    
    print("已生成可视化：雅可比矩阵结构对比")
    print("注意：下三角矩阵的行列式计算仅需O(n)时间，而随机矩阵需要O(n³)时间")


def main():
    """
    主函数，运行所有示例
    """
    print("Normalizing Flows - 知识点3：Normalizing Flows 基础")
    print("=" * 60)
    
    normalizing_flows_definition()
    transformation_flows()
    learning_and_inference()
    flow_model_design_points()
    
    print("\n" + "=" * 60)
    print("知识点3：Normalizing Flows 基础 示例演示完成")


if __name__ == "__main__":
    main()
