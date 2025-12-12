#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalizing Flows - 知识点1：基本概念与背景

本文件介绍了Normalizing Flows的基本概念和背景知识，包括：
1. 从简单先验到复杂数据分布
2. VAE与Normalizing Flows的对比
3. 连续随机变量回顾
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


def simple_prior_to_complex_distribution():
    """
    从简单先验到复杂数据分布
    展示如何通过变换将简单分布转换为复杂分布
    """
    print("=== 从简单先验到复杂数据分布 ===")
    print("任何模型分布 p_theta(x) 所需的理想性质：")
    print("- 易于评估，具有闭合形式的密度函数（对训练有用）")
    print("- 易于采样（对生成有用）")
    print("- 许多简单分布满足上述性质，例如高斯分布、均匀分布")
    print("- 遗憾的是，数据分布更为复杂（多模态）")
    print("- Normalizing Flows（NFs）的核心思想：通过可逆变换将简单分布映射到复杂分布")
    
    # 可视化简单分布到复杂分布的转换
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 简单高斯分布
    z = np.random.normal(0, 1, 10000)
    ax1.hist(z, bins=50, density=True, alpha=0.7, color='blue')
    ax1.set_title('简单分布：标准高斯')
    ax1.set_xlabel('z')
    ax1.set_ylabel('概率密度')
    
    # 通过变换创建复杂分布
    # 使用sigmoid变换和另一个非线性变换创建多模态分布
    x1 = 2 * norm.cdf(z) - 1  # 将高斯映射到[-1, 1]区间
    x2 = np.sin(3 * x1) * np.exp(-x1**2) + 0.5 * x1
    
    ax2.hist(x2, bins=50, density=True, alpha=0.7, color='purple')
    ax2.set_title('复杂分布：通过变换生成')
    ax2.set_xlabel('x')
    ax2.set_ylabel('概率密度')
    
    plt.tight_layout()
    plt.savefig('./figure/simple_to_complex.png')
    plt.close()
    
    print("已生成可视化：简单分布到复杂分布的转换")


def vae_vs_normalizing_flows():
    """
    VAE与Normalizing Flows的对比
    比较两种生成模型的异同
    """
    print("\n=== VAE与Normalizing Flows的对比 ===")
    print("相似点：")
    print("- 从简单分布 z ~ p(z) = N(0, I_k) 开始")
    print("- 通过变换生成复杂分布")
    print("- 边缘分布 p_theta(x) = ∫ p_theta(x, z)dz 非常复杂/灵活")
    
    print("\n不同点：")
    print("- VAE：使用概率映射 x ~ p_theta(x|z) = N(f_theta(z), σ²_dec I_n)")
    print("- Normalizing Flows：使用确定性可逆函数 x = f_theta(z)")
    print("- 在NF中，对于任何x都有唯一对应的z")
    print("- NF可以精确计算似然，而VAE只能估计下界")


def continuous_random_variables_review():
    """
    连续随机变量回顾
    复习连续随机变量的基本概念
    """
    print("\n=== 连续随机变量回顾 ===")
    print("累积分布函数（CDF）：F_X(x) = P(X ≤ x)")
    print("概率密度函数（pdf）：p_X(x) = F'_X(x) = dF_X(x)/dx")
    
    # 示例：高斯分布和均匀分布
    print("\n高斯分布：X ~ N(μ, σ²)")
    print("p_X(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))")
    
    print("\n均匀分布：X ~ U(a, b)")
    print("p_X(x) = 1/(b-a) * I_{a ≤ x ≤ b}")
    
    # 多维高斯分布
    print("\n多维高斯分布：X ~ N(μ, Σ)")
    print("p_X(x) = (1/√((2π)^d det(Σ))) * exp(-(x-μ)^T Σ⁻¹(x-μ)/2)")
    
    # 可视化1D高斯分布
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(-3, 3, 1000)
    ax.plot(x, norm.pdf(x, 0, 1), label='标准高斯', color='blue')
    ax.fill_between(x, norm.pdf(x, 0, 1), alpha=0.3, color='blue')
    ax.set_title('1D 高斯分布')
    ax.set_xlabel('x')
    ax.set_ylabel('概率密度')
    ax.legend()
    plt.tight_layout()
    plt.savefig('./figure/gaussian_1d.png')
    plt.close()
    
    print("已生成可视化：1D高斯分布")


def main():
    """
    主函数，运行所有示例
    """
    print("Normalizing Flows - 知识点1：基本概念与背景")
    print("=" * 60)
    
    simple_prior_to_complex_distribution()
    vae_vs_normalizing_flows()
    continuous_random_variables_review()
    
    print("\n" + "=" * 60)
    print("知识点1：基本概念与背景 示例演示完成")


if __name__ == "__main__":
    main()
