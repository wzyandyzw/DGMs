#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最优传输（Optimal Transport）知识分类 - 知识点3：熵正则化最优传输

本文件对应知识点3的内容，主要介绍熵正则化最优传输的概念。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def entropy_regularized_ot_definition():
    """
    熵正则化最优传输的定义
    - OT问题是线性规划（LP），可以通过LP求解器求解
    - 但我们可以通过熵正则化OT问题（带有近似误差）来简化它：
      d_{c,λ}(p, q) := min_{γ∈Γ(p,q)} ⟨c, γ⟩ - λ H(γ)
      其中λ>0是正则化参数，H(γ) = -Σ_{x,y} γ(x,y)log(γ(x,y))是熵
    """
    print("="*60)
    print("知识点3：熵正则化最优传输")
    print("="*60)
    print("\n1. Entropy-regularized OT")
    print("- OT问题是线性规划（LP），可以通过LP求解器求解")
    print("- 但我们可以通过熵正则化OT问题（带有近似误差）来简化它：")
    print("  d_{c,λ}(p, q) := min_{γ∈Γ(p,q)} ⟨c, γ⟩ - λ H(γ)")
    print("  其中λ>0是正则化参数，H(γ) = -Σ_{x,y} γ(x,y)log(γ(x,y))是熵")
    print()

def kl_divergence_relation():
    """
    与KL散度的关系
    - 类似于最大似然的KL散度重构：
      令k(x,y) = exp(-c(x,y)/λ)，Z_λ = Σ_{x,y} k(x,y)（因此k(x,y)/Z_λ是pdf p_{c,λ}）
      则D_KL(γ || p_{c,λ}) = Σ_{x,y} γ(x,y)log(γ(x,y)) - Σ_{x,y} γ(x,y)log(p_{c,λ})
                           = -H(γ) - Σ_{x,y} γ(x,y)log(k(x,y)/Z_λ)
                           = -H(γ) + (1/λ)⟨c, γ⟩ + log(Z_λ)
    - 因此：
      min_{γ∈Γ(p,q)} D_KL(γ || p_{c,λ}) = min_{γ∈Γ(p,q)} ⟨c, γ⟩ - λ H(γ)
    """
    print("2. 与KL散度的关系")
    print("- 类似于最大似然的KL散度重构：")
    print("  令k(x,y) = exp(-c(x,y)/λ)，Z_λ = Σ_{x,y} k(x,y)（因此k(x,y)/Z_λ是pdf p_{c,λ}）")
    print("  则D_KL(γ || p_{c,λ}) = Σ_{x,y} γ(x,y)log(γ(x,y)) - Σ_{x,y} γ(x,y)log(p_{c,λ})")
    print("                       = -H(γ) - Σ_{x,y} γ(x,y)log(k(x,y)/Z_λ)")
    print("                       = -H(γ) + (1/λ)⟨c, γ⟩ + log(Z_λ)")
    print("- 因此：")
    print("  min_{γ∈Γ(p,q)} D_KL(γ || p_{c,λ}) = min_{γ∈Γ(p,q)} ⟨c, γ⟩ - λ H(γ)")

def entropy_function(γ):
    """
    计算分布γ的熵
    H(γ) = -Σ_{x,y} γ(x,y)log(γ(x,y))
    """
    # 避免log(0)的情况
    γ_positive = γ[γ > 1e-10]
    return -np.sum(γ_positive * np.log(γ_positive))

def kl_divergence(γ, p_c_λ):
    """
    计算KL散度 D_KL(γ || p_c_λ)
    """
    # 避免log(0)的情况
    γ_positive = γ[γ > 1e-10]
    p_c_λ_positive = p_c_λ[γ > 1e-10]
    return np.sum(γ_positive * np.log(γ_positive / p_c_λ_positive))

def simple_entropy_regularized_ot_example():
    """
    简单的熵正则化OT示例
    - 创建两个简单的分布
    - 计算不同正则化参数λ下的熵正则化OT解
    """
    print("\n3. 熵正则化OT示例")
    
    # 创建两个简单的分布
    X = np.array([0, 1, 2])  # 源分布的支持点
    Y = np.array([0, 2, 4])  # 目标分布的支持点
    p = np.array([0.5, 0.3, 0.2])  # 源分布
    q = np.array([0.2, 0.5, 0.3])  # 目标分布
    
    print(f"源分布支持点 X: {X}")
    print(f"源分布概率 p: {p}")
    print(f"目标分布支持点 Y: {Y}")
    print(f"目标分布概率 q: {q}")
    
    # 计算成本矩阵（欧几里得距离）
    c = np.abs(X[:, np.newaxis] - Y)
    print(f"\n成本矩阵 c:")
    print(c)
    
    # 不同的正则化参数
    lambda_values = [0.1, 0.5, 1.0, 2.0]
    
    results = []
    
    for λ in lambda_values:
        print(f"\n正则化参数 λ = {λ}")
        
        # 计算核函数k(x,y) = exp(-c(x,y)/λ)
        k = np.exp(-c / λ)
        Z_λ = np.sum(k)
        p_c_λ = k / Z_λ  # pdf
        
        print(f"核函数k的和 Z_λ = {Z_λ:.4f}")
        
        # 使用KL散度形式求解熵正则化OT
        # 这里我们使用一个简单的方法：固定迭代次数的Sinkhorn算法
        gamma = sinkhorn_algorithm(p, q, c, λ, max_iter=1000, tol=1e-6)
        
        if gamma is not None:
            # 计算目标函数值
            cost = np.sum(c * gamma)
            entropy = entropy_function(gamma)
            objective = cost - λ * entropy
            
            # 计算KL散度
            kl = kl_divergence(gamma, p_c_λ)
            
            print(f"最小传输成本（包含熵正则化）: {objective:.4f}")
            print(f"其中成本项: {cost:.4f}, 熵项: {λ * entropy:.4f}")
            print(f"KL散度 D_KL(γ || p_c_λ): {kl:.4f}")
            
            results.append({
                'lambda': λ,
                'gamma': gamma,
                'cost': cost,
                'entropy': entropy,
                'objective': objective
            })
    
    return results, X, Y, p, q, c

def sinkhorn_algorithm(p, q, C, ε, max_iter=1000, tol=1e-6):
    """
    Sinkhorn算法实现（用于熵正则化OT）
    
    参数:
    p: 源分布，形状为(m,)
    q: 目标分布，形状为(n,)
    C: 成本矩阵，形状为(m, n)
    ε: 正则化参数
    max_iter: 最大迭代次数
    tol: 收敛阈值
    
    返回:
    gamma: 最优传输计划，形状为(m, n)
    """
    m, n = C.shape
    
    # 初始化u和v
    u = np.ones(m) / m
    v = np.ones(n) / n
    
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
    
    return gamma

def visualize_entropy_regularization_effect(results, X, Y, p, q, c):
    """
    可视化不同正则化参数对传输计划的影响
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        ax = axes[i]
        λ = result['lambda']
        gamma = result['gamma']
        
        ax.set_title(f'正则化参数 λ = {λ}')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 绘制传输箭头
        max_arrow_width = 0.8
        for i_x in range(len(X)):
            for i_y in range(len(Y)):
                if gamma[i_x, i_y] > 1e-6:  # 只绘制非零传输
                    # 计算箭头的起点和终点
                    start_x, start_y = X[i_x], 0
                    end_x, end_y = Y[i_y], 0
                    
                    # 绘制箭头
                    ax.arrow(start_x, start_y, end_x - start_x, 0, 
                              length_includes_head=True, head_width=0.05, head_length=0.2,
                              color='green', alpha=0.6, width=gamma[i_x, i_y] * max_arrow_width)
        
        # 标记分布点
        ax.scatter(X, np.zeros_like(X), s=p*500, color='blue', alpha=0.7, zorder=5)
        ax.scatter(Y, np.zeros_like(Y), s=q*500, color='red', alpha=0.7, zorder=5)
        
        # 标记点的概率值
        for i in range(len(X)):
            ax.text(X[i], -0.05, f'p={p[i]:.2f}', ha='center', va='top')
        for j in range(len(Y)):
            ax.text(Y[j], 0.05, f'q={q[j]:.2f}', ha='center', va='bottom')
        
        ax.set_xlim(min(X.min(), Y.min()) - 0.5, max(X.max(), Y.max()) + 0.5)
        ax.set_ylim(-0.1, 0.1)
        ax.set_xlabel('位置')
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('./figure/entropy_regularization_effect.png')
    print("\n已生成熵正则化效果可视化图：./figure/entropy_regularization_effect.png")
    plt.close()

def compare_with_standard_ot(X, Y, p, q, c):
    """
    比较标准OT和熵正则化OT的结果
    """
    print(f"\n4. 与标准OT的比较")
    
    # 求解标准OT
    m, n = len(p), len(q)
    c_flat = c.flatten()
    
    # 构建行约束矩阵
    A_eq_row = np.zeros((m, m*n))
    for i in range(m):
        A_eq_row[i, i*n:(i+1)*n] = 1
    
    # 构建列约束矩阵
    A_eq_col = np.zeros((n, m*n))
    for j in range(n):
        A_eq_col[j, j::n] = 1
    
    # 合并约束矩阵
    A_eq = np.vstack([A_eq_row, A_eq_col])
    b_eq = np.hstack([p, q])
    
    # 求解线性规划
    result_lp = linprog(c_flat, A_eq=A_eq, b_eq=b_eq, method='highs')
    
    if result_lp.success:
        gamma_standard = result_lp.x.reshape(m, n)
        cost_standard = result_lp.fun
        
        print(f"标准OT的最小传输成本: {cost_standard:.4f}")
        
        # 求解熵正则化OT（小λ）
        λ = 0.1
        gamma_regularized = sinkhorn_algorithm(p, q, c, λ, max_iter=1000, tol=1e-6)
        
        if gamma_regularized is not None:
            cost_regularized = np.sum(c * gamma_regularized)
            print(f"熵正则化OT (λ={λ})的传输成本: {cost_regularized:.4f}")
            print(f"相对误差: {abs(cost_regularized - cost_standard) / cost_standard * 100:.2f}%")
    
    return result_lp.success

if __name__ == "__main__":
    entropy_regularized_ot_definition()
    kl_divergence_relation()
    results, X, Y, p, q, c = simple_entropy_regularized_ot_example()
    visualize_entropy_regularization_effect(results, X, Y, p, q, c)
    compare_with_standard_ot(X, Y, p, q, c)
