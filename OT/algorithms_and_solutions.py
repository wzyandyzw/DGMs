#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最优传输（Optimal Transport）知识分类 - 知识点4：算法与求解方法

本文件对应知识点4的内容，主要介绍最优传输的算法与求解方法。
"""

import numpy as np
import matplotlib.pyplot as plt

def conceptual_algorithm():
    """
    概念算法
    - 目标是找到：
      arg min_{γ∈Γ(p,q)} ⟨c, γ⟩ - λ H(γ) = arg min_{γ∈Γ(p,q)} D_KL(γ || p_{c,λ})
    - 两个约束条件：Γ 1_n = p 和 Γ^T 1_m = q
    - 交替最小化方法：初始化 γ^(0) = p_{c,λ}，然后迭代：
      γ^(ℓ+1) = 
        { arg min_{Γ 1_n = p} D_KL(γ || γ^(ℓ)),  ℓ 偶数
        { arg min_{Γ^T 1_m = q} D_KL(γ || γ^(ℓ)),  ℓ 奇数
    - 这些子问题有闭合形式解！
    """
    print("="*60)
    print("知识点4：算法与求解方法")
    print("="*60)
    print("\n1. A conceptual algorithm")
    print("- 目标是找到：")
    print("  arg min_{γ∈Γ(p,q)} ⟨c, γ⟩ - λ H(γ) = arg min_{γ∈Γ(p,q)} D_KL(γ || p_{c,λ})")
    print("- 两个约束条件：Γ 1_n = p 和 Γ^T 1_m = q")
    print("- 交替最小化方法：初始化 γ^(0) = p_{c,λ}，然后迭代：")
    print("  γ^(ℓ+1) = ")
    print("    { arg min_{Γ 1_n = p} D_KL(γ || γ^(ℓ)),  ℓ 偶数")
    print("    { arg min_{Γ^T 1_m = q} D_KL(γ || γ^(ℓ)),  ℓ 奇数")
    print("- 这些子问题有闭合形式解！")
    print()

def solve_subproblem_even():
    """
    求解偶数迭代的子问题
    - 假设ℓ是偶数，我们的目标是找到 arg min_{Γ 1_n = p} D_KL(γ || γ^(ℓ))
    - 引入拉格朗日乘子（回忆KKT条件）：
      L_v(γ) := D_KL(γ || γ^(ℓ)) + v^T (Γ 1_n - p)
    - 一阶最优性：
      ∂L_v/∂γ_ij = 1 + log(γ_ij) - log(γ_ij^(ℓ)) + v_i = 0
    - 这给出：
      γ_ij = exp(-(1 + v_i)) · γ_ij^(ℓ)
    - 由于Σ_j γ_ij = p_i，我们得到：
      exp(-(1 + v_i)) · Σ_j γ_ij^(ℓ) = p_i,  exp(-(1 + v_i)) = p_i / Σ_j γ_ij^(ℓ)
    - 因此：
      Γ = Diag(p / (Γ^(ℓ) 1_n)) Γ^(ℓ)
    - 注意：Γ^(ℓ) 1_n 不保证等于 p
    """
    print("2. Solving the sub-problems (偶数迭代)")
    print("- 假设ℓ是偶数，我们的目标是找到 arg min_{Γ 1_n = p} D_KL(γ || γ^(ℓ))")
    print("- 引入拉格朗日乘子（回忆KKT条件）：")
    print("  L_v(γ) := D_KL(γ || γ^(ℓ)) + v^T (Γ 1_n - p)")
    print("- 一阶最优性：")
    print("  ∂L_v/∂γ_ij = 1 + log(γ_ij) - log(γ_ij^(ℓ)) + v_i = 0")
    print("- 这给出：")
    print("  γ_ij = exp(-(1 + v_i)) · γ_ij^(ℓ)")
    print("- 由于Σ_j γ_ij = p_i，我们得到：")
    print("  exp(-(1 + v_i)) · Σ_j γ_ij^(ℓ) = p_i,  exp(-(1 + v_i)) = p_i / Σ_j γ_ij^(ℓ)")
    print("- 因此：")
    print("  Γ = Diag(p / (Γ^(ℓ) 1_n)) Γ^(ℓ)")
    print("- 注意：Γ^(ℓ) 1_n 不保证等于 p")
    print()

def solve_subproblem_odd():
    """
    求解奇数迭代的子问题
    - 类似地，假设ℓ是奇数，我们的目标是找到 arg min_{Γ^T 1_m = q} D_KL(γ || γ^(ℓ))
    - 闭合形式解为：
      Γ = Diag(q / ((Γ^(ℓ))^T 1_m)) Γ^(ℓ)
    - 每次迭代中，Γ ∈ R^(m×n)
    - 但只有 m + n 个约束
    - 我们可以在对偶空间中更高效地优化
    """
    print("3. Solving the sub-problems (奇数迭代)")
    print("- 类似地，假设ℓ是奇数，我们的目标是找到 arg min_{Γ^T 1_m = q} D_KL(γ || γ^(ℓ))")
    print("- 闭合形式解为：")
    print("  Γ = Diag(q / ((Γ^(ℓ))^T 1_m)) Γ^(ℓ)")
    print("- 每次迭代中，Γ ∈ R^(m×n)")
    print("- 但只有 m + n 个约束")
    print("- 我们可以在对偶空间中更高效地优化")
    print()

def dual_perspective():
    """
    对偶视角
    - 令γ_{c,λ} = arg min_{γ∈Γ(p,q)} ⟨c, γ⟩ - λ H(γ) = arg min_{γ∈Γ(p,q)} D_KL(γ || p_{c,λ})
    - 命题：令K ∈ R^(m×n)，其中k_{ij} = exp(-c(x_i, y_j)/λ)。则存在u ∈ R^m和v ∈ R^n，使得：
      Γ_{c,λ} = Diag(u) K Diag(v)
    - 证明：引入对偶变量s ∈ R^n和t ∈ R^m，考虑拉格朗日乘子：
      L_{s,t}(γ) := ⟨c, γ⟩ - λ H(γ) - s^T (Γ 1_n - p) - t^T (Γ^T 1_m - q)
      一阶最优性发生在γ_{c,λ}满足：
      ∂L_{s,t}/∂γ_ij = 0
      这给出：
      c(x_i, y_j) + λ(1 + log((Γ_{c,λ})_{ij})) - s_i - t_j = 0
      然后：
      (Γ_{c,λ})_{ij} = exp((-c(x_i, y_j) + s_i + t_j)/λ - 1) = exp(s_i/λ - 1/2) k_{ij} exp(t_j/λ - 1/2)
      这就完成了证明
    """
    print("4. The dual perspective")
    print("- 令γ_{c,λ} = arg min_{γ∈Γ(p,q)} ⟨c, γ⟩ - λ H(γ) = arg min_{γ∈Γ(p,q)} D_KL(γ || p_{c,λ})")
    print("- 命题：令K ∈ R^(m×n)，其中k_{ij} = exp(-c(x_i, y_j)/λ)。则存在u ∈ R^m和v ∈ R^n，使得：")
    print("  Γ_{c,λ} = Diag(u) K Diag(v)")
    print("- 证明：引入对偶变量s ∈ R^n和t ∈ R^m，考虑拉格朗日乘子：")
    print("  L_{s,t}(γ) := ⟨c, γ⟩ - λ H(γ) - s^T (Γ 1_n - p) - t^T (Γ^T 1_m - q)")
    print("  一阶最优性发生在γ_{c,λ}满足：")
    print("  ∂L_{s,t}/∂γ_ij = 0")
    print("  这给出：")
    print("  c(x_i, y_j) + λ(1 + log((Γ_{c,λ})_{ij})) - s_i - t_j = 0")
    print("  然后：")
    print("  (Γ_{c,λ})_{ij} = exp((-c(x_i, y_j) + s_i + t_j)/λ - 1) = exp(s_i/λ - 1/2) k_{ij} exp(t_j/λ - 1/2)")
    print("  这就完成了证明")
    print()

def alternating_minimization_example():
    """
    交替最小化方法示例
    - 创建两个简单的分布
    - 实现交替最小化方法求解熵正则化OT
    """
    print("\n5. 交替最小化方法示例")
    
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
    
    # 正则化参数
    λ = 0.5
    
    # 初始化γ^(0) = p_{c,λ}
    k = np.exp(-c / λ)
    Z_λ = np.sum(k)
    gamma = k / Z_λ
    
    print(f"\n正则化参数 λ = {λ}")
    print(f"初始γ^(0):")
    print(gamma)
    
    # 交替最小化迭代
    max_iter = 10
    
    for l in range(max_iter):
        if l % 2 == 0:  # 偶数迭代：满足行约束 Γ 1_n = p
            row_sums = np.sum(gamma, axis=1)
            scaling_factors = p / row_sums
            gamma = np.diag(scaling_factors) @ gamma
            print(f"\n迭代 {l+1} (偶数):")
            print(f"行和: {row_sums}")
            print(f"缩放因子: {scaling_factors}")
        else:  # 奇数迭代：满足列约束 Γ^T 1_m = q
            col_sums = np.sum(gamma, axis=0)
            scaling_factors = q / col_sums
            gamma = gamma @ np.diag(scaling_factors)
            print(f"\n迭代 {l+1} (奇数):")
            print(f"列和: {col_sums}")
            print(f"缩放因子: {scaling_factors}")
        
        print(f"γ^({l+1}):")
        print(gamma)
        
        # 检查收敛性
        current_row_sums = np.sum(gamma, axis=1)
        current_col_sums = np.sum(gamma, axis=0)
        row_error = np.max(np.abs(current_row_sums - p))
        col_error = np.max(np.abs(current_col_sums - q))
        print(f"行约束误差: {row_error:.6f}")
        print(f"列约束误差: {col_error:.6f}")
        
        if row_error < 1e-6 and col_error < 1e-6:
            print(f"\n迭代 {l+1} 后收敛！")
            break
    
    # 计算最终的目标函数值
    cost = np.sum(c * gamma)
    entropy = -np.sum(gamma[gamma > 1e-10] * np.log(gamma[gamma > 1e-10]))
    objective = cost - λ * entropy
    
    print(f"\n最终结果：")
    print(f"最优传输计划 γ:")
    print(gamma)
    print(f"成本项: {cost:.4f}")
    print(f"熵项: {λ * entropy:.4f}")
    print(f"目标函数值: {objective:.4f}")
    
    return X, Y, p, q, c, gamma, λ

def dual_representation_example():
    """
    对偶表示示例
    - 展示如何将最优传输计划表示为 Γ = Diag(u) K Diag(v)
    """
    print("\n6. 对偶表示示例")
    
    # 创建两个简单的分布
    X = np.array([0, 1, 2])  # 源分布的支持点
    Y = np.array([0, 2, 4])  # 目标分布的支持点
    p = np.array([0.5, 0.3, 0.2])  # 源分布
    q = np.array([0.2, 0.5, 0.3])  # 目标分布
    
    # 计算成本矩阵（欧几里得距离）
    c = np.abs(X[:, np.newaxis] - Y)
    
    # 正则化参数
    λ = 0.5
    
    # 使用Sinkhorn算法求解最优传输计划
    u, v, gamma = sinkhorn_algorithm(p, q, c, λ, max_iter=1000, tol=1e-6)
    
    if gamma is not None:
        print(f"\n使用Sinkhorn算法求解的最优传输计划 γ:")
        print(gamma)
        
        # 验证对偶表示 Γ = Diag(u) K Diag(v)
        K = np.exp(-c / λ)
        gamma_dual = np.diag(u) @ K @ np.diag(v)
        
        print(f"\n对偶表示 Γ = Diag(u) K Diag(v) 得到的传输计划:")
        print(gamma_dual)
        
        # 检查误差
        error = np.max(np.abs(gamma - gamma_dual))
        print(f"\n两种表示方法的最大误差: {error:.10f}")
        
        if error < 1e-6:
            print("对偶表示验证成功！")
    
    return X, Y, p, q, c, gamma, u, v, λ

def sinkhorn_algorithm(p, q, C, ε, max_iter=1000, tol=1e-6):
    """
    Sinkhorn算法实现
    
    参数:
    p: 源分布，形状为(m,)
    q: 目标分布，形状为(n,)
    C: 成本矩阵，形状为(m, n)
    ε: 正则化参数
    max_iter: 最大迭代次数
    tol: 收敛阈值
    
    返回:
    u: 行缩放因子，形状为(m,)
    v: 列缩放因子，形状为(n,)
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
    
    return u, v, gamma

def visualize_alternating_minimization(X, Y, p, q, c, gamma, λ):
    """
    可视化交替最小化过程
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 绘制源分布
    ax1.set_title('源分布 p')
    ax1.bar(X, p, color='blue', alpha=0.7)
    ax1.set_ylim(0, max(p.max(), q.max()) * 1.1)
    ax1.set_xlabel('位置')
    ax1.set_ylabel('概率质量')
    
    # 绘制目标分布
    ax2.set_title('目标分布 q')
    ax2.bar(Y, q, color='red', alpha=0.7)
    ax2.set_ylim(0, max(p.max(), q.max()) * 1.1)
    ax2.set_xlabel('位置')
    ax2.set_ylabel('概率质量')
    
    # 绘制传输计划
    ax3.set_title(f'交替最小化后的传输计划 (λ={λ})')
    ax3.grid(True, linestyle='--', alpha=0.3)
    
    # 绘制传输箭头
    max_arrow_width = 0.8
    for i in range(len(X)):
        for j in range(len(Y)):
            if gamma[i, j] > 1e-6:  # 只绘制非零传输
                # 计算箭头的起点和终点
                start_x, start_y = X[i], 0
                end_x, end_y = Y[j], 0
                
                # 绘制箭头
                ax3.arrow(start_x, start_y, end_x - start_x, 0, 
                          length_includes_head=True, head_width=0.05, head_length=0.2,
                          color='green', alpha=0.6, width=gamma[i, j] * max_arrow_width)
    
    # 标记分布点
    ax3.scatter(X, np.zeros_like(X), s=p*500, color='blue', alpha=0.7, zorder=5)
    ax3.scatter(Y, np.zeros_like(Y), s=q*500, color='red', alpha=0.7, zorder=5)
    
    # 标记点的概率值
    for i in range(len(X)):
        ax3.text(X[i], -0.05, f'p={p[i]:.2f}', ha='center', va='top')
    for j in range(len(Y)):
        ax3.text(Y[j], 0.05, f'q={q[j]:.2f}', ha='center', va='bottom')
    
    ax3.set_xlim(min(X.min(), Y.min()) - 0.5, max(X.max(), Y.max()) + 0.5)
    ax3.set_ylim(-0.1, 0.1)
    ax3.set_xlabel('位置')
    ax3.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('./figure/alternating_minimization.png')
    print("\n已生成交替最小化可视化图：./figure/alternating_minimization.png")
    plt.close()

if __name__ == "__main__":
    conceptual_algorithm()
    solve_subproblem_even()
    solve_subproblem_odd()
    dual_perspective()
    X, Y, p, q, c, gamma, λ = alternating_minimization_example()
    visualize_alternating_minimization(X, Y, p, q, c, gamma, λ)
    X, Y, p, q, c, gamma, u, v, λ = dual_representation_example()
