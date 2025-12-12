#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最优传输（Optimal Transport）知识分类 - 知识点2：最优传输问题的数学定义

本文件对应知识点2的内容，主要介绍最优传输问题的数学定义。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def ot_problem_definition():
    """
    最优传输问题的数学定义
    - 设(X, p)和(Y, q)是有限概率空间，|X|=m，|Y|=n
    - 设Γ(p, q)⊆Δ^(m×n)是乘积空间X×Y上具有边缘分布p和q的分布集合
    - 考虑成本函数c: X×Y→R+，OT问题定义为：
      d_c(p, q) = min_{γ∈Γ(p,q)} ⟨c, γ⟩ = min_{γ∈Γ(p,q)} Σ_{x,y} c(x,y)γ(x,y)
    """
    print("="*60)
    print("知识点2：最优传输问题的数学定义")
    print("="*60)
    print("\n1. The OT problem")
    print("- 设(X, p)和(Y, q)是有限概率空间，|X|=m，|Y|=n")
    print("- 设Γ(p, q)⊆Δ^(m×n)是乘积空间X×Y上具有边缘分布p和q的分布集合")
    print("- 考虑成本函数c: X×Y→R+，OT问题定义为：")
    print("  d_c(p, q) = min_{γ∈Γ(p,q)} ⟨c, γ⟩ = min_{γ∈Γ(p,q)} Σ_{x,y} c(x,y)γ(x,y)")
    print()
    
    # 示例：计算两个简单分布之间的Wasserstein-1距离
    print("\n示例：计算两个简单分布之间的Wasserstein-1距离")
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
    
    # 使用线性规划求解OT问题
    m, n = len(p), len(q)
    
    # 转换为线性规划格式
    c_flat = c.flatten()
    
    # 约束条件：
    # 1. 行和等于p: A_eq_row @ x = p
    # 2. 列和等于q: A_eq_col @ x = q
    
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
    result = linprog(c_flat, A_eq=A_eq, b_eq=b_eq, method='highs')
    
    if result.success:
        gamma_opt = result.x.reshape(m, n)
        print(f"\n最优传输计划 γ:")
        print(gamma_opt)
        print(f"\n最小传输成本 d_c(p, q): {result.fun:.4f}")
        print(f"这就是Wasserstein-1距离: W_1(p, q) = {result.fun:.4f}")
    else:
        print("\n线性规划求解失败")
    
    return X, Y, p, q, c, gamma_opt if result.success else None

def ot_problem_matrix_form():
    """
    OT问题的矩阵形式
    - 设Γ∈R^(m×n)是对应于γ的矩阵，其(i,j)项为γ(x_i, y_j)
    - 边缘约束可以写成：
      p(x_i) = Σ_{j=1}^n γ(x_i, y_j) = (Γ 1_n)_i
      q(y_j) = Σ_{i=1}^m γ(x_i, y_j) = (Γ^T 1_m)_j
    - 因此，OT问题可以写成：
      d_c(p, q) = min_{Γ 1_n=p, Γ^T 1_m=q} Σ_{x,y} c(x,y)γ(x,y)
    """
    print("\n2. The OT problem (矩阵形式)")
    print("- 设Γ∈R^(m×n)是对应于γ的矩阵，其(i,j)项为γ(x_i, y_j)")
    print("- 边缘约束可以写成：")
    print("  p(x_i) = Σ_{j=1}^n γ(x_i, y_j) = (Γ 1_n)_i")
    print("  q(y_j) = Σ_{i=1}^m γ(x_i, y_j) = (Γ^T 1_m)_j")
    print("- 因此，OT问题可以写成：")
    print("  d_c(p, q) = min_{Γ 1_n=p, Γ^T 1_m=q} Σ_{x,y} c(x,y)γ(x,y)")

def visualize_transport_plan(X, Y, p, q, c, gamma):
    """
    可视化传输计划
    """
    if gamma is None:
        return
    
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
    ax3.set_title('最优传输计划')
    
    # 绘制网格
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
    plt.savefig('./figure/ot_transport_plan.png')
    print("\n已生成OT传输计划可视化图：./figure/ot_transport_plan.png")
    plt.close()

if __name__ == "__main__":
    X, Y, p, q, c, gamma = ot_problem_definition()
    ot_problem_matrix_form()
    if gamma is not None:
        visualize_transport_plan(X, Y, p, q, c, gamma)
