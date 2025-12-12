#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最优传输（Optimal Transport）知识分类 - 知识点5：矩阵缩放与Sinkhorn-Knopp算法

本文件对应知识点5的内容，主要介绍矩阵缩放与Sinkhorn-Knopp算法。
"""

import numpy as np
import matplotlib.pyplot as plt

def matrix_scaling():
    """
    矩阵缩放
    - 令γ_{c,λ} = arg min_{γ∈Γ(p,q)} ⟨c, γ⟩ - λ H(γ) = arg min_{γ∈Γ(p,q)} D_KL(γ || p_{c,λ})
    - 最优解满足：Γ_{c,λ} = Diag(u) K Diag(v)
    - 其中K ∈ R^(m×n)，k_{ij} = exp(-c(x_i, y_j)/λ)
    - 对应的u和v满足约束：
      p = Γ_{c,λ} 1_n = Diag(u) K v
      q = Γ_{c,λ}^T 1_m = Diag(v) K^T u
    - 这是一个矩阵缩放问题的实例，即寻找矩阵的缩放，使其列和行分别求和到两个给定的向量
    """
    print("="*60)
    print("知识点5：矩阵缩放与Sinkhorn-Knopp算法")
    print("="*60)
    print("\n1. 矩阵缩放")
    print("- 令γ_{c,λ} = arg min_{γ∈Γ(p,q)} ⟨c, γ⟩ - λ H(γ) = arg min_{γ∈Γ(p,q)} D_KL(γ || p_{c,λ})")
    print("- 最优解满足：Γ_{c,λ} = Diag(u) K Diag(v)")
    print("- 其中K ∈ R^(m×n)，k_{ij} = exp(-c(x_i, y_j)/λ)")
    print("- 对应的u和v满足约束：")
    print("  p = Γ_{c,λ} 1_n = Diag(u) K v")
    print("  q = Γ_{c,λ}^T 1_m = Diag(v) K^T u")
    print("- 这是一个矩阵缩放问题的实例，即寻找矩阵的缩放，使其列和行分别求和到两个给定的向量")
    print()

def sinkhorn_knopp_algorithm():
    """
    Sinkhorn-Knopp算法
    - 矩阵缩放问题：找到u, v使得
      p = Γ_{c,λ} 1_n = Diag(u) K v
      q = Γ_{c,λ}^T 1_m = Diag(v) K^T u
    - Sinkhorn-Knopp算法：
      - 初始化u^(0) = 1_m和v^(0) = 1_n
      - 交替更新：u^(ℓ+1) = p / (K v^(ℓ))，v^(ℓ+1) = q / (K^T u^(ℓ+1))
    """
    print("2. Sinkhorn-Knopp算法")
    print("- 矩阵缩放问题：找到u, v使得")
    print("  p = Γ_{c,λ} 1_n = Diag(u) K v")
    print("  q = Γ_{c,λ}^T 1_m = Diag(v) K^T u")
    print("- Sinkhorn-Knopp算法：")
    print("  - 初始化u^(0) = 1_m和v^(0) = 1_n")
    print("  - 交替更新：u^(ℓ+1) = p / (K v^(ℓ))，v^(ℓ+1) = q / (K^T u^(ℓ+1))")
    print()

def from_dual_to_primal():
    """
    从对偶迭代到原始迭代
    - 考虑偶数情况。原始更新为：
      Γ^(2ℓ+1) = Diag(p / (Γ^(2ℓ) 1_n)) Γ^(2ℓ)
    - 对应的对偶更新为：
      Γ^(2ℓ+1) = Diag(u^(2ℓ+1)) K Diag(v^(2ℓ))
               = Diag(p / (K v^(2ℓ))) K Diag(v^(2ℓ))
               = Diag(p / (K v^(2ℓ))) Diag(1/u^(2ℓ)) Γ^(2ℓ)
               = Diag(p / (u^(2ℓ) K v^(2ℓ))) Γ^(2ℓ)
               = Diag(p / (Γ^(2ℓ) 1_n)) Γ^(2ℓ)
    """
    print("3. 从对偶迭代到原始迭代")
    print("- 考虑偶数情况。原始更新为：")
    print("  Γ^(2ℓ+1) = Diag(p / (Γ^(2ℓ) 1_n)) Γ^(2ℓ)")
    print("- 对应的对偶更新为：")
    print("  Γ^(2ℓ+1) = Diag(u^(2ℓ+1)) K Diag(v^(2ℓ))")
    print("           = Diag(p / (K v^(2ℓ))) K Diag(v^(2ℓ))")
    print("           = Diag(p / (K v^(2ℓ))) Diag(1/u^(2ℓ)) Γ^(2ℓ)")
    print("           = Diag(p / (u^(2ℓ) K v^(2ℓ))) Γ^(2ℓ)")
    print("           = Diag(p / (Γ^(2ℓ) 1_n)) Γ^(2ℓ)")
    print()

def sinkhorn_knopp_implementation():
    """
    Sinkhorn-Knopp算法的实现
    - 创建一个简单的示例来演示算法的工作原理
    """
    print("\n4. Sinkhorn-Knopp算法实现")
    
    # 创建两个简单的分布
    p = np.array([0.5, 0.3, 0.2])  # 源分布
    q = np.array([0.2, 0.5, 0.3])  # 目标分布
    
    print(f"源分布 p: {p}")
    print(f"目标分布 q: {q}")
    
    # 创建一个简单的成本矩阵
    C = np.array([[0, 2, 4],
                  [2, 0, 2],
                  [4, 2, 0]])
    
    print(f"\n成本矩阵 C:")
    print(C)
    
    # 正则化参数
    λ = 0.5
    
    # 计算核矩阵 K
    K = np.exp(-C / λ)
    
    print(f"\n正则化参数 λ = {λ}")
    print(f"核矩阵 K:")
    print(K)
    
    # Sinkhorn-Knopp算法实现
    def sinkhorn(p, q, K, max_iter=100, tol=1e-6):
        m, n = K.shape
        u = np.ones(m) / m
        v = np.ones(n) / n
        
        for i in range(max_iter):
            # 保存旧的u和v用于收敛检查
            u_old = u.copy()
            v_old = v.copy()
            
            # 更新u
            u = p / (K @ v)
            
            # 更新v
            v = q / (K.T @ u)
            
            # 检查收敛性
            if np.max(np.abs(u - u_old)) < tol and np.max(np.abs(v - v_old)) < tol:
                print(f"\n算法在迭代 {i+1} 次后收敛")
                break
        
        # 计算传输计划
        gamma = np.diag(u) @ K @ np.diag(v)
        
        return u, v, gamma
    
    # 运行Sinkhorn-Knopp算法
    u, v, gamma = sinkhorn(p, q, K)
    
    print(f"\n最优行缩放因子 u:")
    print(u)
    print(f"最优列缩放因子 v:")
    print(v)
    print(f"\n最优传输计划 Γ:")
    print(gamma)
    
    # 验证约束
    print(f"\n验证约束:")
    print(f"行和: {np.sum(gamma, axis=1)}")
    print(f"源分布 p: {p}")
    print(f"行约束误差: {np.max(np.abs(np.sum(gamma, axis=1) - p)):.10f}")
    
    print(f"\n列和: {np.sum(gamma, axis=0)}")
    print(f"目标分布 q: {q}")
    print(f"列约束误差: {np.max(np.abs(np.sum(gamma, axis=0) - q)):.10f}")
    
    return p, q, C, K, u, v, gamma, λ

def matrix_scaling_example():
    """
    矩阵缩放示例
    - 展示矩阵缩放的概念和应用
    """
    print("\n5. 矩阵缩放示例")
    
    # 创建一个随机非负矩阵
    np.random.seed(42)
    A = np.random.rand(3, 3)
    
    print(f"原始矩阵 A:")
    print(A)
    print(f"原始行和: {np.sum(A, axis=1)}")
    print(f"原始列和: {np.sum(A, axis=0)}")
    
    # 定义目标行和和列和
    target_row_sums = np.array([1.5, 2.0, 1.0])
    target_col_sums = np.array([1.2, 1.8, 1.5])
    
    print(f"\n目标行和: {target_row_sums}")
    print(f"目标列和: {target_col_sums}")
    
    # 使用Sinkhorn-Knopp算法进行矩阵缩放
    def matrix_scaling(A, target_row_sums, target_col_sums, max_iter=100, tol=1e-6):
        m, n = A.shape
        
        # 初始化缩放因子
        u = np.ones(m)
        v = np.ones(n)
        
        for i in range(max_iter):
            u_old = u.copy()
            v_old = v.copy()
            
            # 更新u以匹配目标行和
            u = target_row_sums / (A @ v)
            
            # 更新v以匹配目标列和
            v = target_col_sums / (A.T @ u)
            
            # 检查收敛性
            if np.max(np.abs(u - u_old)) < tol and np.max(np.abs(v - v_old)) < tol:
                print(f"\n矩阵缩放算法在迭代 {i+1} 次后收敛")
                break
        
        # 缩放后的矩阵
        scaled_A = np.diag(u) @ A @ np.diag(v)
        
        return scaled_A, u, v
    
    # 运行矩阵缩放
    scaled_A, u, v = matrix_scaling(A, target_row_sums, target_col_sums)
    
    print(f"\n缩放后的矩阵:")
    print(scaled_A)
    print(f"缩放后的行和: {np.sum(scaled_A, axis=1)}")
    print(f"缩放后的列和: {np.sum(scaled_A, axis=0)}")
    print(f"行约束误差: {np.max(np.abs(np.sum(scaled_A, axis=1) - target_row_sums)):.10f}")
    print(f"列约束误差: {np.max(np.abs(np.sum(scaled_A, axis=0) - target_col_sums)):.10f}")
    
    return A, scaled_A, u, v, target_row_sums, target_col_sums

def visualize_sinkhorn_results(p, q, C, gamma, λ):
    """
    可视化Sinkhorn-Knopp算法结果
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制传输计划
    im = ax1.imshow(gamma, cmap='viridis', interpolation='nearest')
    ax1.set_title(f'最优传输计划 (λ={λ})')
    ax1.set_xlabel('目标分布索引')
    ax1.set_ylabel('源分布索引')
    
    # 添加颜色条
    fig.colorbar(im, ax=ax1, label='传输质量')
    
    # 在每个单元格中添加数值
    for i in range(gamma.shape[0]):
        for j in range(gamma.shape[1]):
            ax1.text(j, i, f'{gamma[i, j]:.3f}', ha='center', va='center', color='white')
    
    # 绘制成本矩阵和传输计划的关系
    ax2.scatter(C.flatten(), gamma.flatten())
    ax2.set_title('成本与传输计划的关系')
    ax2.set_xlabel('成本 c_ij')
    ax2.set_ylabel('传输计划 γ_ij')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figure/sinkhorn_results.png')
    print("\n已生成Sinkhorn-Knopp算法结果可视化图：./figure/sinkhorn_results.png")
    plt.close()

def compare_with_ot():
    """
    与标准最优传输的比较
    - 展示Sinkhorn-Knopp算法与标准OT的区别
    """
    print("\n6. 与标准最优传输的比较")
    
    # 这里我们将Sinkhorn-Knopp的结果与标准OT进行比较
    # 由于标准OT的实现较为复杂，我们使用一个简化的示例
    
    # 创建简单的分布和成本矩阵
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    C = np.array([[0, 1], [1, 0]])
    
    print(f"源分布 p: {p}")
    print(f"目标分布 q: {q}")
    print(f"成本矩阵 C:")
    print(C)
    
    # 标准OT解（对于这个简单情况，我们可以直接写出解）
    ot_gamma = np.array([[0.5, 0], [0, 0.5]])
    
    print(f"\n标准OT传输计划:")
    print(ot_gamma)
    print(f"标准OT成本: {np.sum(C * ot_gamma):.2f}")
    
    # 使用Sinkhorn-Knopp算法求解不同λ的值
    for λ in [0.1, 0.5, 1.0]:
        K = np.exp(-C / λ)
        
        def sinkhorn(p, q, K, max_iter=100, tol=1e-6):
            m, n = K.shape
            u = np.ones(m)
            v = np.ones(n)
            
            for i in range(max_iter):
                u_old = u.copy()
                v_old = v.copy()
                
                u = p / (K @ v)
                v = q / (K.T @ u)
                
                if np.max(np.abs(u - u_old)) < tol and np.max(np.abs(v - v_old)) < tol:
                    break
            
            gamma = np.diag(u) @ K @ np.diag(v)
            return gamma
        
        sinkhorn_gamma = sinkhorn(p, q, K)
        
        print(f"\nSinkhorn-Knopp传输计划 (λ={λ}):")
        print(sinkhorn_gamma)
        print(f"Sinkhorn-Knopp成本: {np.sum(C * sinkhorn_gamma):.2f}")
        print(f"与标准OT的L2误差: {np.linalg.norm(sinkhorn_gamma - ot_gamma):.6f}")

if __name__ == "__main__":
    matrix_scaling()
    sinkhorn_knopp_algorithm()
    from_dual_to_primal()
    p, q, C, K, u, v, gamma, λ = sinkhorn_knopp_implementation()
    matrix_scaling_example()
    visualize_sinkhorn_results(p, q, C, gamma, λ)
    compare_with_ot()
