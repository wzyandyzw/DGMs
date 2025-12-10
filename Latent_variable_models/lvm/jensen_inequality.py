# 知识点7：Jensen不等式
"""
- 对于凹函数f和α ∈ Δ^{N-1} := {v ∈ ℝ^N : v_n ∈ [0,1] ∀n ∈ [N], Σ_{n=1}^N v_n = 1}，有：
  f(Σ_{n=1}^N α_n x_n) ≥ Σ_{n=1}^N α_n f(x_n)
- 特别地，log(·)是凹函数
- 应用Jensen不等式（凹函数）：
  log(E_{z~q(z)} [g(z)]) = log(Σ_z q(z) g(z)) ≥ Σ_z q(z) log(g(z))
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 凹函数示例（log函数）
def concave_function_example():
    """演示凹函数的Jensen不等式"""
    print("=== 凹函数的Jensen不等式示例 ===")
    
    # 定义凹函数 f(x) = log(x)
    def f(x):
        return np.log(x)
    
    # 定义凸函数 g(x) = x²
    def g(x):
        return x**2
    
    # 选择两个点
    x1 = 1.0
    x2 = 5.0
    
    # 选择权重
    alpha = 0.3
    
    # 计算加权平均
    x_bar = alpha * x1 + (1 - alpha) * x2
    
    # 计算函数在加权平均点的值
    f_bar = f(x_bar)
    g_bar = g(x_bar)
    
    # 计算函数值的加权平均
    f_avg = alpha * f(x1) + (1 - alpha) * f(x2)
    g_avg = alpha * g(x1) + (1 - alpha) * g(x2)
    
    # 打印结果
    print(f"1. 两个点: x1 = {x1}, x2 = {x2}")
    print(f"2. 权重: alpha = {alpha}")
    print(f"3. 加权平均: x̄ = {alpha}*x1 + {1-alpha}*x2 = {x_bar}")
    
    print(f"\n4. 凹函数 f(x) = log(x):")
    print(f"   f(αx1 + (1-α)x2) = f({x_bar}) = {f_bar:.4f}")
    print(f"   αf(x1) + (1-α)f(x2) = {alpha}*f({x1}) + {1-alpha}*f({x2}) = {f_avg:.4f}")
    print(f"   Jensen不等式: f(αx1 + (1-α)x2) ≥ αf(x1) + (1-α)f(x2) → {f_bar:.4f} ≥ {f_avg:.4f} → {f_bar >= f_avg}")
    
    print(f"\n5. 凸函数 g(x) = x²:")
    print(f"   g(αx1 + (1-α)x2) = g({x_bar}) = {g_bar:.4f}")
    print(f"   αg(x1) + (1-α)g(x2) = {alpha}*g({x1}) + {1-alpha}*g({x2}) = {g_avg:.4f}")
    print(f"   Jensen不等式（凸函数）: g(αx1 + (1-α)x2) ≤ αg(x1) + (1-α)g(x2) → {g_bar:.4f} ≤ {g_avg:.4f} → {g_bar <= g_avg}")
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    # 1. 凹函数
    plt.subplot(1, 2, 1)
    x = np.linspace(0.1, 6, 100)
    plt.plot(x, f(x), 'b-', label='f(x) = log(x) (凹函数)')
    
    # 绘制两个点
    plt.scatter([x1, x2], [f(x1), f(x2)], c='red', s=100, label='数据点')
    
    # 绘制加权平均点
    plt.scatter([x_bar], [f_bar], c='green', s=150, marker='*', label='f(αx1 + (1-α)x2)')
    
    # 绘制函数值的加权平均
    plt.scatter([x_bar], [f_avg], c='orange', s=150, marker='o', label='αf(x1) + (1-α)f(x2)')
    
    # 绘制连线
    plt.plot([x1, x2], [f(x1), f(x2)], 'r--', alpha=0.5)
    plt.axvline(x=x_bar, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('凹函数的Jensen不等式')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    
    # 2. 凸函数
    plt.subplot(1, 2, 2)
    x = np.linspace(-3, 3, 100)
    plt.plot(x, g(x), 'b-', label='g(x) = x² (凸函数)')
    
    # 选择两个新的点以便更好地可视化
    x1 = -2.0
    x2 = 1.0
    x_bar = alpha * x1 + (1 - alpha) * x2
    
    # 绘制两个点
    plt.scatter([x1, x2], [g(x1), g(x2)], c='red', s=100, label='数据点')
    
    # 绘制加权平均点
    plt.scatter([x_bar], [g(x_bar)], c='green', s=150, marker='*', label='g(αx1 + (1-α)x2)')
    
    # 绘制函数值的加权平均
    plt.scatter([x_bar], [alpha*g(x1) + (1-alpha)*g(x2)], c='orange', s=150, marker='o', label='αg(x1) + (1-α)g(x2)')
    
    # 绘制连线
    plt.plot([x1, x2], [g(x1), g(x2)], 'r--', alpha=0.5)
    plt.axvline(x=x_bar, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('凸函数的Jensen不等式')
    plt.xlabel('x')
    plt.ylabel('g(x)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../figure/jensen_inequality_convex_concave.png')
    plt.close()

# 概率中的Jensen不等式应用
def jensen_inequality_probability():
    """演示Jensen不等式在概率中的应用"""
    print("\n=== 概率中的Jensen不等式应用 ===")
    
    # 定义离散分布 q(z)
    z_values = np.array([1, 2, 3, 4, 5])
    q_probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    
    # 定义函数 g(z)
    def g(z):
        return z**2
    
    # 计算期望 E_q[g(z)]
    E_g = np.sum(q_probs * g(z_values))
    
    # 计算 log(E_q[g(z)])
    log_E_g = np.log(E_g)
    
    # 计算 E_q[log(g(z))]
    E_log_g = np.sum(q_probs * np.log(g(z_values)))
    
    # 打印结果
    print(f"1. 离散分布 q(z):")
    print(f"   z值: {z_values}")
    print(f"   概率: {q_probs}")
    
    print(f"\n2. 函数 g(z) = z²")
    print(f"   g(z): {g(z_values)}")
    
    print(f"\n3. 计算结果:")
    print(f"   E_q[g(z)] = Σ_z q(z)g(z) = {E_g:.4f}")
    print(f"   log(E_q[g(z)]) = log({E_g:.4f}) = {log_E_g:.4f}")
    print(f"   E_q[log(g(z))] = Σ_z q(z)log(g(z)) = {E_log_g:.4f}")
    
    print(f"\n4. Jensen不等式（凹函数log）:")
    print(f"   log(E_q[g(z)]) ≥ E_q[log(g(z))]")
    print(f"   {log_E_g:.4f} ≥ {E_log_g:.4f} → {log_E_g >= E_log_g}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    
    # 绘制函数 g(z)
    z = np.linspace(0.5, 5.5, 100)
    plt.plot(z, g(z), 'b-', label='g(z) = z²')
    
    # 绘制离散点
    plt.scatter(z_values, g(z_values), c='red', s=100, label='g(z)')
    
    # 绘制期望点
    plt.scatter([np.sum(q_probs * z_values)], [E_g], c='green', s=150, marker='*', label='E_q[g(z)]')
    
    plt.title('Jensen不等式在概率中的应用')
    plt.xlabel('z')
    plt.ylabel('g(z)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../figure/jensen_inequality_probability.png')
    plt.close()

# 主函数
if __name__ == "__main__":
    concave_function_example()
    jensen_inequality_probability()
    
    print("\nJensen不等式示例完成！")
    print("\n要点总结:")
    print("- 对于凹函数，Jensen不等式为 f(Σα_nx_n) ≥ Σα_nf(x_n)")
    print("- 对于凸函数，Jensen不等式为 f(Σα_nx_n) ≤ Σα_nf(x_n)")
    print("- log(·)是凹函数，因此 log(E_q[g(z)]) ≥ E_q[log(g(z))]")
    print("- Jensen不等式在机器学习中有广泛应用，特别是在变分推断中")
