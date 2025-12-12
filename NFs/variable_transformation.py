#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalizing Flows - 知识点2：变量变换公式

本文件介绍了变量变换公式的基本概念和应用，包括：
1. 1D变量变换公式
2. 几何视角：行列式与体积
3. 广义变量变换公式
4. 二维示例
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, norm
from matplotlib.patches import Polygon

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


def one_dimensional_variable_transformation():
    """
    1D变量变换公式
    展示一维情况下的变量变换
    """
    print("=== 1D变量变换公式 ===")
    
    # 示例1：线性变换 X = 4Z，Z ~ U[0, 2]
    print("\n示例1：线性变换")
    print("考虑均匀随机变量 Z ~ U[0, 2]，其密度为 p_Z(z) = 1/2")
    print("令 X = 4Z，求 p_X(4)")
    print("解析解：X 在 [0, 8] 上均匀分布，因此 p_X(4) = 1/8")
    
    # 变量变换公式计算
    z = 1  # 对应x=4的z值
    h = lambda x: x / 4  # 逆变换
    h_prime = 1/4  # 逆变换导数
    p_z = 1/2  # Z在z=1处的密度
    p_x = p_z * h_prime
    print(f"使用变量变换公式计算：p_X(4) = p_Z(1) * |h'(4)| = {p_z} * {h_prime} = {p_x}")
    
    # 示例2：非线性变换 X = exp(Z)，Z ~ U[0, 2]
    print("\n示例2：非线性变换")
    print("考虑均匀随机变量 Z ~ U[0, 2]，令 X = exp(Z)")
    print("求 p_X(x)")
    
    h = lambda x: np.log(x)  # 逆变换
    h_prime = lambda x: 1/x  # 逆变换导数
    p_z = lambda z: 1/2 if 0 <= z <= 2 else 0
    p_x = lambda x: p_z(np.log(x)) * (1/x) if 1 <= x <= np.exp(2) else 0
    
    print(f"p_X(x) = p_Z(log(x)) * (1/x)，其中 x ∈ [1, exp(2)]")
    print(f"p_X(2) = {p_x(2):.4f}")
    print(f"p_X(4) = {p_x(4):.4f}")
    
    # 可视化非线性变换
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Z的分布
    z_samples = uniform.rvs(0, 2, size=10000)
    ax1.hist(z_samples, bins=50, density=True, alpha=0.7, color='blue')
    ax1.set_title('Z ~ U[0, 2]')
    ax1.set_xlabel('z')
    ax1.set_ylabel('概率密度')
    
    # X = exp(Z)的分布
    x_samples = np.exp(z_samples)
    ax2.hist(x_samples, bins=50, density=True, alpha=0.7, color='purple')
    ax2.set_title('X = exp(Z)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('概率密度')
    
    plt.tight_layout()
    plt.savefig('./figure/1d_transformation.png')
    plt.close()
    
    print("已生成可视化：1D变量变换")


def geometric_perspective():
    """
    几何视角：行列式与体积
    从几何角度解释变量变换中的行列式作用
    """
    print("\n=== 几何视角：行列式与体积 ===")
    
    # 2D线性变换示例
    print("2D线性变换示例：")
    print("考虑单位正方形 [0,1]^2 上的均匀分布")
    print("通过矩阵 A 进行线性变换：X = A Z")
    
    # 定义变换矩阵
    A = np.array([[2, 1], [1, 2]])
    print(f"变换矩阵 A = \n{A}")
    
    # 计算行列式
    det_A = np.linalg.det(A)
    print(f"行列式 det(A) = {det_A}")
    print(f"体积变化因子 = |det(A)| = {abs(det_A)}")
    
    # 可视化变换
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 单位正方形
    square = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]], closed=True, alpha=0.5, color='blue')
    ax1.add_patch(square)
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(-0.5, 3.5)
    ax1.set_title('单位正方形 [0,1]^2')
    ax1.set_aspect('equal')
    
    # 变换后的平行四边形
    transformed_points = np.dot(A, np.array([[0, 1, 1, 0], [0, 0, 1, 1]]))
    transformed_square = Polygon(transformed_points.T, closed=True, alpha=0.5, color='purple')
    ax2.add_patch(transformed_square)
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(-0.5, 3.5)
    ax2.set_title(f'变换后的平行四边形 (det(A)={det_A:.2f})')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('./figure/geometric_transformation.png')
    plt.close()
    
    print("已生成可视化：几何视角下的变换")


def general_variable_transformation():
    """
    广义变量变换公式
    介绍多维情况下的变量变换公式
    """
    print("\n=== 广义变量变换公式 ===")
    
    print("对于可逆变换 f: R^n → R^n，使得 X = f(Z) 和 Z = f⁻¹(X)")
    print("则有：")
    print("p_X(x) = p_Z(f⁻¹(x)) * |det(∂f⁻¹(x)/∂x)|")
    print("或者等价于：")
    print("p_X(x) = p_Z(z) * |det(∂f(z)/∂z)|⁻¹")
    
    print("\n其中 det(∂f(z)/∂z) 是变换的雅可比行列式")
    print("雅可比矩阵 J 的元素为 J_ij = ∂f_i/∂z_j")
    
    # 3D高斯变换示例
    print("\n3D示例：高斯变换")
    mu_z = np.zeros(3)
    sigma_z = np.eye(3)
    
    # 线性变换
    A = np.array([[1, 0.5, 0.2], [0.5, 2, 0.3], [0.2, 0.3, 1.5]])
    mu_x = np.array([1, 2, 3])
    
    # 变换后的分布参数
    sigma_x = A @ sigma_z @ A.T
    
    print(f"原始高斯分布：Z ~ N(0, I_3)")
    print(f"线性变换：X = A Z + μ_x")
    print(f"变换后的分布：X ~ N(μ_x, A A^T)")
    print(f"雅可比行列式：det(A) = {np.linalg.det(A):.4f}")
    print(f"体积变化因子：|det(A)| = {abs(np.linalg.det(A)):.4f}")


def two_dimensional_example():
    """
    二维示例
    展示二维情况下的变量变换
    """
    print("\n=== 二维示例 ===")
    
    # 定义变换：极坐标到直角坐标
    print("示例：极坐标到直角坐标变换")
    print("考虑极坐标 (r, θ)，变换为直角坐标 (x, y)")
    print("x = r cosθ")
    print("y = r sinθ")
    
    # 雅可比矩阵
    print("\n雅可比矩阵 J：")
    print("[∂x/∂r  ∂x/∂θ]")
    print("[∂y/∂r  ∂y/∂θ]")
    print("\n=\n")
    print("[cosθ  -r sinθ]")
    print("[sinθ   r cosθ]")
    
    # 雅可比行列式
    # 注意：cosθ和sinθ是角度θ的余弦和正弦值
    # 对于任意θ，cos²θ + sin²θ = 1，因此det(J) = r
    det_J = r  # 简化后的结果
    print(f"雅可比行列式：det(J) = r")
    
    # 示例：径向对称分布
    print("\n示例：径向对称分布")
    print("考虑 r ~ Exp(1)，θ ~ U[0, 2π]")
    print("求 p_X(x, y)")
    
    # 逆变换：直角坐标到极坐标
    r = lambda x, y: np.sqrt(x**2 + y**2)
    theta = lambda x, y: np.arctan2(y, x)
    
    # 原始分布密度
    p_r = lambda r: np.exp(-r) if r >= 0 else 0
    p_theta = lambda theta: 1/(2*np.pi) if 0 <= theta < 2*np.pi else 0
    p_z = lambda r, theta: p_r(r) * p_theta(theta)
    
    # 变换后的密度
    p_x = lambda x, y: p_z(r(x,y), theta(x,y)) * (1/r(x,y)) if r(x,y) > 0 else 0
    
    # 可视化变换后的分布
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 生成极坐标样本
    n_samples = 10000
    r_samples = np.random.exponential(1, n_samples)
    theta_samples = np.random.uniform(0, 2*np.pi, n_samples)
    
    # 转换为直角坐标
    x_samples = r_samples * np.cos(theta_samples)
    y_samples = r_samples * np.sin(theta_samples)
    
    # 绘制散点图
    ax.scatter(x_samples, y_samples, alpha=0.5, s=1, color='purple')
    ax.set_title('极坐标到直角坐标变换后的分布')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    
    plt.tight_layout()
    plt.savefig('./figure/2d_transformation.png')
    plt.close()
    
    print("已生成可视化：二维变换后的分布")
    print(f"示例：p_X(1, 0) = {p_x(1, 0):.4f}")
    print(f"示例：p_X(0, 1) = {p_x(0, 1):.4f}")
    print(f"示例：p_X(1, 1) = {p_x(1, 1):.4f}")


def main():
    """
    主函数，运行所有示例
    """
    print("Normalizing Flows - 知识点2：变量变换公式")
    print("=" * 60)
    
    one_dimensional_variable_transformation()
    geometric_perspective()
    general_variable_transformation()
    two_dimensional_example()
    
    print("\n" + "=" * 60)
    print("知识点2：变量变换公式 示例演示完成")


if __name__ == "__main__":
    main()
