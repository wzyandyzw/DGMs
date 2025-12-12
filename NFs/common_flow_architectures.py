#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalizing Flows - 知识点4：常用 Flow 架构

本文件介绍了常用的Normalizing Flows架构，包括：
1. Planar Flows (Rezende & Mohamed, 2016)
2. 三角 Jacobian 矩阵
3. NICE 模型：加性耦合层
4. NICE 模型：缩放层
5. Real NVP：NICE 的非体积保持扩展
6. 连续自回归模型作为 NFs
7. Masked Autoregressive Flows (MAFs)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, uniform
import torch
import torch.nn as nn

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


def planar_flows():
    """
    Planar Flows (Rezende & Mohamed, 2016)
    实现并可视化Planar Flow变换
    """
    print("=== Planar Flows (Rezende & Mohamed, 2016) ===")
    
    print("Planar Flow定义：可逆变换")
    print("x = f_theta(z) = z + u h(w^T z + b)")
    print("其中 theta = (w, u, b) 是参数，h(·) 是非线性函数")
    
    print("\n雅可比行列式的绝对值：")
    print("|det(∂f_theta(z)/∂z)| = |1 + h'(w^T z + b) w^T u|")
    
    print("\n可逆性条件：")
    print("h'(w^T z + b) w^T u > -1")
    print("常用非线性函数：h(·) = tanh(·)")
    
    # Planar Flow实现
    class PlanarFlow(nn.Module):
        def __init__(self, dim):
            super(PlanarFlow, self).__init__()
            self.w = nn.Parameter(torch.randn(1, dim))
            self.u = nn.Parameter(torch.randn(1, dim))
            self.b = nn.Parameter(torch.randn(1))
            
            # 确保可逆性条件
            self._u_hat()
        
        def _u_hat(self):
            """确保可逆性条件的参数变换"""
            wtu = torch.matmul(self.w, self.u.t())
            m_wtu = -1 + torch.log(1 + torch.exp(wtu))
            self.u.data = self.u + (m_wtu - wtu) * self.w / torch.norm(self.w)**2
        
        def forward(self, z):
            """正向变换 z → x"""
            z = z.view(-1, z.size(-1))
            activation = torch.tanh(torch.matmul(z, self.w.t()) + self.b)
            x = z + activation * self.u
            
            # 计算雅可比行列式
            psi = (1 - activation**2) * self.w
            log_det_jacobian = torch.log(torch.abs(1 + torch.matmul(psi, self.u.t())))
            
            return x, log_det_jacobian
    
    # 可视化Planar Flow
    def visualize_planar_flow():
        # 创建平面流模型
        planar_flow = PlanarFlow(2)
        
        # 设置固定参数以便可视化
        with torch.no_grad():
            planar_flow.w.data = torch.tensor([[1.0, 0.5]])
            planar_flow.u.data = torch.tensor([[0.5, 1.0]])
            planar_flow.b.data = torch.tensor([0.0])
        
        # 生成初始样本
        z = torch.randn(10000, 2)
        
        # 应用平面流
        x, log_det_jac = planar_flow(z)
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 初始分布
        ax1.scatter(z[:, 0], z[:, 1], alpha=0.3, s=1, color='blue')
        ax1.set_title('初始分布：Z ~ N(0, I)')
        ax1.set_xlabel('z1')
        ax1.set_ylabel('z2')
        ax1.set_aspect('equal')
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        
        # 变换后的分布
        ax2.scatter(x[:, 0], x[:, 1], alpha=0.3, s=1, color='purple')
        ax2.set_title('变换后分布：X = f(Z)')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_aspect('equal')
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 3)
        
        plt.tight_layout()
        plt.savefig('./figure/planar_flow.png')
        plt.close()
    
    visualize_planar_flow()
    print("已生成可视化：Planar Flow 变换")


def triangular_jacobian():
    """
    三角 Jacobian 矩阵
    介绍三角Jacobian矩阵的优势和应用
    """
    print("\n=== 三角 Jacobian 矩阵 ===")
    
    print("三角Jacobian矩阵的优势：")
    print("- 行列式计算高效（对角线元素乘积，O(n)时间）")
    print("- 逆矩阵计算高效（O(n²)时间）")
    
    print("\n下三角Jacobian矩阵：")
    print("如果 x_i = f_i(z) 仅依赖于 z_{≤i}，则Jacobian矩阵是下三角的")
    print("J_ij = ∂f_i/∂z_j = 0 当 j > i")
    
    print("\n上三角Jacobian矩阵：")
    print("如果 x_i = f_i(z) 仅依赖于 z_{≥i}，则Jacobian矩阵是上三角的")
    print("J_ij = ∂f_i/∂z_j = 0 当 j < i")
    
    # 可视化三角矩阵结构
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    n = 10
    
    # 下三角矩阵
    lower_tri = np.tril(np.random.randn(n, n))
    ax1.imshow(np.abs(lower_tri) > 0.1, cmap='binary')
    ax1.set_title('下三角Jacobian矩阵')
    ax1.set_xlabel('z_j')
    ax1.set_ylabel('x_i')
    
    # 上三角矩阵
    upper_tri = np.triu(np.random.randn(n, n))
    ax2.imshow(np.abs(upper_tri) > 0.1, cmap='binary')
    ax2.set_title('上三角Jacobian矩阵')
    ax2.set_xlabel('z_j')
    ax2.set_ylabel('x_i')
    
    plt.tight_layout()
    plt.savefig('./figure/triangular_jacobian.png')
    plt.close()
    
    print("已生成可视化：三角Jacobian矩阵结构")


def nice_model_additive_coupling():
    """
    NICE 模型：加性耦合层
    实现并可视化NICE模型的加性耦合层
    """
    print("\n=== NICE 模型：加性耦合层 ===")
    
    print("NICE模型将变量z划分为两个不相交的子集：")
    print("z = [z_{1:d}, z_{(d+1):n}]")
    
    print("\n正向映射 z → x：")
    print("x_{1:d} = z_{1:d} （恒等变换）")
    print("x_{(d+1):n} = z_{(d+1):n} + m_theta(z_{1:d}) （加性变换）")
    
    print("\n逆映射 x → z：")
    print("z_{1:d} = x_{1:d} （恒等变换）")
    print("z_{(d+1):n} = x_{(d+1):n} - m_theta(x_{1:d}) （减性变换）")
    
    print("\n雅可比矩阵：")
    print("J = [[I_d, 0], [∂x_{(d+1):n}/∂z_{1:d}, I_{n-d}]]")
    print("行列式：det(J) = 1 （体积保持变换）")
    
    # 简单的加性耦合层实现
    class AdditiveCouplingLayer(nn.Module):
        def __init__(self, dim, split_dim):
            super(AdditiveCouplingLayer, self).__init__()
            self.split_dim = split_dim
            # 简单的多层感知机作为变换函数
            self.mlp = nn.Sequential(
                nn.Linear(split_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, dim - split_dim)
            )
        
        def forward(self, z):
            """正向变换 z → x"""
            z1, z2 = z[:, :self.split_dim], z[:, self.split_dim:]
            m = self.mlp(z1)
            x1, x2 = z1, z2 + m
            x = torch.cat([x1, x2], dim=1)
            log_det_jacobian = torch.zeros(z.size(0), 1)
            return x, log_det_jacobian
        
        def inverse(self, x):
            """逆变换 x → z"""
            x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
            m = self.mlp(x1)
            z1, z2 = x1, x2 - m
            z = torch.cat([z1, z2], dim=1)
            log_det_jacobian = torch.zeros(x.size(0), 1)
            return z, log_det_jacobian
    
    # 可视化加性耦合层
    def visualize_additive_coupling():
        # 创建加性耦合层
        dim = 2
        split_dim = 1
        coupling_layer = AdditiveCouplingLayer(dim, split_dim)
        
        # 生成初始样本
        z = torch.randn(10000, dim)
        
        # 应用加性耦合层
        x, _ = coupling_layer(z)
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 初始分布
        ax1.scatter(z[:, 0], z[:, 1], alpha=0.3, s=1, color='blue')
        ax1.set_title('初始分布：Z ~ N(0, I)')
        ax1.set_xlabel('z1')
        ax1.set_ylabel('z2')
        ax1.set_aspect('equal')
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        
        # 变换后的分布
        ax2.scatter(x[:, 0], x[:, 1], alpha=0.3, s=1, color='purple')
        ax2.set_title('加性耦合层变换后')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_aspect('equal')
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 3)
        
        plt.tight_layout()
        plt.savefig('./figure/additive_coupling.png')
        plt.close()
    
    visualize_additive_coupling()
    print("已生成可视化：加性耦合层变换")


def nice_model_scaling():
    """
    NICE 模型：缩放层
    实现并可视化NICE模型的缩放层
    """
    print("\n=== NICE 模型：缩放层 ===")
    
    print("NICE模型的最后一层应用缩放变换：")
    
    print("\n正向映射 z → x：")
    print("x_i = s_i z_i，其中 s_i > 0 是缩放因子")
    
    print("\n逆映射 x → z：")
    print("z_i = x_i / s_i")
    
    print("\n雅可比矩阵：")
    print("J = Diag(s) （对角矩阵）")
    print("行列式：det(J) = ∏_{i=1}^n s_i")
    
    # 缩放层实现
    class ScalingLayer(nn.Module):
        def __init__(self, dim):
            super(ScalingLayer, self).__init__()
            self.log_s = nn.Parameter(torch.zeros(dim))
        
        def forward(self, z):
            """正向变换 z → x"""
            x = z * torch.exp(self.log_s)
            log_det_jacobian = torch.sum(self.log_s) * torch.ones(z.size(0), 1)
            return x, log_det_jacobian
        
        def inverse(self, x):
            """逆变换 x → z"""
            z = x * torch.exp(-self.log_s)
            log_det_jacobian = -torch.sum(self.log_s) * torch.ones(x.size(0), 1)
            return z, log_det_jacobian
    
    # 可视化缩放层
    def visualize_scaling():
        # 创建缩放层
        dim = 2
        scaling_layer = ScalingLayer(dim)
        
        # 设置固定缩放因子
        with torch.no_grad():
            scaling_layer.log_s.data = torch.tensor([0.5, -0.5])  # 沿x1放大，沿x2缩小
        
        # 生成初始样本
        z = torch.randn(10000, dim)
        
        # 应用缩放层
        x, _ = scaling_layer(z)
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 初始分布
        ax1.scatter(z[:, 0], z[:, 1], alpha=0.3, s=1, color='blue')
        ax1.set_title('初始分布：Z ~ N(0, I)')
        ax1.set_xlabel('z1')
        ax1.set_ylabel('z2')
        ax1.set_aspect('equal')
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        
        # 变换后的分布
        ax2.scatter(x[:, 0], x[:, 1], alpha=0.3, s=1, color='purple')
        ax2.set_title('缩放层变换后')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_aspect('equal')
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 3)
        
        plt.tight_layout()
        plt.savefig('./figure/scaling_layer.png')
        plt.close()
    
    visualize_scaling()
    print("已生成可视化：缩放层变换")


def real_nvp():
    """
    Real NVP：NICE 的非体积保持扩展
    实现并可视化Real NVP模型
    """
    print("\n=== Real NVP：NICE 的非体积保持扩展 ===")
    
    print("Real NVP是NICE模型的扩展，引入了缩放因子：")
    
    print("\n正向映射 z → x：")
    print("x_{1:d} = z_{1:d} （恒等变换）")
    print("x_{(d+1):n} = z_{(d+1):n} ⊙ exp(α_theta(z_{1:d})) + μ_theta(z_{1:d}) （仿射变换）")
    
    print("\n逆映射 x → z：")
    print("z_{1:d} = x_{1:d} （恒等变换）")
    print("z_{(d+1):n} = (x_{(d+1):n} - μ_theta(x_{1:d})) ⊙ exp(-α_theta(x_{1:d}))")
    
    print("\n雅可比行列式：")
    print("det(J) = exp(Σ_{i=d+1}^n (α_theta(z_{1:d}))_i)")
    
    # Real NVP耦合层实现
    class RealNVPCouplingLayer(nn.Module):
        def __init__(self, dim, split_dim):
            super(RealNVPCouplingLayer, self).__init__()
            self.split_dim = split_dim
            # 缩放和偏移的变换函数
            self.net_s = nn.Sequential(
                nn.Linear(split_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, dim - split_dim),
                nn.Tanh()  # 使用tanh限制缩放范围
            )
            self.net_t = nn.Sequential(
                nn.Linear(split_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, dim - split_dim)
            )
        
        def forward(self, z):
            """正向变换 z → x"""
            z1, z2 = z[:, :self.split_dim], z[:, self.split_dim:]
            s = self.net_s(z1)
            t = self.net_t(z1)
            x1, x2 = z1, z2 * torch.exp(s) + t
            x = torch.cat([x1, x2], dim=1)
            log_det_jacobian = torch.sum(s, dim=1, keepdim=True)
            return x, log_det_jacobian
        
        def inverse(self, x):
            """逆变换 x → z"""
            x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
            s = self.net_s(x1)
            t = self.net_t(x1)
            z1, z2 = x1, (x2 - t) * torch.exp(-s)
            z = torch.cat([z1, z2], dim=1)
            log_det_jacobian = -torch.sum(s, dim=1, keepdim=True)
            return z, log_det_jacobian
    
    # 可视化Real NVP
    def visualize_real_nvp():
        # 创建Real NVP耦合层
        dim = 2
        split_dim = 1
        real_nvp_layer = RealNVPCouplingLayer(dim, split_dim)
        
        # 生成初始样本
        z = torch.randn(10000, dim)
        
        # 应用Real NVP层
        x, _ = real_nvp_layer(z)
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 初始分布
        ax1.scatter(z[:, 0], z[:, 1], alpha=0.3, s=1, color='blue')
        ax1.set_title('初始分布：Z ~ N(0, I)')
        ax1.set_xlabel('z1')
        ax1.set_ylabel('z2')
        ax1.set_aspect('equal')
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        
        # 变换后的分布
        ax2.scatter(x[:, 0], x[:, 1], alpha=0.3, s=1, color='purple')
        ax2.set_title('Real NVP变换后')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_aspect('equal')
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-3, 3)
        
        plt.tight_layout()
        plt.savefig('./figure/real_nvp.png')
        plt.close()
    
    visualize_real_nvp()
    print("已生成可视化：Real NVP变换")


def autoregressive_flows():
    """
    连续自回归模型作为 NFs
    介绍自回归模型与Normalizing Flows的关系
    """
    print("\n=== 连续自回归模型作为 NFs ===")
    
    print("高斯自回归模型：")
    print("p(x) = ∏_{i=1}^n p(x_i | x_{<<i})")
    print("其中 p(x_i | x_{<<i}) = N(μ_i(x_{<<i}), exp(α_i(x_{<<i}))²)")
    
    print("\n采样过程：")
    print("1. 采样 z_i ~ N(0, 1) （i = 1, ..., n）")
    print("2. x_1 = μ_1 + exp(α_1) z_1")
    print("3. 对于 i > 1: x_i = μ_i(x_{<<i}) + exp(α_i(x_{<<i})) z_i")
    
    print("\nFlow解释：")
    print("通过可逆变换将标准高斯样本(z_1, ..., z_n)转换为模型样本(x_1, ..., x_n)")
    
    # 简单的自回归流实现
    def simple_autoregressive_flow():
        # 生成自回归样本
        n_samples = 10000
        n_dims = 2
        
        # 简单的自回归参数
        mu_1 = 0.0
        alpha_1 = 0.5
        
        def mu_2(x1):
            return 2 * x1
        
        def alpha_2(x1):
            return np.log(1 + 0.5 * x1**2)
        
        # 采样
        z = np.random.normal(0, 1, (n_samples, n_dims))
        x = np.zeros_like(z)
        
        x[:, 0] = mu_1 + np.exp(alpha_1) * z[:, 0]
        x[:, 1] = mu_2(x[:, 0]) + np.exp(alpha_2(x[:, 0])) * z[:, 1]
        
        return z, x
    
    # 可视化自回归流
    z, x = simple_autoregressive_flow()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 初始分布
    ax1.scatter(z[:, 0], z[:, 1], alpha=0.3, s=1, color='blue')
    ax1.set_title('初始分布：Z ~ N(0, I)')
    ax1.set_xlabel('z1')
    ax1.set_ylabel('z2')
    ax1.set_aspect('equal')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    
    # 变换后的分布
    ax2.scatter(x[:, 0], x[:, 1], alpha=0.3, s=1, color='purple')
    ax2.set_title('自回归流变换后')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_aspect('equal')
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    
    plt.tight_layout()
    plt.savefig('./figure/autoregressive_flow.png')
    plt.close()
    
    print("已生成可视化：自回归流变换")


def masked_autoregressive_flows():
    """
    Masked Autoregressive Flows (MAFs)
    介绍MAFs模型
    """
    print("\n=== Masked Autoregressive Flows (MAFs) ===")
    
    print("MAFs是自回归流的一种实现，使用掩码权重确保自回归性质：")
    
    print("\n正向映射 z → x：")
    print("x_i = z_i * exp(α_i) + μ_i，其中 α_i 和 μ_i 仅依赖于 x_{<<i}")
    
    print("\n逆映射 x → z：")
    print("z_i = (x_i - μ_i) * exp(-α_i)")
    
    print("\n特点：")
    print("- 似然评估简单且可并行化")
    print("- 采样是顺序的且缓慢的（O(n)计算时间）")
    print("- 反转正向和反向映射会产生逆自回归流（IAFs）")
    
    # 可视化MAFs的掩码矩阵
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n = 10
    # 创建自回归掩码
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    ax.imshow(mask, cmap='binary')
    ax.set_title('MAFs的自回归掩码')
    ax.set_xlabel('输入特征')
    ax.set_ylabel('输出特征')
    
    plt.tight_layout()
    plt.savefig('./figure/maf_mask.png')
    plt.close()
    
    print("已生成可视化：MAFs的自回归掩码")
    print("注意：掩码为True的位置表示权重为0，确保自回归性质")


def main():
    """
    主函数，运行所有示例
    """
    print("Normalizing Flows - 知识点4：常用 Flow 架构")
    print("=" * 60)
    
    planar_flows()
    triangular_jacobian()
    nice_model_additive_coupling()
    nice_model_scaling()
    real_nvp()
    autoregressive_flows()
    masked_autoregressive_flows()
    
    print("\n" + "=" * 60)
    print("知识点4：常用 Flow 架构 示例演示完成")


if __name__ == "__main__":
    main()
