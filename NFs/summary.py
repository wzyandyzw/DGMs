#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalizing Flows - 知识点5：总结

本文件总结了Normalizing Flows的核心概念和特点，包括：
1. Normalizing Flows的核心思想
2. 主要优点
3. 主要缺点
4. 应用场景
5. 与其他生成模型的比较
6. 综合示例：使用Real NVP生成数据
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from scipy.stats import norm, uniform

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


def core_idea():
    """
    Normalizing Flows的核心思想
    总结Normalizing Flows的核心概念
    """
    print("=== Normalizing Flows的核心思想 ===")
    
    print("Normalizing Flows的核心思想是通过一系列可逆变换将简单分布转换为复杂分布。")
    print("\n具体来说：")
    print("1. 从简单的先验分布（如标准高斯分布）开始")
    print("2. 应用一系列可逆的、参数化的变换")
    print("3. 变换后的分布可以匹配复杂的数据分布")
    print("4. 利用变量变换公式精确计算似然函数")
    
    print("\n数学表示：")
    print("x = f_theta(z), z ~ p_Z(z)")
    print("p_X(x) = p_Z(f_theta^{-1}(x)) * |det(∂f_theta^{-1}(x)/∂x)|")
    
    # 可视化核心思想
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 简单先验
    z = np.random.normal(0, 1, (10000, 2))
    axes[0].scatter(z[:, 0], z[:, 1], alpha=0.3, s=1, color='blue')
    axes[0].set_title('1. 简单先验分布')
    axes[0].set_xlabel('z1')
    axes[0].set_ylabel('z2')
    axes[0].set_aspect('equal')
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    
    # 中间变换
    # 应用一个简单的可逆变换
    def intermediate_transform(z):
        x1 = z[:, 0] + 0.5 * np.sin(2 * z[:, 1])
        x2 = z[:, 1] + 0.5 * np.cos(2 * z[:, 0])
        return np.column_stack([x1, x2])
    
    x_intermediate = intermediate_transform(z)
    axes[1].scatter(x_intermediate[:, 0], x_intermediate[:, 1], alpha=0.3, s=1, color='green')
    axes[1].set_title('2. 中间变换')
    axes[1].set_xlabel('x1')
    axes[1].set_ylabel('x2')
    axes[1].set_aspect('equal')
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    
    # 最终分布
    # 再应用一个变换
    def final_transform(x):
        y1 = x[:, 0] * np.exp(0.5 * x[:, 1])
        y2 = x[:, 1] * np.exp(-0.5 * x[:, 0])
        return np.column_stack([y1, y2])
    
    y_final = final_transform(x_intermediate)
    axes[2].scatter(y_final[:, 0], y_final[:, 1], alpha=0.3, s=1, color='purple')
    axes[2].set_title('3. 最终复杂分布')
    axes[2].set_xlabel('y1')
    axes[2].set_ylabel('y2')
    axes[2].set_aspect('equal')
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-3, 3)
    
    plt.tight_layout()
    plt.savefig('./figure/nf_core_idea.png')
    plt.close()
    
    print("已生成可视化：Normalizing Flows核心思想")


def key_advantages():
    """
    主要优点
    总结Normalizing Flows的主要优点
    """
    print("\n=== 主要优点 ===")
    
    print("1. 精确的似然评估：")
    print("   - 利用变量变换公式可以精确计算数据的似然")
    print("   - 无需近似，这是与VAE等模型相比的主要优势")
    
    print("\n2. 可逆性：")
    print("   - 变换是可逆的，可以在数据空间和潜在空间之间自由转换")
    print("   - 无需单独的推理网络")
    
    print("\n3. 灵活的分布建模：")
    print("   - 可以建模复杂的多模态分布")
    print("   - 通过堆叠多个变换可以增加表达能力")
    
    print("\n4. 理论基础坚实：")
    print("   - 基于严格的概率理论")
    print("   - 目标函数明确（最大似然估计）")
    
    print("\n5. 易于与其他模型结合：")
    print("   - 可以作为其他模型的组件")
    print("   - 例如，与VAE结合形成NF-VAE，提高表达能力")


def key_disadvantages():
    """
    主要缺点
    总结Normalizing Flows的主要缺点
    """
    print("\n=== 主要缺点 ===")
    
    print("1. 高计算成本：")
    print("   - 可逆变换的设计通常需要O(n²)或更高的计算复杂度")
    print("   - 雅可比行列式的计算可能很昂贵")
    
    print("\n2. 维度限制：")
    print("   - 输入和输出维度必须相同")
    print("   - 对于高维数据（如图像），计算成本显著增加")
    
    print("\n3. 设计复杂：")
    print("   - 需要精心设计可逆变换")
    print("   - 确保变换的可逆性和雅可比行列式的易计算性")
    
    print("\n4. 训练不稳定：")
    print("   - 某些Flow架构可能面临训练不稳定的问题")
    print("   - 需要仔细调整超参数")
    
    print("\n5. 采样效率：")
    print("   - 某些Flow架构（如MAFs）的采样过程是顺序的")
    print("   - 采样速度比GAN等生成模型慢")


def application_scenarios():
    """
    应用场景
    总结Normalizing Flows的主要应用场景
    """
    print("\n=== 应用场景 ===")
    
    print("1. 密度估计：")
    print("   - 精确的密度估计任务")
    print("   - 异常检测和离群值识别")
    
    print("\n2. 生成建模：")
    print("   - 生成高质量的图像、音频等数据")
    print("   - 数据增强和数据补全")
    
    print("\n3. 贝叶斯推理：")
    print("   - 作为贝叶斯模型的后验近似")
    print("   - 变分推断中的灵活近似分布")
    
    print("\n4. 强化学习：")
    print("   - 连续动作空间的策略表示")
    print("   - 状态和动作的概率建模")
    
    print("\n5. 金融建模：")
    print("   - 资产价格分布的建模")
    print("   - 风险评估和投资组合优化")
    
    print("\n6. 物理模拟：")
    print("   - 分子动力学模拟")
    print("   - 量子力学系统建模")


def comparison_with_other_models():
    """
    与其他生成模型的比较
    将Normalizing Flows与VAE、GAN等生成模型进行比较
    """
    print("\n=== 与其他生成模型的比较 ===")
    
    print("| 模型 | 似然计算 | 采样效率 | 训练稳定性 | 表达能力 |")
    print("|------|----------|----------|------------|----------|")
    print("| Normalizing Flows | 精确 | 中等/低 | 中等 | 高 |")
    print("| VAE | 下界估计 | 高 | 高 | 中等 |")
    print("| GAN | 不可行 | 高 | 低 | 高 |")
    print("| Autoregressive Models | 精确 | 低 | 高 | 高 |")
    
    print("\n详细比较：")
    print("\n1. 与VAE比较：")
    print("   - NF提供精确似然，VAE仅提供下界")
    print("   - VAE可以处理维度不匹配的情况")
    print("   - NF的计算成本通常高于VAE")
    
    print("\n2. 与GAN比较：")
    print("   - NF可以计算似然，GAN不能")
    print("   - GAN的采样效率通常高于NF")
    print("   - NF的训练通常比GAN更稳定")
    
    print("\n3. 与自回归模型比较：")
    print("   - 两者都可以精确计算似然")
    print("   - NF的采样可能更高效（取决于架构）")
    print("   - 自回归模型的并行计算能力有限")


def real_nvp_implementation():
    """
    Real NVP实现
    简单实现Real NVP模型，用于综合示例
    """
    class RealNVPCouplingLayer(nn.Module):
        def __init__(self, dim, split_dim):
            super(RealNVPCouplingLayer, self).__init__()
            self.split_dim = split_dim
            # 缩放和偏移的变换函数
            self.net_s = nn.Sequential(
                nn.Linear(split_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, dim - split_dim),
                nn.Tanh()
            )
            self.net_t = nn.Sequential(
                nn.Linear(split_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, dim - split_dim)
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
    
    class RealNVP(nn.Module):
        def __init__(self, dim, num_layers=5, split_dim=None):
            super(RealNVP, self).__init__()
            self.dim = dim
            self.split_dim = split_dim if split_dim is not None else dim // 2
            
            # 创建多个耦合层，交替分割方式
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                self.layers.append(RealNVPCouplingLayer(dim, self.split_dim))
                # 交换分割方式
                self.split_dim = dim - self.split_dim
        
        def forward(self, z):
            """正向变换 z → x"""
            x = z
            log_det_jacobian = 0
            for layer in self.layers:
                x, ldj = layer(x)
                log_det_jacobian += ldj
            return x, log_det_jacobian
        
        def inverse(self, x):
            """逆变换 x → z"""
            z = x
            log_det_jacobian = 0
            for layer in reversed(self.layers):
                z, ldj = layer.inverse(z)
                log_det_jacobian += ldj
            return z, log_det_jacobian
    
    return RealNVP


def comprehensive_example():
    """
    综合示例：使用Real NVP生成数据
    使用Real NVP模型生成2D月牙形数据
    """
    print("\n=== 综合示例：使用Real NVP生成数据 ===")
    
    # 生成月牙形数据
    def generate_moon_data(n_samples=10000, noise=0.1):
        """生成2D月牙形数据"""
        # 上半月牙
        theta1 = np.random.uniform(0, np.pi, n_samples // 2)
        r1 = 1.0
        x1 = r1 * np.cos(theta1)
        y1 = r1 * np.sin(theta1)
        
        # 下半月牙
        theta2 = np.random.uniform(0, np.pi, n_samples // 2)
        r2 = 2.0
        x2 = r2 * np.cos(theta2) + r2
        y2 = r2 * np.sin(theta2) - 1.0
        
        # 合并数据
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        
        # 添加噪声
        x += np.random.normal(0, noise, x.shape)
        y += np.random.normal(0, noise, y.shape)
        
        return np.column_stack([x, y])
    
    # 生成训练数据
    data = generate_moon_data(n_samples=10000, noise=0.05)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    print(f"生成月牙形数据：{data.shape[0]}个样本")
    
    # 创建并训练Real NVP模型
    print("\n训练Real NVP模型...")
    
    # 设置超参数
    dim = 2
    num_layers = 5
    batch_size = 128
    num_epochs = 50
    learning_rate = 1e-3
    
    # 创建模型
    RealNVP = real_nvp_implementation()
    model = RealNVP(dim, num_layers)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        # 打乱数据
        permutation = torch.randperm(data_tensor.size(0))
        
        total_loss = 0
        for i in range(0, data_tensor.size(0), batch_size):
            # 获取批次数据
            indices = permutation[i:i+batch_size]
            batch_data = data_tensor[indices]
            
            # 前向传播
            z, log_det_jacobian = model.inverse(batch_data)
            
            # 计算负对数似然损失
            # p_X(x) = p_Z(z) * |det(∂f⁻¹(x)/∂x)|
            # log p_X(x) = log p_Z(z) + log_det_jacobian
            log_pz = torch.sum(norm.logpdf(z, 0, 1), dim=1, keepdim=True)
            log_px = log_pz + log_det_jacobian
            loss = -torch.mean(log_px)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_data.size(0)
        
        # 计算平均损失
        avg_loss = total_loss / data_tensor.size(0)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # 生成样本
    print("\n生成样本...")
    model.eval()
    with torch.no_grad():
        # 从先验分布采样
        z_samples = torch.randn(10000, dim)
        # 正向变换生成样本
        x_samples, _ = model(z_samples)
    
    # 转换为numpy数组
    x_samples_np = x_samples.numpy()
    
    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 原始数据
    ax1.scatter(data[:, 0], data[:, 1], alpha=0.3, s=1, color='blue')
    ax1.set_title('原始月牙形数据')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_aspect('equal')
    ax1.set_xlim(-2, 4)
    ax1.set_ylim(-2, 2)
    
    # 生成样本
    ax2.scatter(x_samples_np[:, 0], x_samples_np[:, 1], alpha=0.3, s=1, color='purple')
    ax2.set_title('Real NVP生成样本')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_aspect('equal')
    ax2.set_xlim(-2, 4)
    ax2.set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.savefig('./figure/real_nvp_moon.png')
    plt.close()
    
    print("已生成可视化：Real NVP生成月牙形数据")
    print("\n示例总结：")
    print("1. 成功使用Real NVP模型学习了月牙形数据的分布")
    print("2. 模型能够生成与原始数据分布相似的样本")
    print("3. 展示了Normalizing Flows在密度估计和生成建模中的应用")


def future_directions():
    """
    未来发展方向
    简要介绍Normalizing Flows的未来发展方向
    """
    print("\n=== 未来发展方向 ===")
    
    print("1. 更高效的架构：")
    print("   - 设计计算成本更低的可逆变换")
    print("   - 提高高维数据的处理能力")
    
    print("\n2. 与深度学习结合：")
    print("   - 结合Transformer等先进的深度学习架构")
    print("   - 探索更强大的特征表示")
    
    print("\n3. 应用扩展：")
    print("   - 在更多领域应用，如计算机视觉、自然语言处理")
    print("   - 与其他技术结合，如强化学习、贝叶斯推理")
    
    print("\n4. 理论研究：")
    print("   - 深入理解Flow模型的表达能力")
    print("   - 探索新的训练方法和正则化技术")
    
    print("\n5. 实用工具开发：")
    print("   - 开发更易用的Flow模型库")
    print("   - 提供预训练模型和最佳实践")


def main():
    """
    主函数，运行所有总结内容
    """
    print("Normalizing Flows - 知识点5：总结")
    print("=" * 60)
    
    core_idea()
    key_advantages()
    key_disadvantages()
    application_scenarios()
    comparison_with_other_models()
    comprehensive_example()
    future_directions()
    
    print("\n" + "=" * 60)
    print("Normalizing Flows 知识点总结完成！")
    print("\n关键要点：")
    print("- Normalizing Flows通过可逆变换将简单分布转换为复杂分布")
    print("- 可以精确计算似然函数，这是其主要优势")
    print("- 适用于密度估计、生成建模等任务")
    print("- 未来发展方向包括更高效的架构和更广泛的应用")


if __name__ == "__main__":
    main()
