#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识点3：GAN基本原理与架构

本文件包含GAN基本原理与架构的相关代码实现，包括：
1. GAN的博弈论框架
2. 生成器与判别器的结构
3. GAN的目标函数推导
4. Jensen-Shannon散度的作用
5. GAN的总结
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from scipy.stats import norm


def gan_game_theory():
    """
    GAN的博弈论框架
    
    GAN是生成器和判别器之间的两人零和博弈（minimax game）
    
    生成器G尝试生成看起来像真实数据的样本，以欺骗判别器
    判别器D尝试区分真实数据和生成器生成的假数据
    
    最终目标是达到纳什均衡，此时生成器生成的样本与真实数据无法区分，
    判别器只能随机猜测（准确率为50%）
    """
    print("=== GAN的博弈论框架 ===")
    print("GAN是生成器和判别器之间的两人零和博弈（minimax game）")
    print("- 生成器G尝试生成看起来像真实数据的样本，以欺骗判别器")
    print("- 判别器D尝试区分真实数据和生成器生成的假数据")
    print("- 最终目标是达到纳什均衡")
    
    # 简单的博弈可视化
    plt.figure(figsize=(8, 6))
    
    # 绘制博弈过程
    x = np.linspace(0, 1, 100)  # 生成器质量（0到1）
    
    # 判别器性能随生成器质量变化
    # 当生成器质量差时，判别器性能好；当生成器质量好时，判别器性能接近0.5
    discriminator_performance = 0.5 + 0.5 * np.exp(-5 * x)
    
    plt.plot(x, discriminator_performance, label='判别器性能')
    plt.axhline(y=0.5, color='r', linestyle='--', label='随机猜测')
    
    plt.title('GAN的博弈过程')
    plt.xlabel('生成器质量')
    plt.ylabel('判别器性能（准确率）')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('figure/gan_game_theory.png')
    plt.close()
    
    print("\nGAN博弈过程已可视化并保存到figure/gan_game_theory.png")


def generator_discriminator_structure():
    """
    生成器与判别器的结构
    
    生成器G：
    - 有向 latent variable model，z和x之间由G_θ给出确定性映射
    - 从简单先验分布p_z采样z，设置x = G_θ(z)
    - 与高斯VAE的生成过程类似
    - 最小化两样本测试目标（支持零假设p_data = p_θ）
    
    判别器D：
    - 二分类器，区分真实数据和生成数据
    - 最大化两样本测试目标（支持备择假设p_data ≠ p_θ）
    """
    print("\n=== 生成器与判别器的结构 ===")
    
    # 创建一个简单的生成器模型
    def build_generator(latent_dim, output_dim):
        """构建生成器模型"""
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=latent_dim),
            layers.Dense(256, activation='relu'),
            layers.Dense(output_dim, activation='tanh')  # 使用tanh激活输出在[-1, 1]范围内
        ])
        return model
    
    # 创建一个简单的判别器模型
    def build_discriminator(input_dim):
        """构建判别器模型"""
        model = tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_dim=input_dim),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # 使用sigmoid激活输出概率
        ])
        return model
    
    # 构建模型示例
    latent_dim = 10
    data_dim = 2
    
    generator = build_generator(latent_dim, data_dim)
    discriminator = build_discriminator(data_dim)
    
    print("\n生成器结构：")
    generator.summary()
    
    print("\n判别器结构：")
    discriminator.summary()


def gan_objective_derivation():
    """
    GAN的目标函数推导
    
    GAN的优化问题对应：
    min_G max_D [E_{x~p_data}[log(D(x))] + E_{x~p_G}[log(1 - D(x))]]
    
    对于最优判别器D^*_G(·)，我们有：
    V(G, D^*_G) = 2*JS(p_data || p_G) - log(4)
    """
    print("\n=== GAN的目标函数推导 ===")
    
    # 生成数据分布
    np.random.seed(42)
    
    # 真实数据分布：二维高斯混合分布
    def generate_real_data(n_samples):
        """生成真实数据样本"""
        # 两个高斯分布的混合
        n_samples1 = n_samples // 2
        n_samples2 = n_samples - n_samples1
        
        samples1 = np.random.multivariate_normal([-1, -1], [[0.1, 0], [0, 0.1]], n_samples1)
        samples2 = np.random.multivariate_normal([1, 1], [[0.1, 0], [0, 0.1]], n_samples2)
        
        return np.concatenate([samples1, samples2])
    
    # 生成器分布：不同质量的生成数据
    def generate_generator_data(n_samples, quality=0.5):
        """生成生成器数据样本"""
        # 质量参数控制生成数据与真实数据的接近程度
        mean_shift = 2 * (1 - quality)
        
        n_samples1 = n_samples // 2
        n_samples2 = n_samples - n_samples1
        
        samples1 = np.random.multivariate_normal([-1 + mean_shift, -1 + mean_shift], 
                                               [[0.1 + 0.2*(1-quality), 0], [0, 0.1 + 0.2*(1-quality)]], 
                                               n_samples1)
        samples2 = np.random.multivariate_normal([1 - mean_shift, 1 - mean_shift], 
                                               [[0.1 + 0.2*(1-quality), 0], [0, 0.1 + 0.2*(1-quality)]], 
                                               n_samples2)
        
        return np.concatenate([samples1, samples2])
    
    # 计算GAN目标函数
    def gan_objective(real_samples, generated_samples, discriminator):
        """计算GAN目标函数值"""
        # 判别器对真实样本的预测
        d_real = discriminator.predict(real_samples)
        
        # 判别器对生成样本的预测
        d_generated = discriminator.predict(generated_samples)
        
        # 计算目标函数
        objective = np.mean(np.log(d_real + 1e-10)) + np.mean(np.log(1 - d_generated + 1e-10))
        
        return objective
    
    # 测试不同生成器质量下的GAN目标函数
    real_data = generate_real_data(1000)
    
    # 创建并训练判别器
    discriminator = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # 准备训练数据
    latent_dim = 10
    gen_data_bad = generate_generator_data(1000, quality=0.2)
    gen_data_good = generate_generator_data(1000, quality=0.8)
    
    X = np.concatenate([real_data, gen_data_bad])
    y = np.concatenate([np.ones(len(real_data)), np.zeros(len(gen_data_bad))])
    
    # 编译并训练判别器
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    discriminator.fit(X, y, epochs=10, batch_size=32, verbose=0)
    
    # 计算不同质量生成器的目标函数
    obj_bad = gan_objective(real_data, gen_data_bad, discriminator)
    obj_good = gan_objective(real_data, gen_data_good, discriminator)
    
    print(f"\n生成器质量差时的GAN目标函数值: {obj_bad:.4f}")
    print(f"生成器质量好时的GAN目标函数值: {obj_good:.4f}")
    print("\n说明：GAN目标函数值越接近-2log(2)≈-1.386，生成器质量越好")


def js_divergence_role():
    """
    Jensen-Shannon散度的作用
    
    JS散度定义：
    JS(P||Q) = JS(p||q) = (1/2)KL(p||(p+q)/2) + (1/2)KL(q||(p+q)/2)
    
    性质：
    - JS(p||q) ≥ 0
    - JS(p||q) = 0 当且仅当 p = q
    - JS(p||q) = JS(q||p) （对称）
    
    JS散度GAN的最优生成器：p_G^* = p_data
    
    对于最优判别器D^*_G^*(·)和生成器G^*(·)，我们有V(G^*, D^*_G^*) = -log(4)
    """
    print("\n=== Jensen-Shannon散度的作用 ===")
    
    # 计算JS散度
    def js_divergence(p, q):
        """计算两个分布之间的JS散度"""
        # 计算KL散度
        def kl_divergence(p, q):
            return np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
        
        # 计算中间分布
        m = (p + q) / 2
        
        # 计算JS散度
        js = (kl_divergence(p, m) + kl_divergence(q, m)) / 2
        
        return js
    
    # 示例：计算两个高斯分布之间的JS散度
    x = np.linspace(-5, 5, 1000)
    
    # 计算不同均值的高斯分布之间的JS散度
    means = np.linspace(0, 3, 10)
    js_values = []
    
    for mean in means:
        # 分布p：N(0, 1)
        p = norm.pdf(x, 0, 1)
        p = p / np.sum(p)  # 归一化
        
        # 分布q：N(mean, 1)
        q = norm.pdf(x, mean, 1)
        q = q / np.sum(q)  # 归一化
        
        js = js_divergence(p, q)
        js_values.append(js)
    
    # 可视化JS散度随均值差变化
    plt.figure(figsize=(8, 6))
    plt.plot(means, js_values)
    plt.title('JS散度随分布差异变化')
    plt.xlabel('两个高斯分布的均值差')
    plt.ylabel('JS散度值')
    plt.grid(True)
    
    plt.savefig('figure/js_divergence_role.png')
    plt.close()
    
    print("\nJS散度变化已可视化并保存到figure/js_divergence_role.png")
    print("\nJS散度性质：")
    print("1. JS(p||q) ≥ 0")
    print("2. JS(p||q) = 0 当且仅当 p = q")
    print("3. JS(p||q) = JS(q||p) （对称）")


def gan_summary():
    """
    GAN的总结
    
    - 选择d(P_data, P_θ)作为两样本测试统计量
      - 通过训练分类器（判别器）学习统计量
      - 在理想条件下，相当于选择d(P_data, P_θ)为JS(P_data||P_θ)
    
    优点：
    - 损失只需要来自p_θ的样本，不需要似然！
    - 神经网络架构有很大的灵活性
    - 快速采样（单次前向传播）
    
    缺点：实际训练非常困难
    """
    print("\n=== GAN的总结 ===")
    
    print("GAN的核心思想：")
    print("- 选择d(P_data, P_θ)作为两样本测试统计量")
    print("- 通过训练分类器（判别器）学习统计量")
    print("- 在理想条件下，相当于选择JS散度作为距离度量")
    
    print("\nGAN的优点：")
    print("1. 损失只需要来自p_θ的样本，不需要似然函数")
    print("2. 神经网络架构有很大的灵活性")
    print("3. 快速采样（单次前向传播）")
    
    print("\nGAN的缺点：")
    print("1. 实际训练非常困难")
    print("2. 可能出现模式崩溃")
    print("3. 难以评估训练进度")


if __name__ == "__main__":
    gan_game_theory()
    generator_discriminator_structure()
    gan_objective_derivation()
    js_divergence_role()
    gan_summary()
