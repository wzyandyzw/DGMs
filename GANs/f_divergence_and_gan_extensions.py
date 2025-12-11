#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识点5：f-散度与GAN扩展

本文件包含f-散度与GAN扩展的相关代码实现，包括：
1. f-散度的定义与性质
2. 不同f函数对应的散度度量
3. f-GAN的原理与实现
4. 可视化不同散度度量的比较
5. GAN的扩展介绍
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


def f_divergence_definition():
    """
    f-散度的定义与性质
    
    f-散度是两个概率分布之间差异的度量，定义为：
    D_f(P || Q) = ∫ Q(x) f(P(x)/Q(x)) dx
    
    其中f是凸函数，且满足f(1) = 0
    """
    print("=== f-散度的定义与性质 ===")
    
    print("f-散度的定义：")
    print("D_f(P || Q) = ∫ Q(x) f(P(x)/Q(x)) dx")
    print("其中f是凸函数，且满足f(1) = 0")
    
    print("\nf-散度的性质：")
    print("1. 非负性：D_f(P || Q) ≥ 0，当且仅当P = Q时取等号")
    print("2. 凸性：对于分布的凸组合，f-散度是凸的")
    print("3. 平移不变性：对于连续随机变量的线性变换，具有不变性")
    print("4. 可分解性：对于独立变量的联合分布，f-散度可以分解为各个变量的f-散度之和")


def different_f_functions():
    """
    不同f函数对应的散度度量
    
    一些常见的f函数及其对应的散度：
    - f(t) = t log t - t + 1 → KL散度
    - f(t) = -log t → 逆向KL散度
    - f(t) = (t - 1)^2 → Pearson χ²散度
    - f(t) = (√t - 1)^2 → Squared Hellinger距离
    - f(t) = (t log t - (t - 1))/(log 2) → Jensen-Shannon散度
    """
    print("\n=== 不同f函数对应的散度度量 ===")
    
    # 定义各种f函数
    def f_kl(t):
        """KL散度对应的f函数"""
        return t * np.log(t + 1e-10) - t + 1
    
    def f_reverse_kl(t):
        """逆向KL散度对应的f函数"""
        return -np.log(t + 1e-10)
    
    def f_chi_squared(t):
        """Pearson χ²散度对应的f函数"""
        return (t - 1)**2
    
    def f_hellinger(t):
        """Squared Hellinger距离对应的f函数"""
        return (np.sqrt(t + 1e-10) - 1)**2
    
    def f_js(t):
        """Jensen-Shannon散度对应的f函数"""
        return (t * np.log(t + 1e-10) - (t - 1)) / np.log(2)
    
    # 可视化各种f函数
    t = np.linspace(0.1, 3, 100)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.plot(t, f_kl(t))
    plt.title('KL散度的f函数')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(t, f_reverse_kl(t))
    plt.title('逆向KL散度的f函数')
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(t, f_chi_squared(t))
    plt.title('Pearson χ²散度的f函数')
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(t, f_hellinger(t))
    plt.title('Squared Hellinger距离的f函数')
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(t, f_js(t))
    plt.title('Jensen-Shannon散度的f函数')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('figure/f_functions.png')
    plt.close()
    
    print("\n各种f函数已可视化并保存到figure/f_functions.png")
    print("\n常见的f函数及其对应的散度：")
    print("- f(t) = t log t - t + 1 → KL散度")
    print("- f(t) = -log t → 逆向KL散度")
    print("- f(t) = (t - 1)^2 → Pearson χ²散度")
    print("- f(t) = (√t - 1)^2 → Squared Hellinger距离")
    print("- f(t) = (t log t - (t - 1))/(log 2) → Jensen-Shannon散度")


def f_gan_principle():
    """
    f-GAN的原理与实现
    
    f-GAN是基于f-散度的GAN扩展框架，其目标函数为：
    min_θ max_φ [ E_{x~p_data}[D_φ(x)] - E_{z~p_z}[f^*(D_φ(G_θ(z)))] ]
    
    其中f^*(·)是f的Fenchel共轭函数
    """
    print("\n=== f-GAN的原理与实现 ===")
    
    # 定义Fenchel共轭函数
    def f_star(f, u, t_range=(0.1, 100), num_samples=1000):
        """计算f的Fenchel共轭函数"""
        t_values = np.linspace(*t_range, num_samples)
        return np.max(t_values * u - f(t_values))
    
    # 定义KL散度对应的Fenchel共轭函数
    def f_star_kl(u):
        """KL散度对应的Fenchel共轭函数"""
        return np.exp(u - 1)
    
    # 定义Squared Hellinger距离对应的Fenchel共轭函数
    def f_star_hellinger(u):
        """Squared Hellinger距离对应的Fenchel共轭函数"""
        return (u / 2)**2 + u
    
    # 可视化Fenchel共轭函数
    u = np.linspace(-2, 5, 100)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(u, f_star_kl(u), label='KL散度的Fenchel共轭函数')
    plt.plot(u, f_star_hellinger(u), label='Squared Hellinger距离的Fenchel共轭函数')
    
    plt.title('Fenchel共轭函数示例')
    plt.xlabel('u')
    plt.ylabel('f^*(u)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('figure/fenchel_conjugate.png')
    plt.close()
    
    print("\nFenchel共轭函数示例已可视化并保存到figure/fenchel_conjugate.png")
    print("\nf-GAN的目标函数：")
    print("min_θ max_φ [ E_{x~p_data}[D_φ(x)] - E_{z~p_z}[f^*(D_φ(G_θ(z)))] ]")
    print("其中f^*(·)是f的Fenchel共轭函数")


def f_gan_implementation():
    """
    f-GAN的简单实现
    
    使用不同的f-散度实现f-GAN
    """
    print("\n=== f-GAN的简单实现 ===")
    
    class SimpleFGAN:
        def __init__(self, latent_dim=10, data_dim=2, divergence_type='kl'):
            self.latent_dim = latent_dim
            self.data_dim = data_dim
            self.divergence_type = divergence_type
            
            # 设置Fenchel共轭函数
            if divergence_type == 'kl':
                self.f_star = lambda u: tf.exp(u - 1)
            elif divergence_type == 'hellinger':
                self.f_star = lambda u: (u / 2)**2 + u
            elif divergence_type == 'chi_squared':
                self.f_star = lambda u: u * (u + 4) / 4
            else:
                raise ValueError(f"不支持的散度类型: {divergence_type}")
            
            # 构建生成器
            self.generator = self.build_generator()
            
            # 构建判别器
            self.discriminator = self.build_discriminator()
            
            # 构建完整的f-GAN模型
            self.gan = self.build_f_gan()
            
        def build_generator(self):
            """构建生成器"""
            model = tf.keras.Sequential([
                layers.Dense(128, activation='relu', input_dim=self.latent_dim),
                layers.Dense(256, activation='relu'),
                layers.Dense(self.data_dim, activation='tanh')
            ])
            return model
        
        def build_discriminator(self):
            """构建判别器"""
            model = tf.keras.Sequential([
                layers.Dense(256, activation='relu', input_dim=self.data_dim),
                layers.Dense(128, activation='relu'),
                layers.Dense(1, activation=None)  # 注意：f-GAN的判别器输出通常没有激活函数
            ])
            return model
        
        def build_f_gan(self):
            """构建完整的f-GAN模型"""
            # 冻结判别器的权重
            self.discriminator.trainable = False
            
            # GAN输入（噪声）
            gan_input = layers.Input(shape=(self.latent_dim,))
            
            # 生成器生成图像
            generated_data = self.generator(gan_input)
            
            # 判别器判断生成图像
            gan_output = self.discriminator(generated_data)
            
            # 编译f-GAN模型
            gan = tf.keras.Model(gan_input, gan_output)
            gan.compile(optimizer='adam', loss=self.f_gan_loss)
            
            return gan
        
        def discriminator_loss(self, real_output, fake_output):
            """判别器损失"""
            return tf.reduce_mean(real_output) - tf.reduce_mean(self.f_star(fake_output))
        
        def f_gan_loss(self, _, fake_output):
            """f-GAN损失（生成器损失）"""
            return tf.reduce_mean(self.f_star(fake_output))
        
        def train_discriminator(self, real_data, batch_size=32):
            """训练判别器"""
            # 从真实数据中采样batch_size个样本
            idx = np.random.randint(0, real_data.shape[0], batch_size)
            real_batch = real_data[idx]
            
            # 生成噪声向量
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # 使用生成器生成假样本
            generated_batch = self.generator.predict(noise)
            
            # 计算判别器输出
            real_output = self.discriminator(real_batch)
            fake_output = self.discriminator(generated_batch)
            
            # 计算损失
            loss = self.discriminator_loss(real_output, fake_output)
            
            # 反向传播
            with tf.GradientTape() as tape:
                real_output = self.discriminator(real_batch)
                fake_output = self.discriminator(generated_batch)
                loss = self.discriminator_loss(real_output, fake_output)
            
            # 更新判别器权重
            gradients_of_discriminator = tape.gradient(loss, self.discriminator.trainable_variables)
            optimizer = tf.keras.optimizers.Adam()
            optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
            
            return loss.numpy()
        
        def train(self, real_data, epochs=1000, batch_size=32, d_steps=1):
            """训练f-GAN模型"""
            self.g_losses = []
            self.d_losses = []
            
            for epoch in range(epochs):
                # --------------------- #
                #  训练判别器
                # --------------------- #
                
                for _ in range(d_steps):
                    d_loss = self.train_discriminator(real_data, batch_size)
                
                # --------------------- #
                #  训练生成器
                # --------------------- #
                
                # 生成新的噪声向量
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                
                # 准备生成器的训练数据（标签可以是任意值，因为损失函数不使用）
                dummy_labels = np.ones((batch_size, 1))
                
                # 训练生成器
                g_loss = self.gan.train_on_batch(noise, dummy_labels)
                
                # 保存损失
                self.g_losses.append(g_loss)
                self.d_losses.append(d_loss)
                
                # 每隔100个epoch打印一次进度
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
    
    # 生成真实数据
    def generate_real_data(n_samples):
        """生成真实数据样本"""
        return np.concatenate([
            np.random.multivariate_normal([-1, -1], [[0.1, 0], [0, 0.1]], n_samples // 2),
            np.random.multivariate_normal([1, 1], [[0.1, 0], [0, 0.1]], n_samples // 2)
        ])
    
    # 训练不同f-散度的f-GAN
    np.random.seed(42)
    real_data = generate_real_data(1000)
    
    # 可视化不同f-散度的训练效果
    plt.figure(figsize=(12, 6))
    
    for i, divergence_type in enumerate(['kl', 'hellinger']):
        # 训练f-GAN模型
        f_gan = SimpleFGAN(latent_dim=10, data_dim=2, divergence_type=divergence_type)
        f_gan.train(real_data, epochs=500, batch_size=32)
        
        # 生成测试样本
        noise = np.random.normal(0, 1, (1000, f_gan.latent_dim))
        generated_data = f_gan.generator.predict(noise)
        
        # 可视化生成结果
        plt.subplot(1, 2, i+1)
        plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.3, label='真实数据')
        plt.scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.3, label='生成数据')
        plt.title(f'{divergence_type.upper()} f-GAN生成结果')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('figure/f_gan_results.png')
    plt.close()
    
    print("\nf-GAN生成结果已可视化并保存到figure/f_gan_results.png")


def visualize_divergence_comparison():
    """
    可视化不同散度度量的比较
    
    比较不同散度度量对两个分布差异的敏感性
    """
    print("\n=== 可视化不同散度度量的比较 ===")
    
    # 计算各种散度
    def kl_divergence(p, q):
        """计算KL散度"""
        return np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
    
    def reverse_kl_divergence(p, q):
        """计算逆向KL散度"""
        return np.sum(q * np.log(q / (p + 1e-10) + 1e-10))
    
    def chi_squared_divergence(p, q):
        """计算Pearson χ²散度"""
        return np.sum((p - q)**2 / (q + 1e-10))
    
    def hellinger_distance(p, q):
        """计算Squared Hellinger距离"""
        return 0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2)
    
    def js_divergence(p, q):
        """计算Jensen-Shannon散度"""
        m = (p + q) / 2
        return (kl_divergence(p, m) + kl_divergence(q, m)) / 2
    
    # 生成两个分布并计算不同散度
    x = np.linspace(-5, 5, 1000)
    
    # 分布p：N(0, 1)
    p = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    p = p / np.sum(p)  # 归一化
    
    # 计算不同均值下的各种散度
    means = np.linspace(0, 3, 20)
    kl_values = []
    reverse_kl_values = []
    chi_squared_values = []
    hellinger_values = []
    js_values = []
    
    for mean in means:
        # 分布q：N(mean, 1)
        q = np.exp(-0.5 * (x - mean)**2) / np.sqrt(2 * np.pi)
        q = q / np.sum(q)  # 归一化
        
        # 计算各种散度
        kl_values.append(kl_divergence(p, q))
        reverse_kl_values.append(reverse_kl_divergence(p, q))
        chi_squared_values.append(chi_squared_divergence(p, q))
        hellinger_values.append(hellinger_distance(p, q))
        js_values.append(js_divergence(p, q))
    
    # 可视化不同散度的比较
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(means, kl_values, label='KL散度')
    plt.plot(means, reverse_kl_values, label='逆向KL散度')
    plt.plot(means, js_values, label='Jensen-Shannon散度')
    plt.title('不同散度度量的比较')
    plt.xlabel('两个高斯分布的均值差')
    plt.ylabel('散度值')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(means, chi_squared_values, label='Pearson χ²散度')
    plt.plot(means, hellinger_values, label='Squared Hellinger距离')
    plt.title('不同散度度量的比较（续）')
    plt.xlabel('两个高斯分布的均值差')
    plt.ylabel('散度值')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('figure/divergence_comparison.png')
    plt.close()
    
    print("\n不同散度度量的比较已可视化并保存到figure/divergence_comparison.png")


def gan_extensions():
    """
    GAN的扩展介绍
    
    GAN的一些重要扩展：
    - DCGAN：深度卷积GAN
    - WGAN：Wasserstein GAN
    - CycleGAN：循环一致性GAN
    - Conditional GAN：条件GAN
    - StackGAN：堆叠GAN
    - StyleGAN：风格GAN
    """
    print("\n=== GAN的扩展介绍 ===")
    
    print("GAN的重要扩展：")
    print("1. DCGAN（Deep Convolutional GAN）：使用深度卷积神经网络的GAN")
    print("2. WGAN（Wasserstein GAN）：使用Wasserstein距离作为损失函数的GAN")
    print("3. CycleGAN：处理无配对数据的图像到图像翻译")
    print("4. Conditional GAN：通过条件信息控制生成样本的类型")
    print("5. StackGAN：用于生成高分辨率图像的堆叠结构")
    print("6. StyleGAN：能够控制生成图像风格的GAN")
    
    # 可视化不同GAN扩展的应用场景
    print("\n不同GAN扩展的应用场景：")
    print("- 图像生成：DCGAN, StyleGAN")
    print("- 图像翻译：CycleGAN, pix2pix (Conditional GAN)")
    print("- 文本到图像生成：StackGAN, AttnGAN")
    print("- 视频生成：VideoGAN, TGAN")
    print("- 超分辨率：SRGAN, ESRGAN")
    print("- 图像编辑：StyleGAN, GauGAN")


if __name__ == "__main__":
    f_divergence_definition()
    different_f_functions()
    f_gan_principle()
    f_gan_implementation()
    visualize_divergence_comparison()
    gan_extensions()
