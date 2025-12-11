#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识点4：GAN的优化与挑战

本文件包含GAN的优化与挑战的相关代码实现，包括：
1. GAN的训练过程
2. GAN研究的前沿与挑战
3. 训练不稳定性问题
4. 模式崩溃现象
5. 超越KL和JS散度
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


def gan_training_process():
    """
    GAN的训练过程
    
    GAN的优化问题：
    min_θ max_φ [ E_{x~p_data}[log(D_φ(x))] + E_{z~p_z}[log(1 - D_φ(G_θ(z)))] ]
    
    训练步骤：
    1. 从D中采样m个训练点x_1,...,x_m，从p_z中采样m个噪声向量z_1,...,z_m
    2. 通过随机梯度上升更新判别器参数φ
       φ ← φ + (η1/m) ∇_φ [ Σ_{i=1}^m (log(D_φ(x_i)) + log(1 - D_φ(G_θ(z_i)))) ]
    3. 通过随机梯度下降更新生成器参数θ
       θ ← θ - (η2/m) ∇_θ [ Σ_{i=1}^m log(1 - D_φ(G_θ(z_i))) ]
    4. 重复固定次数的epoch
    """
    print("=== GAN的训练过程 ===")
    
    # 创建一个简单的GAN示例
    class SimpleGAN:
        def __init__(self, latent_dim=10, data_dim=2):
            self.latent_dim = latent_dim
            self.data_dim = data_dim
            
            # 构建生成器
            self.generator = self.build_generator()
            
            # 构建判别器
            self.discriminator = self.build_discriminator()
            
            # 编译判别器
            self.discriminator.compile(optimizer='adam',
                                     loss='binary_crossentropy',
                                     metrics=['accuracy'])
            
            # 构建完整的GAN模型
            self.gan = self.build_gan()
            
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
                layers.Dense(1, activation='sigmoid')
            ])
            return model
        
        def build_gan(self):
            """构建完整的GAN模型"""
            # 冻结判别器的权重
            self.discriminator.trainable = False
            
            # GAN输入（噪声）
            gan_input = layers.Input(shape=(self.latent_dim,))
            
            # 生成器生成图像
            generated_data = self.generator(gan_input)
            
            # 判别器判断生成图像
            gan_output = self.discriminator(generated_data)
            
            # 编译GAN模型
            gan = tf.keras.Model(gan_input, gan_output)
            gan.compile(optimizer='adam', loss='binary_crossentropy')
            
            return gan
        
        def train(self, real_data, epochs=1000, batch_size=32):
            """训练GAN模型"""
            self.g_losses = []
            self.d_losses = []
            self.d_accs = []
            
            for epoch in range(epochs):
                # --------------------- #
                #  训练判别器
                # --------------------- #
                
                # 从真实数据中采样batch_size个样本
                idx = np.random.randint(0, real_data.shape[0], batch_size)
                real_batch = real_data[idx]
                
                # 生成噪声向量
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                
                # 使用生成器生成假样本
                generated_batch = self.generator.predict(noise)
                
                # 准备判别器的训练数据
                X = np.concatenate([real_batch, generated_batch])
                y = np.concatenate([np.ones((batch_size, 1)),
                                  np.zeros((batch_size, 1))])
                
                # 训练判别器
                d_loss, d_acc = self.discriminator.train_on_batch(X, y)
                
                # --------------------- #
                #  训练生成器
                # --------------------- #
                
                # 生成新的噪声向量
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                
                # 准备生成器的训练数据（标签全部为1，希望生成器欺骗判别器）
                misleading_targets = np.ones((batch_size, 1))
                
                # 训练生成器
                g_loss = self.gan.train_on_batch(noise, misleading_targets)
                
                # 保存损失和准确率
                self.g_losses.append(g_loss)
                self.d_losses.append(d_loss)
                self.d_accs.append(d_acc)
                
                # 每隔100个epoch打印一次进度
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss:.4f}, D Acc: {d_acc:.4f}, G Loss: {g_loss:.4f}")
    
    # 生成真实数据
    def generate_real_data(n_samples):
        """生成真实数据样本"""
        return np.concatenate([
            np.random.multivariate_normal([-1, -1], [[0.1, 0], [0, 0.1]], n_samples // 2),
            np.random.multivariate_normal([1, 1], [[0.1, 0], [0, 0.1]], n_samples // 2)
        ])
    
    # 训练GAN
    np.random.seed(42)
    real_data = generate_real_data(1000)
    
    gan = SimpleGAN(latent_dim=10, data_dim=2)
    gan.train(real_data, epochs=500, batch_size=32)
    
    # 可视化训练过程
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(gan.g_losses, label='生成器损失')
    plt.plot(gan.d_losses, label='判别器损失')
    plt.title('GAN训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(gan.d_accs)
    plt.title('判别器准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('figure/gan_training_process.png')
    plt.close()
    
    print("\nGAN训练过程已可视化并保存到figure/gan_training_process.png")


def gan_research_challenges():
    """
    GAN研究的前沿与挑战
    
    GAN已成功应用于多个领域和任务，但在实践中使用GAN可能非常具有挑战性：
    - 训练不稳定
    - 模式崩溃
    
    需要各种技巧来成功训练GAN
    """
    print("\n=== GAN研究的前沿与挑战 ===")
    
    print("GAN的主要挑战：")
    print("1. 训练不稳定性：生成器和判别器的损失在训练过程中不断振荡")
    print("2. 模式崩溃：生成器崩溃到一个或几个样本（称为"模式"）")
    print("\n成功训练GAN的技巧：")
    print("- 使用合适的网络架构（如DCGAN中推荐的架构）")
    print("- 使用合适的损失函数（如WGAN的Wasserstein损失）")
    print("- 平衡生成器和判别器的训练进度")
    print("- 使用批量归一化和其他正则化技术")
    print("- 小心调整学习率")
    print("- 使用噪声注入技术")


def training_instability():
    """
    训练不稳定性问题
    
    定理（非正式）：如果生成器更新在函数空间中，且判别器在每一步都是最优的，
    则生成器保证收敛到数据分布
    
    但这个假设在实践中是不现实的！
    
    在实践中，生成器和判别器的损失在训练过程中不断振荡
    """
    print("\n=== 训练不稳定性问题 ===")
    
    # 模拟不稳定的训练过程
    np.random.seed(42)
    epochs = 200
    
    # 生成振荡的损失值
    g_losses = []
    d_losses = []
    
    for epoch in range(epochs):
        # 生成器损失：周期性振荡
        g_loss = 0.5 + 0.3 * np.sin(epoch * 0.1) + 0.1 * np.random.randn()
        
        # 判别器损失：与生成器损失相反的振荡
        d_loss = 0.5 - 0.2 * np.sin(epoch * 0.1) + 0.1 * np.random.randn()
        
        g_losses.append(g_loss)
        d_losses.append(d_loss)
    
    # 可视化不稳定的训练
    plt.figure(figsize=(10, 6))
    
    plt.plot(g_losses, label='生成器损失')
    plt.plot(d_losses, label='判别器损失')
    
    plt.title('GAN训练不稳定性示例')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('figure/training_instability.png')
    plt.close()
    
    print("\n训练不稳定性示例已可视化并保存到figure/training_instability.png")
    print("\n训练不稳定性的特点：")
    print("- 生成器和判别器的损失不断振荡")
    print("- 没有可靠的停止标准")
    print("- 需要仔细调整超参数来缓解")


def mode_collapse():
    """
    模式崩溃现象
    
    GAN因遭受模式崩溃而臭名昭著
    直观地说，这指的是GAN的生成器崩溃到一个或几个样本（称为"模式"）的现象
    """
    print("\n=== 模式崩溃现象 ===")
    
    # 模拟模式崩溃
    np.random.seed(42)
    
    # 真实分布：混合高斯分布（4个模式）
    def true_distribution(n_samples):
        """生成真实分布的样本"""
        means = [(-3, -3), (-3, 3), (3, -3), (3, 3)]
        samples = []
        
        for _ in range(n_samples):
            mean = means[np.random.choice(len(means))]
            sample = np.random.multivariate_normal(mean, [[0.5, 0], [0, 0.5]])
            samples.append(sample)
        
        return np.array(samples)
    
    # 模拟模式崩溃的生成器
    def generator_with_mode_collapse(n_samples, current_mode=0):
        """模拟具有模式崩溃的生成器"""
        means = [(-3, -3), (-3, 3), (3, -3), (3, 3)]
        
        # 只生成当前模式的样本
        mean = means[current_mode]
        samples = np.random.multivariate_normal(mean, [[0.5, 0], [0, 0.5]], n_samples)
        
        return samples
    
    # 生成真实数据
    real_data = true_distribution(1000)
    
    # 可视化模式崩溃过程
    plt.figure(figsize=(12, 9))
    
    for i, mode in enumerate([0, 1, 2, 3]):
        # 生成当前模式的样本
        generated_data = generator_with_mode_collapse(1000, mode)
        
        plt.subplot(2, 2, i+1)
        plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.3, label='真实数据')
        plt.scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.3, label='生成数据')
        plt.title(f'模式崩溃示例 - 生成器只生成模式 {mode+1}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('figure/mode_collapse.png')
    plt.close()
    
    print("\n模式崩溃示例已可视化并保存到figure/mode_collapse.png")
    print("\n模式崩溃的特点：")
    print("- 生成器只生成真实分布中的一个或几个模式")
    print("- 生成器分布在不同模式之间不断振荡")
    print("- 难以捕获真实分布的全部多样性")


def beyond_kl_js_divergence():
    """
    超越KL和JS散度
    
    对于d(P_data, P_θ)，我们有哪些选择？
    - KL散度：VAEs, ARMs
    - （缩放和移位的）JS散度（近似）：Vanilla GAN
    
    其他选择包括：
    - Wasserstein距离（WGAN）
    - Earth Mover距离
    - Pearson χ²散度
    - Squared Hellinger距离
    - 等
    """
    print("\n=== 超越KL和JS散度 ===")
    
    # 比较不同的散度度量
    def kl_divergence(p, q):
        """计算KL散度"""
        return np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
    
    def js_divergence(p, q):
        """计算JS散度"""
        m = (p + q) / 2
        return (kl_divergence(p, m) + kl_divergence(q, m)) / 2
    
    def wasserstein_distance(p, q, x):
        """近似计算Wasserstein距离"""
        # 简单近似：计算累积分布函数的L1距离
        cdf_p = np.cumsum(p) / np.sum(p)
        cdf_q = np.cumsum(q) / np.sum(q)
        return np.mean(np.abs(cdf_p - cdf_q))
    
    # 生成两个分布
    x = np.linspace(-5, 5, 1000)
    
    # 分布p：N(0, 1)
    p = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    p = p / np.sum(p)  # 归一化
    
    # 计算不同均值下的各种散度
    means = np.linspace(0, 3, 20)
    kl_values = []
    js_values = []
    wasserstein_values = []
    
    for mean in means:
        # 分布q：N(mean, 1)
        q = np.exp(-0.5 * (x - mean)**2) / np.sqrt(2 * np.pi)
        q = q / np.sum(q)  # 归一化
        
        # 计算各种散度
        kl = kl_divergence(p, q)
        js = js_divergence(p, q)
        wasserstein = wasserstein_distance(p, q, x)
        
        kl_values.append(kl)
        js_values.append(js)
        wasserstein_values.append(wasserstein)
    
    # 可视化不同散度的比较
    plt.figure(figsize=(10, 6))
    
    plt.plot(means, kl_values, label='KL散度')
    plt.plot(means, js_values, label='JS散度')
    plt.plot(means, wasserstein_values, label='Wasserstein距离（近似）')
    
    plt.title('不同散度度量的比较')
    plt.xlabel('两个高斯分布的均值差')
    plt.ylabel('散度值')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('figure/div_measures_comparison.png')
    plt.close()
    
    print("\n不同散度度量的比较已可视化并保存到figure/div_measures_comparison.png")
    print("\n不同散度的特点：")
    print("- KL散度：不对称，可能导致模式丢失")
    print("- JS散度：对称，但在分布不重叠时存在梯度消失问题")
    print("- Wasserstein距离：平滑，提供更稳定的训练信号")


if __name__ == "__main__":
    gan_training_process()
    gan_research_challenges()
    training_instability()
    mode_collapse()
    beyond_kl_js_divergence()
