#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识点5：自回归模型与其他模型的比较

这个文件包含自回归模型与其他生成模型的比较，包括：
1. 自回归模型（ARMs）与自动编码器的比较
2. 自回归自编码器
3. MADE（掩码自动编码器）
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


def sigmoid(x):
    """
    Sigmoid激活函数
    
    参数:
        x: 输入值
    
    返回:
        sigmoid(x) 的值
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """
    ReLU激活函数
    
    参数:
        x: 输入值
    
    返回:
        relu(x) 的值
    """
    return np.maximum(0, x)


class Autoencoder:
    """
    自动编码器类
    """
    def __init__(self, input_dim, hidden_dim):
        """
        初始化自动编码器
        
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 编码器权重
        self.encoder_weights = np.random.randn(input_dim, hidden_dim) * 0.01
        self.encoder_bias = np.zeros((1, hidden_dim))
        
        # 解码器权重
        self.decoder_weights = np.random.randn(hidden_dim, input_dim) * 0.01
        self.decoder_bias = np.zeros((1, input_dim))
    
    def encode(self, x):
        """
        编码过程
        
        参数:
            x: 输入向量或矩阵
        
        返回:
            隐藏层表示
        """
        return sigmoid(np.dot(x, self.encoder_weights) + self.encoder_bias)
    
    def decode(self, h):
        """
        解码过程
        
        参数:
            h: 隐藏层表示
        
        返回:
            重建的输入
        """
        return sigmoid(np.dot(h, self.decoder_weights) + self.decoder_bias)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入向量或矩阵
        
        返回:
            重建的输入和隐藏层表示
        """
        h = self.encode(x)
        x_recon = self.decode(h)
        return x_recon, h
    
    def compute_loss(self, x):
        """
        计算重构损失（交叉熵）
        
        参数:
            x: 输入向量或矩阵
        
        返回:
            平均重构损失
        """
        x_recon, _ = self.forward(x)
        loss = -np.mean(x * np.log(x_recon + 1e-8) + (1 - x) * np.log(1 - x_recon + 1e-8))
        return loss
    
    def sample(self, n_samples=1):
        """
        从自动编码器采样
        
        参数:
            n_samples: 采样数量
        
        返回:
            采样结果
        """
        # 自动编码器不是直接生成模型，需要从隐藏空间采样
        h = np.random.binomial(1, 0.5, size=(n_samples, self.hidden_dim))
        return self.decode(h)


class MaskedAutoencoderForDistributionEstimation(MADE):
    """
    掩码自动编码器用于分布估计（MADE）
    """
    def __init__(self, input_dim, hidden_dims, output_dim=None):
        """
        初始化MADE
        
        参数:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度（默认与输入维度相同）
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim or input_dim
        
        # 堆叠所有层的维度
        self.all_dims = [input_dim] + hidden_dims + [self.output_dim]
        self.n_layers = len(self.all_dims) - 1
        
        # 初始化掩码
        self.masks = []
        self.weights = []
        self.biases = []
        
        # 为每个输入神经元分配随机顺序（1到input_dim）
        self.input_order = np.arange(1, input_dim + 1)
        
        # 创建掩码和权重
        for i in range(self.n_layers):
            in_dim = self.all_dims[i]
            out_dim = self.all_dims[i+1]
            
            # 为隐藏层分配随机顺序
            if i < self.n_layers - 1:  # 不是最后一层
                # 隐藏层单元顺序在1到input_dim-1之间
                hidden_order = np.random.randint(
                    1, 
                    input_dim, 
                    size=out_dim
                )
            else:  # 最后一层
                # 输出层单元顺序与输入层相同（每个输入维度对应一个输出）
                hidden_order = self.input_order
            
            # 创建掩码
            if i == 0:  # 输入层到第一层隐藏层
                prev_order = self.input_order
            else:  # 隐藏层之间
                prev_order = self.masks[i-1][1]
            
            # 创建下三角掩码：只有当prev_order < current_order时才连接
            mask = prev_order[:, np.newaxis] < hidden_order[np.newaxis, :]
            self.masks.append((mask, hidden_order))
            
            # 初始化权重（并应用掩码）
            weight = np.random.randn(in_dim, out_dim) * 0.01
            weight *= mask
            self.weights.append(weight)
            
            # 初始化偏置
            bias = np.zeros(out_dim)
            self.biases.append(bias)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入向量或矩阵
        
        返回:
            输出概率
        """
        h = x
        for i in range(self.n_layers):
            h = np.dot(h, self.weights[i]) + self.biases[i]
            if i < self.n_layers - 1:  # 不是最后一层
                h = sigmoid(h)
            else:  # 最后一层
                h = sigmoid(h)  # 对于二值输出
        return h
    
    def sample(self, n_samples=1):
        """
        从MADE模型采样
        
        参数:
            n_samples: 采样数量
        
        返回:
            采样结果
        """
        samples = np.zeros((n_samples, self.input_dim))
        
        # 按照输入顺序依次采样
        for i in range(self.input_dim):
            # 找到第i个输入对应的顺序
            idx = np.where(self.input_order == (i+1))[0][0]
            
            # 计算条件概率
            probs = self.forward(samples)
            
            # 采样当前维度
            p = probs[:, idx]
            samples[:, idx] = np.random.binomial(1, p)
        
        return samples
    
    def compute_loss(self, x):
        """
        计算交叉熵损失
        
        参数:
            x: 输入向量或矩阵
        
        返回:
            平均损失
        """
        probs = self.forward(x)
        loss = -np.mean(x * np.log(probs + 1e-8) + (1 - x) * np.log(1 - probs + 1e-8))
        return loss
    
    def plot_masks(self):
        """
        绘制掩码矩阵
        """
        fig, axes = plt.subplots(1, self.n_layers, figsize=(15, 5))
        
        for i in range(self.n_layers):
            mask, order = self.masks[i]
            axes[i].imshow(mask, cmap='binary')
            axes[i].set_title(f'层 {i+1} 掩码')
            axes[i].set_xlabel('输出单元')
            axes[i].set_ylabel('输入单元')
        
        plt.tight_layout()
        plt.savefig('figure/made_masks.png')
        print("MADE掩码图已保存到 figure/made_masks.png")


class AutoregressiveAutoencoder:
    """
    自回归自动编码器
    """
    def __init__(self, input_dim, hidden_dim):
        """
        初始化自回归自动编码器
        
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 创建MADE模型作为自回归组件
        self.made = MaskedAutoencoderForDistributionEstimation(input_dim, [hidden_dim])
        
        # 创建编码器（与MADE共享部分权重）
        self.encoder_weights = np.random.randn(input_dim, hidden_dim) * 0.01
        self.encoder_bias = np.zeros((1, hidden_dim))
    
    def encode(self, x):
        """
        编码过程
        
        参数:
            x: 输入向量或矩阵
        
        返回:
            隐藏层表示
        """
        return sigmoid(np.dot(x, self.encoder_weights) + self.encoder_bias)
    
    def generate(self, x):
        """
        生成过程（自回归）
        
        参数:
            x: 输入向量或矩阵
        
        返回:
            生成的概率
        """
        return self.made.forward(x)
    
    def sample(self, n_samples=1):
        """
        从模型采样
        
        参数:
            n_samples: 采样数量
        
        返回:
            采样结果
        """
        return self.made.sample(n_samples)


# 辅助函数
def binarize_mnist():
    """
    加载并二值化MNIST数据集
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 归一化并二值化
    x_train = (x_train / 255.0 > 0.5).astype(int)
    x_test = (x_test / 255.0 > 0.5).astype(int)
    
    # 展平
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    
    return x_train, x_test


def plot_reconstructions(original, reconstructed, filename="figure/reconstructions.png"):
    """
    绘制原始图像和重建图像
    
    参数:
        original: 原始图像数据
        reconstructed: 重建图像数据
        filename: 保存文件名
    """
    n_samples = min(5, original.shape[0])
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples*2, 4))
    
    for i in range(n_samples):
        # 原始图像
        axes[0, i].imshow(original[i].reshape(28, 28), cmap='binary')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('原始')
        
        # 重建图像
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='binary')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('重建')
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"重建图像已保存到 {filename}")


def plot_samples(samples, filename="figure/samples.png"):
    """
    绘制采样结果
    
    参数:
        samples: 采样的图像数据
        filename: 保存文件名
    """
    n_samples = min(10, samples.shape[0])
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples*2, 2))
    
    for i in range(n_samples):
        axes[i].imshow(samples[i].reshape(28, 28), cmap='binary')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"采样图像已保存到 {filename}")


# 示例函数
def autoencoder_demo():
    """
    自动编码器演示
    """
    print("\n=== 自动编码器演示 ===")
    
    # 加载二值化MNIST数据
    x_train, x_test = binarize_mnist()
    
    # 创建自动编码器
    autoencoder = Autoencoder(input_dim=784, hidden_dim=128)
    
    # 测试前向传播
    x_recon, h = autoencoder.forward(x_test[:10])
    print(f"自动编码器输入形状: {x_test[:10].shape}")
    print(f"自动编码器隐藏层形状: {h.shape}")
    print(f"自动编码器重建输出形状: {x_recon.shape}")
    
    # 绘制重建结果
    plot_reconstructions(x_test[:5], x_recon[:5], "figure/autoencoder_reconstructions.png")
    
    # 测试采样
    samples = autoencoder.sample(5)
    plot_samples(samples, "figure/autoencoder_samples.png")


def made_demo():
    """
    MADE演示
    """
    print("\n=== MADE演示 ===")
    
    # 创建MADE模型
    made = MaskedAutoencoderForDistributionEstimation(
        input_dim=10,
        hidden_dims=[5, 3]
    )
    
    # 绘制掩码
    made.plot_masks()
    
    # 生成随机数据
    x_train = np.random.binomial(1, 0.5, size=(100, 10))
    
    # 测试前向传播
    probs = made.forward(x_train)
    print(f"MADE前向传播输出形状: {probs.shape}")
    
    # 采样
    samples = made.sample(5)
    print(f"MADE采样结果: {samples}")


def compare_models():
    """
    比较自回归模型与自动编码器
    """
    print("\n=== 模型比较 ===")
    
    # 加载二值化MNIST数据（使用小样本）
    x_train, x_test = binarize_mnist()
    x_train_small = x_train[:1000]
    x_test_small = x_test[:100]
    
    # 创建模型
    input_dim = 784
    hidden_dim = 128
    
    # 自动编码器
    autoencoder = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    
    # MADE（自回归模型）
    made = MaskedAutoencoderForDistributionEstimation(
        input_dim=input_dim,
        hidden_dims=[hidden_dim]
    )
    
    print(f"模型比较：")
    print(f"- 输入维度: {input_dim}")
    print(f"- 隐藏层维度: {hidden_dim}")
    print()
    
    # 比较参数数量
    autoencoder_params = (
        autoencoder.encoder_weights.size +
        autoencoder.encoder_bias.size +
        autoencoder.decoder_weights.size +
        autoencoder.decoder_bias.size
    )
    
    made_params = sum(w.size for w in made.weights) + sum(b.size for b in made.biases)
    
    print(f"参数数量比较：")
    print(f"  自动编码器: {autoencoder_params:,}")
    print(f"  MADE: {made_params:,}")
    print()
    
    # 比较功能
    print(f"功能比较：")
    print(f"  自动编码器:")
    print(f"    - 可以进行数据压缩")
    print(f"    - 可以进行去噪")
    print(f"    - 不是直接的生成模型")
    print(f"    - 采样需要从隐藏空间随机采样")
    print()
    print(f"  MADE:")
    print(f"    - 是直接的生成模型")
    print(f"    - 可以进行精确的密度估计")
    print(f"    - 采样是顺序的，依赖于之前的采样结果")
    print(f"    - 使用掩码确保自回归结构")


def autoregressive_autoencoder_demo():
    """
    自回归自动编码器演示
    """
    print("\n=== 自回归自动编码器演示 ===")
    
    # 创建自回归自动编码器
    arae = AutoregressiveAutoencoder(input_dim=10, hidden_dim=5)
    
    # 生成随机数据
    x_train = np.random.binomial(1, 0.5, size=(100, 10))
    
    # 测试编码
    h = arae.encode(x_train[:5])
    print(f"自回归自动编码器编码输出形状: {h.shape}")
    
    # 测试生成
    probs = arae.generate(x_train[:5])
    print(f"自回归自动编码器生成输出形状: {probs.shape}")
    
    # 测试采样
    samples = arae.sample(5)
    print(f"自回归自动编码器采样结果: {samples}")


if __name__ == "__main__":
    print("===== 自回归模型与其他模型的比较 =====")
    
    # 演示自动编码器
    # autoencoder_demo()  # 注意：在实际运行时会消耗较多内存
    
    # 演示MADE
    made_demo()
    
    # 比较模型
    compare_models()
    
    # 演示自回归自动编码器
    autoregressive_autoencoder_demo()
    
    print("\n所有演示完成！")
