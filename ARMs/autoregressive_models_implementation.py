#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识点4：自回归模型的具体实现

这个文件包含自回归模型的具体实现，包括：
1. 完全可见的sigmoid信念网络（FVSBNs）
2. 神经自回归分布估计（NADE）
3. NADE的权重共享
4. 一般离散分布的NADE
5. 实值NADE（RNADE）
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


def softmax(x):
    """
    Softmax激活函数
    
    参数:
        x: 输入向量
    
    返回:
        softmax(x) 的值
    """
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 数值稳定
    return exps / np.sum(exps, axis=-1, keepdims=True)


class FVSBN:
    """
    完全可见的sigmoid信念网络（Fully Visible Sigmoid Belief Network）
    """
    def __init__(self, n_features):
        """
        初始化FVSBN
        
        参数:
            n_features: 特征数量
        """
        self.n_features = n_features
        # 初始化权重矩阵：W[i][j] 是x_j对x_i的权重（i > j）
        self.W = np.random.randn(n_features, n_features) * 0.01
        # 对角线权重设为0（因为x_i不依赖自己）
        np.fill_diagonal(self.W, 0)
        # 偏置项
        self.b = np.random.randn(n_features) * 0.01
    
    def conditional_probability(self, x, i):
        """
        计算条件概率 P(x_i=1 | x_{<i})
        
        参数:
            x: 输入向量
            i: 特征索引
        
        返回:
            x_i=1的条件概率
        """
        if i == 0:
            return sigmoid(self.b[i])
        else:
            # 前i个特征的线性组合
            linear = self.b[i] + np.dot(self.W[i, :i], x[:i])
            return sigmoid(linear)
    
    def forward(self, x):
        """
        前向传播，计算所有条件概率
        
        参数:
            x: 输入向量或矩阵
        
        返回:
            所有条件概率 P(x_i=1 | x_{<i})
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n_samples = x.shape[0]
        probs = np.zeros_like(x, dtype=float)
        
        for i in range(self.n_features):
            if i == 0:
                probs[:, i] = sigmoid(self.b[i])
            else:
                # 计算前i个特征的线性组合
                linear = self.b[i] + np.dot(x[:, :i], self.W[i, :i].T)
                probs[:, i] = sigmoid(linear)
        
        return probs
    
    def sample(self, n_samples=1):
        """
        从模型采样
        
        参数:
            n_samples: 采样数量
        
        返回:
            采样结果
        """
        samples = np.zeros((n_samples, self.n_features), dtype=int)
        
        for i in range(self.n_features):
            if i == 0:
                # 第一个特征
                p = sigmoid(self.b[i])
            else:
                # 前i个特征的线性组合
                linear = self.b[i] + np.dot(samples[:, :i], self.W[i, :i].T)
                p = sigmoid(linear)
            
            # 采样
            samples[:, i] = np.random.binomial(1, p)
        
        return samples
    
    def compute_loss(self, X):
        """
        计算交叉熵损失
        
        参数:
            X: 训练数据
        
        返回:
            平均损失
        """
        probs = self.forward(X)
        # 交叉熵损失
        loss = -np.mean(X * np.log(probs + 1e-8) + (1 - X) * np.log(1 - probs + 1e-8))
        return loss


class NADE:
    """
    神经自回归分布估计（Neural Autoregressive Distribution Estimation）
    """
    def __init__(self, input_dim, hidden_dim):
        """
        初始化NADE
        
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 初始化权重（不共享版本）
        # 隐藏层权重：每个特征i有一个权重矩阵A_i和偏置c_i
        self.A = np.random.randn(input_dim, hidden_dim, input_dim) * 0.01
        self.c = np.random.randn(input_dim, hidden_dim) * 0.01
        
        # 输出层权重：每个特征i有一个权重向量alpha_i和偏置b_i
        self.alpha = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b = np.random.randn(input_dim) * 0.01
        
        # 设置为不共享模式
        self.shared_weights = False
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入向量或矩阵
        
        返回:
            所有条件概率 P(x_i=1 | x_{<i})
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n_samples = x.shape[0]
        probs = np.zeros_like(x, dtype=float)
        
        for i in range(self.input_dim):
            if i == 0:
                # 第一个特征没有输入，隐藏层输出为0
                h = np.zeros((n_samples, self.hidden_dim))
            else:
                # 隐藏层计算
                if self.shared_weights:
                    # 权重共享版本
                    linear = np.dot(x[:, :i], self.W[:i, :]) + self.c
                else:
                    # 不共享版本
                    linear = np.dot(x[:, :i], self.A[i, :i, :].T) + self.c[i, :]
                h = sigmoid(linear)
            
            # 输出层计算
            if self.shared_weights:
                linear_out = np.dot(h, self.V[:, i]) + self.b[i]
            else:
                linear_out = np.dot(h, self.alpha[i, :].T) + self.b[i]
            
            probs[:, i] = sigmoid(linear_out)
        
        return probs
    
    def enable_weight_sharing(self):
        """
        启用权重共享
        """
        # 初始化共享权重
        self.W = np.random.randn(self.input_dim, self.hidden_dim) * 0.01
        self.V = np.random.randn(self.hidden_dim, self.input_dim) * 0.01
        self.shared_weights = True
    
    def sample(self, n_samples=1):
        """
        从模型采样
        
        参数:
            n_samples: 采样数量
        
        返回:
            采样结果
        """
        samples = np.zeros((n_samples, self.input_dim), dtype=int)
        
        for i in range(self.input_dim):
            if i == 0:
                h = np.zeros((n_samples, self.hidden_dim))
            else:
                if self.shared_weights:
                    linear = np.dot(samples[:, :i], self.W[:i, :]) + self.c
                else:
                    linear = np.dot(samples[:, :i], self.A[i, :i, :].T) + self.c[i, :]
                h = sigmoid(linear)
            
            if self.shared_weights:
                linear_out = np.dot(h, self.V[:, i]) + self.b[i]
            else:
                linear_out = np.dot(h, self.alpha[i, :].T) + self.b[i]
            
            p = sigmoid(linear_out)
            samples[:, i] = np.random.binomial(1, p)
        
        return samples


class GeneralDiscreteNADE:
    """
    一般离散分布的NADE（支持多类别特征）
    """
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        """
        初始化一般离散分布的NADE
        
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_classes: 每个特征的类别数
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 启用权重共享
        self.W = np.random.randn(input_dim, hidden_dim) * 0.01
        self.c = np.random.randn(hidden_dim) * 0.01
        
        # 输出层权重：每个特征i有num_classes个输出
        self.V = np.random.randn(hidden_dim, input_dim * num_classes) * 0.01
        self.b = np.random.randn(input_dim * num_classes) * 0.01
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入向量或矩阵（每个元素是类别索引）
        
        返回:
            所有条件概率 P(x_i | x_{<i})
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n_samples = x.shape[0]
        probs = np.zeros((n_samples, self.input_dim, self.num_classes), dtype=float)
        
        for i in range(self.input_dim):
            if i == 0:
                h = np.zeros((n_samples, self.hidden_dim))
            else:
                # 隐藏层计算
                linear = np.dot(x[:, :i], self.W[:i, :]) + self.c
                h = sigmoid(linear)
            
            # 输出层计算
            start_idx = i * self.num_classes
            end_idx = start_idx + self.num_classes
            linear_out = np.dot(h, self.V[:, start_idx:end_idx]) + self.b[start_idx:end_idx]
            
            # 使用softmax得到分类概率
            probs[:, i, :] = softmax(linear_out)
        
        return probs
    
    def sample(self, n_samples=1):
        """
        从模型采样
        
        参数:
            n_samples: 采样数量
        
        返回:
            采样结果（每个元素是类别索引）
        """
        samples = np.zeros((n_samples, self.input_dim), dtype=int)
        
        for i in range(self.input_dim):
            if i == 0:
                h = np.zeros((n_samples, self.hidden_dim))
            else:
                linear = np.dot(samples[:, :i], self.W[:i, :]) + self.c
                h = sigmoid(linear)
            
            # 输出层计算
            start_idx = i * self.num_classes
            end_idx = start_idx + self.num_classes
            linear_out = np.dot(h, self.V[:, start_idx:end_idx]) + self.b[start_idx:end_idx]
            
            # 计算类别概率
            class_probs = softmax(linear_out)
            
            # 采样类别
            for j in range(n_samples):
                samples[j, i] = np.random.choice(self.num_classes, p=class_probs[j])
        
        return samples


class RNADE:
    """
    实值NADE（Real-valued NADE），使用高斯混合模型
    """
    def __init__(self, input_dim, hidden_dim, num_gaussians=5):
        """
        初始化RNADE
        
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_gaussians: 每个特征的高斯分布数量
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        
        # 权重共享
        self.W = np.random.randn(input_dim, hidden_dim) * 0.01
        self.c = np.random.randn(hidden_dim) * 0.01
        
        # 高斯混合模型参数
        # 均值
        self.mu = np.random.randn(hidden_dim, input_dim * num_gaussians) * 0.01
        self.mu_b = np.random.randn(input_dim * num_gaussians) * 0.01
        
        # 方差（使用对数确保非负）
        self.log_var = np.random.randn(hidden_dim, input_dim * num_gaussians) * 0.01
        self.log_var_b = np.random.randn(input_dim * num_gaussians) * 0.01
    
    def forward(self, x):
        """
        前向传播，计算高斯混合模型参数
        
        参数:
            x: 输入向量或矩阵
        
        返回:
            所有高斯混合模型参数
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        n_samples = x.shape[0]
        mus = np.zeros((n_samples, self.input_dim, self.num_gaussians))
        log_vars = np.zeros((n_samples, self.input_dim, self.num_gaussians))
        
        for i in range(self.input_dim):
            if i == 0:
                h = np.zeros((n_samples, self.hidden_dim))
            else:
                linear = np.dot(x[:, :i], self.W[:i, :]) + self.c
                h = sigmoid(linear)
            
            # 计算均值
            start_idx = i * self.num_gaussians
            end_idx = start_idx + self.num_gaussians
            mus[:, i, :] = np.dot(h, self.mu[:, start_idx:end_idx]) + self.mu_b[start_idx:end_idx]
            
            # 计算对数方差
            log_vars[:, i, :] = np.dot(h, self.log_var[:, start_idx:end_idx]) + self.log_var_b[start_idx:end_idx]
        
        return mus, np.exp(log_vars)
    
    def sample(self, n_samples=1):
        """
        从模型采样
        
        参数:
            n_samples: 采样数量
        
        返回:
            采样结果
        """
        samples = np.zeros((n_samples, self.input_dim))
        
        for i in range(self.input_dim):
            if i == 0:
                h = np.zeros((n_samples, self.hidden_dim))
            else:
                linear = np.dot(samples[:, :i], self.W[:i, :]) + self.c
                h = sigmoid(linear)
            
            # 计算高斯混合模型参数
            start_idx = i * self.num_gaussians
            end_idx = start_idx + self.num_gaussians
            
            mus = np.dot(h, self.mu[:, start_idx:end_idx]) + self.mu_b[start_idx:end_idx]
            log_vars = np.dot(h, self.log_var[:, start_idx:end_idx]) + self.log_var_b[start_idx:end_idx]
            vars = np.exp(log_vars)
            
            # 采样高斯分量
            component_idx = np.random.choice(self.num_gaussians, size=n_samples)
            
            # 从选中的高斯分量采样
            for j in range(n_samples):
                mu = mus[j, component_idx[j]]
                var = vars[j, component_idx[j]]
                samples[j, i] = np.random.normal(mu, np.sqrt(var))
        
        return samples


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


def plot_naive_samples(samples, filename="figure/naive_samples.png"):
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
def fvsbn_demo():
    """
    FVSBN演示
    """
    print("\n=== FVSBN演示 ===")
    
    # 加载二值化MNIST数据
    x_train, x_test = binarize_mnist()
    
    # 创建FVSBN模型
    fvsbn = FVSBN(n_features=784)
    
    # 采样
    samples = fvsbn.sample(10)
    
    # 绘制采样结果
    plot_naive_samples(samples, "figure/fvsbn_samples.png")


def nade_demo():
    """
    NADE演示
    """
    print("\n=== NADE演示 ===")
    
    # 创建NADE模型
    nade = NADE(input_dim=10, hidden_dim=5)
    
    # 启用权重共享
    nade.enable_weight_sharing()
    
    # 生成随机数据
    x_train = np.random.binomial(1, 0.5, size=(100, 10))
    
    # 测试前向传播
    probs = nade.forward(x_train)
    print(f"NADE前向传播输出形状: {probs.shape}")
    
    # 采样
    samples = nade.sample(5)
    print(f"NADE采样结果: {samples}")


def general_discrete_nade_demo():
    """
    一般离散分布NADE演示
    """
    print("\n=== 一般离散分布NADE演示 ===")
    
    # 创建模型（每个特征有3个类别）
    gdnade = GeneralDiscreteNADE(input_dim=5, hidden_dim=3, num_classes=3)
    
    # 生成随机数据
    x_train = np.random.choice(3, size=(100, 5))
    
    # 测试前向传播
    probs = gdnade.forward(x_train)
    print(f"一般离散NADE前向传播输出形状: {probs.shape}")
    
    # 采样
    samples = gdnade.sample(5)
    print(f"一般离散NADE采样结果: {samples}")


def rnade_demo():
    """
    RNADE演示
    """
    print("\n=== RNADE演示 ===")
    
    # 创建RNADE模型
    rnade = RNADE(input_dim=10, hidden_dim=5, num_gaussians=3)
    
    # 生成随机实值数据
    x_train = np.random.normal(0, 1, size=(100, 10))
    
    # 测试前向传播
    mus, vars = rnade.forward(x_train)
    print(f"RNADE前向传播均值输出形状: {mus.shape}")
    print(f"RNADE前向传播方差输出形状: {vars.shape}")
    
    # 采样
    samples = rnade.sample(5)
    print(f"RNADE采样结果形状: {samples.shape}")
    print(f"RNADE采样结果: {samples}")


def weight_sharing_demonstration():
    """
    权重共享演示
    """
    print("\n=== 权重共享演示 ===")
    
    # 创建两个NADE模型
    input_dim = 100
    hidden_dim = 50
    
    nade_no_share = NADE(input_dim, hidden_dim)
    nade_share = NADE(input_dim, hidden_dim)
    nade_share.enable_weight_sharing()
    
    # 计算参数数量
    params_no_share = (input_dim * hidden_dim * input_dim) + (input_dim * hidden_dim) + \
                     (input_dim * hidden_dim) + input_dim
    
    params_share = (input_dim * hidden_dim) + (hidden_dim * input_dim) + hidden_dim + input_dim
    
    print(f"不共享权重的参数数量: {params_no_share}")
    print(f"共享权重的参数数量: {params_share}")
    print(f"参数减少比例: {100 * (params_no_share - params_share) / params_no_share:.2f}%")


if __name__ == "__main__":
    print("===== 自回归模型的具体实现 =====")
    
    # 演示FVSBN
    # fvsbn_demo()  # 注意：在实际运行时会消耗较多内存
    
    # 演示NADE
    nade_demo()
    
    # 演示一般离散分布NADE
    general_discrete_nade_demo()
    
    # 演示RNADE
    rnade_demo()
    
    # 演示权重共享
    weight_sharing_demonstration()
    
    print("\n所有演示完成！")
