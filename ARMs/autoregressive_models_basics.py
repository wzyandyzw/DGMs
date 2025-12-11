#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识点3：自回归模型基础

这个文件包含自回归模型的基础实现，包括：
1. 自回归模型的基本概念
2. 二值化MNIST示例
3. 链式法则因式分解
4. 基于逻辑回归的自回归模型
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


class AutoregressiveModel:
    """
    自回归模型类，基于逻辑回归实现
    """
    def __init__(self, n_features):
        """
        初始化自回归模型
        
        参数:
            n_features: 特征数量
        """
        self.n_features = n_features
        # 初始化参数：每个特征i有i个权重（对应前i-1个特征）
        self.params = []
        for i in range(n_features):
            # 权重包括偏置项和i个特征权重
            self.params.append(np.random.randn(i + 2) * 0.01)  # [bias, w1, w2, ..., wi]
    
    def conditional_probability(self, x, i):
        """
        计算条件概率 P(x_i | x_1, ..., x_{i-1})
        
        参数:
            x: 输入向量
            i: 要计算的第i个特征的索引（从0开始）
        
        返回:
            x_i=1的条件概率
        """
        if i == 0:
            # 第一个特征，只有偏置项
            return sigmoid(self.params[i][0])
        else:
            # 前i个特征（x_0到x_{i-1}）
            x_prev = x[:i]
            # 权重包括偏置项和i个特征权重
            params_i = self.params[i]
            # 计算线性组合
            linear = params_i[0] + np.dot(params_i[1:i+1], x_prev)
            return sigmoid(linear)
    
    def joint_probability(self, x):
        """
        使用链式法则计算联合概率 P(x_1, ..., x_n)
        
        参数:
            x: 输入向量
        
        返回:
            联合概率值
        """
        prob = 1.0
        for i in range(self.n_features):
            p = self.conditional_probability(x, i)
            # 根据x_i的值选择概率
            prob *= p if x[i] == 1 else (1 - p)
        return prob
    
    def sample(self):
        """
        从自回归模型中采样
        
        返回:
            采样的向量
        """
        x = np.zeros(self.n_features, dtype=int)
        for i in range(self.n_features):
            # 计算当前特征的条件概率
            p = self.conditional_probability(x, i)
            # 采样
            x[i] = np.random.binomial(1, p)
        return x
    
    def compute_loss(self, X):
        """
        计算交叉熵损失
        
        参数:
            X: 训练数据，形状为 (n_samples, n_features)
        
        返回:
            平均损失
        """
        n_samples = X.shape[0]
        loss = 0.0
        
        for x in X:
            for i in range(self.n_features):
                p = self.conditional_probability(x, i)
                # 交叉熵损失
                loss -= x[i] * np.log(p + 1e-8) + (1 - x[i]) * np.log(1 - p + 1e-8)
        
        return loss / n_samples
    
    def train(self, X, learning_rate=0.01, n_iterations=100):
        """
        使用梯度下降训练自回归模型
        
        参数:
            X: 训练数据，形状为 (n_samples, n_features)
            learning_rate: 学习率
            n_iterations: 迭代次数
        """
        n_samples = X.shape[0]
        
        for iteration in range(n_iterations):
            # 初始化梯度
            gradients = [np.zeros_like(params_i) for params_i in self.params]
            
            # 计算梯度
            for x in X:
                for i in range(self.n_features):
                    p = self.conditional_probability(x, i)
                    error = p - x[i]
                    
                    if i == 0:
                        # 只有偏置项
                        gradients[i][0] += error
                    else:
                        # 偏置项梯度
                        gradients[i][0] += error
                        # 特征权重梯度
                        gradients[i][1:i+1] += error * x[:i]
            
            # 更新参数
            for i in range(self.n_features):
                self.params[i] -= (learning_rate / n_samples) * gradients[i]
            
            # 计算损失
            if (iteration + 1) % 10 == 0:
                loss = self.compute_loss(X)
                print(f"迭代 {iteration+1}/{n_iterations}, 损失: {loss:.4f}")


class BinarizedMNIST:
    """
    二值化MNIST数据集处理类
    """
    def __init__(self):
        """
        加载并二值化MNIST数据集
        """
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        
        # 归一化到[0, 1]
        self.x_train = self.x_train.astype(np.float32) / 255.0
        self.x_test = self.x_test.astype(np.float32) / 255.0
        
        # 二值化（使用0.5作为阈值）
        self.x_train = (self.x_train > 0.5).astype(int)
        self.x_test = (self.x_test > 0.5).astype(int)
    
    def get_flattened_data(self, dataset='train', max_samples=None):
        """
        获取展平的二值化MNIST数据
        
        参数:
            dataset: 'train' 或 'test'
            max_samples: 最大样本数
        
        返回:
            展平的数据，形状为 (n_samples, 784)
        """
        if dataset == 'train':
            data = self.x_train
        else:
            data = self.x_test
        
        # 展平图像 (28x28 -> 784)
        flattened = data.reshape(-1, 28*28)
        
        if max_samples is not None:
            flattened = flattened[:max_samples]
        
        return flattened
    
    def plot_image(self, image_flat):
        """
        绘制展平的MNIST图像
        
        参数:
            image_flat: 展平的图像数据，长度为784
        """
        image_2d = image_flat.reshape(28, 28)
        plt.imshow(image_2d, cmap='binary')
        plt.axis('off')


# 示例函数
def chain_rule_demonstration():
    """
    链式法则示例
    """
    print("\n=== 链式法则演示 ===")
    
    # 创建一个简单的自回归模型（3个特征）
    ar_model = AutoregressiveModel(3)
    
    # 设置简单的参数以便演示
    # P(x1) = sigmoid(0) = 0.5
    ar_model.params[0] = np.array([0.0])
    
    # P(x2 | x1) = sigmoid(0 + 1*x1)
    ar_model.params[1] = np.array([0.0, 1.0])
    
    # P(x3 | x1, x2) = sigmoid(0 + 1*x1 + 1*x2)
    ar_model.params[2] = np.array([0.0, 1.0, 1.0])
    
    # 计算一些条件概率
    x = np.array([1, 0, 1])
    print(f"条件概率:")
    for i in range(3):
        p = ar_model.conditional_probability(x, i)
        print(f"P(x_{i+1}={x[i]} | x_1,...,x_{i}) = {p:.4f}")
    
    # 计算联合概率
    joint_p = ar_model.joint_probability(x)
    print(f"\n联合概率 P(x1={x[0]}, x2={x[1]}, x3={x[2]}) = {joint_p:.4f}")


def binary_mnist_example():
    """
    二值化MNIST示例
    """
    print("\n=== 二值化MNIST示例 ===")
    
    # 加载二值化MNIST数据
    bmnist = BinarizedMNIST()
    
    # 获取展平的数据
    x_train = bmnist.get_flattened_data(dataset='train', max_samples=1000)
    print(f"训练数据形状: {x_train.shape}")
    
    # 绘制一个示例图像
    print("绘制第一个训练图像...")
    plt.figure(figsize=(4, 4))
    bmnist.plot_image(x_train[0])
    plt.savefig('figure/binarized_mnist_example.png')
    print("图像已保存到 figure/binarized_mnist_example.png")


def simple_autoregressive_model_demo():
    """
    简单自回归模型演示
    """
    print("\n=== 简单自回归模型演示 ===")
    
    # 创建一个小的自回归模型（10个特征）
    ar_model = AutoregressiveModel(10)
    
    # 生成一些随机训练数据
    np.random.seed(42)
    X_train = np.random.binomial(1, 0.5, size=(1000, 10))
    
    # 训练模型
    print("训练自回归模型...")
    ar_model.train(X_train, learning_rate=0.1, n_iterations=50)
    
    # 从模型采样
    sample = ar_model.sample()
    print(f"\n从模型采样: {sample}")
    
    # 计算联合概率
    prob = ar_model.joint_probability(sample)
    print(f"采样的联合概率: {prob:.6f}")


def raster_scan_order_demonstration():
    """
    光栅扫描顺序演示（用于MNIST图像）
    """
    print("\n=== 光栅扫描顺序演示 ===")
    
    # 创建一个28x28的索引矩阵，表示光栅扫描顺序
    raster_order = np.arange(28*28).reshape(28, 28)
    
    # 绘制光栅扫描顺序
    plt.figure(figsize=(6, 6))
    plt.imshow(raster_order, cmap='viridis')
    plt.colorbar(label='光栅扫描顺序')
    plt.title('MNIST图像的光栅扫描顺序')
    plt.savefig('figure/raster_scan_order.png')
    print("光栅扫描顺序图已保存到 figure/raster_scan_order.png")


if __name__ == "__main__":
    print("===== 自回归模型基础 =====")
    
    # 链式法则演示
    chain_rule_demonstration()
    
    # 二值化MNIST示例
    binary_mnist_example()
    
    # 简单自回归模型演示
    simple_autoregressive_model_demo()
    
    # 光栅扫描顺序演示
    raster_scan_order_demonstration()
    
    print("\n所有演示完成！")
