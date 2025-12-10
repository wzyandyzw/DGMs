# 知识点4：K-means相关概念实现
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# K-means作为自编码器示例
class KMeansAutoencoder:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.centroids = None
    
    def fit(self, X):
        """训练K-means模型"""
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(X)
        self.centroids = self.kmeans.cluster_centers_
    
    def encode(self, X):
        """编码：将输入映射到one-hot向量"""
        labels = self.kmeans.predict(X)
        # 转换为one-hot编码
        one_hot = np.zeros((X.shape[0], self.n_clusters))
        one_hot[np.arange(X.shape[0]), labels] = 1
        return one_hot
    
    def decode(self, one_hot):
        """解码：将one-hot向量映射到质心"""
        # 找到每个样本对应的质心索引
        labels = np.argmax(one_hot, axis=1)
        # 映射到质心
        return self.centroids[labels]
    
    def reconstruct(self, X):
        """重构：encode + decode"""
        one_hot = self.encode(X)
        return self.decode(one_hot)
    
    def generate(self, num_samples=10):
        """生成：随机采样one-hot向量并解码"""
        # 随机采样簇标签
        labels = np.random.choice(self.n_clusters, num_samples)
        # 转换为one-hot编码
        one_hot = np.zeros((num_samples, self.n_clusters))
        one_hot[np.arange(num_samples), labels] = 1
        # 解码生成样本
        return self.decode(one_hot)

# 主函数
def main():
    # 生成示例数据
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
    
    # 初始化并训练K-means自编码器
    kmeans_ae = KMeansAutoencoder(n_clusters=3)
    kmeans_ae.fit(X)
    
    # 1. K-means作为EM算法的示例
    print("K-means作为EM算法的示例：")
    print("质心（E-step计算责任，M-step更新质心）：")
    print(kmeans_ae.centroids)
    
    # 2. K-means作为自编码器
    print("\nK-means作为自编码器：")
    # 编码几个样本
    sample_indices = [0, 1, 2]
    samples = X[sample_indices]
    one_hot = kmeans_ae.encode(samples)
    print("输入样本:")
    print(samples)
    print("One-hot编码:")
    print(one_hot)
    
    # 解码回原空间
    decoded = kmeans_ae.decode(one_hot)
    print("解码后的样本（质心）:")
    print(decoded)
    
    # 3. K-means作为生成模型
    print("\nK-means作为生成模型：")
    # 生成样本
    generated = kmeans_ae.generate(num_samples=5)
    print("生成的样本（随机质心）:")
    print(generated)
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    # 原始数据和质心
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.scatter(kmeans_ae.centroids[:, 0], kmeans_ae.centroids[:, 1], 
               c='red', marker='X', s=200, label='质心')
    plt.title('原始数据和K-means质心')
    plt.legend()
    
    # 重构和生成
    plt.subplot(1, 2, 2)
    # 绘制原始数据
    plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.3, label='原始数据')
    # 绘制重构数据
    reconstructed = kmeans_ae.reconstruct(X)
    plt.scatter(reconstructed[:, 0], reconstructed[:, 1], c='green', 
               alpha=0.6, label='重构数据')
    # 绘制生成数据
    generated = kmeans_ae.generate(num_samples=100)
    plt.scatter(generated[:, 0], generated[:, 1], c='red', 
               alpha=0.8, marker='*', s=100, label='生成数据')
    plt.title('K-means自编码器：重构与生成')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()