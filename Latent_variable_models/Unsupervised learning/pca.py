import numpy as np

class PCA:
    def __init__(self, n_components):
        """
        初始化PCA模型
        :param n_components: 保留的主成分数量
        """
        self.n_components = n_components
        self.U_k = None  # 主成分矩阵
        self.mean = None  # 数据均值
    
    def fit(self, X):
        """
        训练PCA模型
        :param X: 输入数据，形状为 (D, N)，其中D是特征维度，N是样本数量
        """
        # 1. 数据中心化
        self.mean = np.mean(X, axis=1, keepdims=True)
        Z = X - self.mean
        
        # 2. 计算协方差矩阵
        covariance_matrix = np.dot(Z, Z.T) / X.shape[1]
        
        # 3. 进行SVD分解
        U, D, V = np.linalg.svd(covariance_matrix)
        
        # 4. 选择前n_components个主成分
        self.U_k = U[:, :self.n_components]
    
    def transform(self, X):
        """
        将数据投影到低维空间
        :param X: 输入数据，形状为 (D, N)
        :return: 投影后的数据，形状为 (n_components, N)
        """
        if self.U_k is None or self.mean is None:
            raise ValueError("PCA模型尚未训练，请先调用fit方法")
        
        # 数据中心化
        Z = X - self.mean
        
        # 降维（编码器）
        X_reduced = np.dot(self.U_k.T, Z)
        return X_reduced
    
    def reconstruct(self, X_reduced):
        """
        将低维数据重建回高维空间
        :param X_reduced: 低维数据，形状为 (n_components, N)
        :return: 重建的高维数据，形状为 (D, N)
        """
        if self.U_k is None or self.mean is None:
            raise ValueError("PCA模型尚未训练，请先调用fit方法")
        
        # 投影（解码器）
        X_reconstructed = np.dot(self.U_k, X_reduced) + self.mean
        return X_reconstructed

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    D = 3  # 特征维度
    N = 100  # 样本数量
    X = np.random.randn(D, N)
    
    # 创建并训练PCA模型
    n_components = 2
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    # 降维
    X_reduced = pca.transform(X)
    print(f"原始数据形状: {X.shape}")
    print(f"降维后数据形状: {X_reduced.shape}")
    
    # 重建
    X_reconstructed = pca.reconstruct(X_reduced)
    print(f"重建后数据形状: {X_reconstructed.shape}")
    
    # 计算重建误差
    reconstruction_error = np.mean(np.sum((X - X_reconstructed) ** 2, axis=0))
    print(f"重建误差: {reconstruction_error:.6f}")