import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=None):
        """
        初始化k-means聚类模型
        :param n_clusters: 簇的数量
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        :param random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None  # 簇中心
        self.labels = None  # 每个样本的簇标签
    
    def _initialize_centroids(self, X):
        """
        初始化簇中心
        :param X: 输入数据，形状为 (D, N)
        """
        D, N = X.shape
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 随机选择n_clusters个样本作为初始簇中心
        indices = np.random.choice(N, self.n_clusters, replace=False)
        self.centroids = X[:, indices]
    
    def _assign_clusters(self, X):
        """
        更新样本的簇分配
        :param X: 输入数据，形状为 (D, N)
        :return: 每个样本的簇标签，形状为 (N,)
        """
        D, N = X.shape
        labels = np.zeros(N, dtype=int)
        
        for i in range(N):
            # 计算当前样本到所有簇中心的距离
            distances = np.linalg.norm(self.centroids - X[:, i:i+1], axis=0)
            # 选择距离最近的簇
            labels[i] = np.argmin(distances)
        
        return labels
    
    def _update_centroids(self, X, labels):
        """
        更新簇中心
        :param X: 输入数据，形状为 (D, N)
        :param labels: 每个样本的簇标签，形状为 (N,)
        :return: 更新后的簇中心，形状为 (D, n_clusters)
        """
        D, N = X.shape
        new_centroids = np.zeros((D, self.n_clusters))
        
        for j in range(self.n_clusters):
            # 找到属于当前簇的所有样本
            cluster_points = X[:, labels == j]
            if cluster_points.shape[1] > 0:
                # 计算簇中心（均值）
                new_centroids[:, j] = np.mean(cluster_points, axis=1)
            else:
                # 如果簇为空，随机选择一个样本作为中心
                new_centroids[:, j] = X[:, np.random.choice(N)]
        
        return new_centroids
    
    def fit(self, X):
        """
        训练k-means聚类模型
        :param X: 输入数据，形状为 (D, N)，其中D是特征维度，N是样本数量
        """
        # 1. 初始化簇中心
        self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            # 2. 更新簇分配
            self.labels = self._assign_clusters(X)
            
            # 3. 更新簇中心
            new_centroids = self._update_centroids(X, self.labels)
            
            # 4. 检查收敛性
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            if centroid_shift < self.tol:
                print(f"k-means算法在第{iteration+1}次迭代时收敛")
                break
            
            self.centroids = new_centroids
        
        # 如果达到最大迭代次数仍未收敛
        if iteration == self.max_iter - 1:
            print(f"k-means算法达到最大迭代次数({self.max_iter})仍未收敛")
    
    def predict(self, X):
        """
        预测新数据的簇标签
        :param X: 输入数据，形状为 (D, N)
        :return: 每个样本的簇标签，形状为 (N,)
        """
        if self.centroids is None:
            raise ValueError("k-means模型尚未训练，请先调用fit方法")
        
        return self._assign_clusters(X)
    
    def compute_inertia(self, X):
        """
        计算平方和失真度量
        :param X: 输入数据，形状为 (D, N)
        :return: 平方和失真值
        """
        if self.centroids is None or self.labels is None:
            raise ValueError("k-means模型尚未训练，请先调用fit方法")
        
        inertia = 0.0
        for j in range(self.n_clusters):
            cluster_points = X[:, self.labels == j]
            if cluster_points.shape[1] > 0:
                # 计算簇内所有点到簇中心的距离平方和
                distances_squared = np.sum((cluster_points - self.centroids[:, j:j+1]) ** 2, axis=0)
                inertia += np.sum(distances_squared)
        
        return inertia

# 示例用法
if __name__ == "__main__":
    # 创建示例数据
    np.random.seed(42)
    D = 2  # 特征维度
    N = 100  # 样本数量
    
    # 创建3个高斯分布的簇
    cluster1 = np.random.randn(D, N//3) + np.array([[5], [5]])
    cluster2 = np.random.randn(D, N//3) + np.array([[0], [0]])
    cluster3 = np.random.randn(D, N//3) + np.array([[5], [0]])
    
    X = np.concatenate([cluster1, cluster2, cluster3], axis=1)
    
    # 创建并训练k-means模型
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    
    # 获取结果
    labels = kmeans.labels
    centroids = kmeans.centroids
    inertia = kmeans.compute_inertia(X)
    
    print(f"簇中心形状: {centroids.shape}")
    print(f"每个样本的簇标签形状: {labels.shape}")
    print(f"平方和失真度量: {inertia:.6f}")
    
    # 预测新数据
    X_new = np.array([[2, 3, 6], [2, -1, 3]])  # 形状为 (D, N_new)
    new_labels = kmeans.predict(X_new)
    print(f"新数据的簇标签: {new_labels}")