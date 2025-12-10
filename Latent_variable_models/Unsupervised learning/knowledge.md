# Unsupervised learning

## 1. Principle Component Analysis (PCA)

### 1.1 基本概念
PCA是一种常用的无监督学习策略，用于计算向量数据中**最大方差的方向**，广泛应用于降维任务。

### 1.2 算法步骤
对于数据集 $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_N] \in \mathbb{R}^{D \times N}$，以及满足 $k \leq \text{rank}(\mathbf{X})$ 的正整数 $k$，$k$-PCA包含以下两步：

1. **数据中心化**：
   得到中心化数据集 $\mathbf{Z} = [\mathbf{z}_1, \dots, \mathbf{z}_N] \in \mathbb{R}^{D \times N}$，其中：
   $$\mathbf{z}_i = \mathbf{x}_i - \bar{\mathbf{x}}$$
   且 $\bar{\mathbf{x}} = \sum_{j=1}^N \mathbf{x}_j / N$（$\bar{\mathbf{x}}$ 是数据均值）。

2. **SVD分解**：
   计算协方差矩阵 $\bar{\Sigma}_N := \mathbf{Z}\mathbf{Z}^\top / N$ 的（紧凑）SVD分解：
   $$\bar{\Sigma}_N = \mathbf{U}_k \mathbf{D}_k \mathbf{U}_k^\top$$

### 1.3 核心公式

- **降维（编码器）**：将高维数据投影到低维空间
  $$\tilde{\mathbf{x}}_i = \mathbf{U}_k^\top \mathbf{x}_i$$

- **投影（解码器）**：将低维表示重建回高维空间
  $$\hat{\mathbf{x}}_i = \mathbf{U}_k \mathbf{U}_k^\top \mathbf{x}_i$$

---

## 2. k-means Clustering

### 2.1 基本概念
k-means是一种常用的聚类算法，对于数据集 $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_N] \in \mathbb{R}^{D \times N}$，以及满足 $k \leq N$ 的正整数 $k$，目标是将数据集划分为 $k$ 个簇，使同一簇内的数据点尽可能相似，不同簇的数据点尽可能不同。

### 2.2 目标函数
设 $C = \{C_1, \dots, C_k\}$ 是 $[N] := \{1, \dots, N\}$ 的一个划分（即 $C_j$ 是两两不交的索引集，且并集为 $[N]$），$\mathbf{M} = [\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_k] \in \mathbb{R}^{D \times k}$ 是各簇的均值向量。基于 $C$ 和 $\mathbf{M}$ 的**平方和失真度量**定义为：
$$\mathcal{L}(C, \mathbf{M}) := \sum_{j=1}^k \sum_{i \in C_j} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|_2^2$$

### 2.3 迭代算法
k-means的每次迭代包含以下两步：

1. **更新划分** $C$：
   将每个数据点分配到距离其最近的簇中心：
   $$C_j \leftarrow \left\{ i \in [N] : \|\mathbf{x}_i - \boldsymbol{\mu}_j\|_2 = \min_{\ell \in [k]} \|\mathbf{x}_i - \boldsymbol{\mu}_\ell\|_2 \right\}$$

2. **更新均值向量** $\mathbf{M}$：
   重新计算每个簇的中心为该簇内所有数据点的均值：
   $$\boldsymbol{\mu}_j \leftarrow \frac{1}{|C_j|} \sum_{i \in C_j} \mathbf{x}_i$$

