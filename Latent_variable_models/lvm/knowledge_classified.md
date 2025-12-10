# 潜在变量模型知识点分类

## 知识点1：潜在变量模型的基本概念
- 图像等数据中存在许多变异因素（如性别、发色、姿态等），这些因素在未标注时是潜在的
- 潜在变量模型使用潜在变量\( \mathbf{z} \)显式建模这些变异因素
- 只有阴影变量\( \mathbf{x} \)是数据中可观测的（如图像像素值）
- 潜在变量\( \mathbf{z} \)对应高级特征
  - 若\( \mathbf{z} \)选择得当，\( p(\mathbf{x}|\mathbf{z}) \)可能比\( p(\mathbf{x}) \)简单得多
  - 训练后可通过\( p(\mathbf{z}|\mathbf{x}) \)识别特征（如\( p(\text{EyeColor}=\text{Blue}|\mathbf{x}) \)
  - 挑战：手动指定这些条件分布非常困难

## 知识点2：深度潜在变量模型
- 使用神经网络建模条件分布
  - \( z \sim \mathcal{N}(0, \mathbf{I}) \)
  - \( p(\mathbf{x}|z) = \mathcal{N}(\mathbf{x}; \mu_\theta(z), \Sigma_\theta(z)) \)，其中\( \mu_\theta \)和\( \Sigma_\theta \)是神经网络
- 希望训练后\( z \)对应有意义的潜在变异因素
- 特征可通过\( p(z|\mathbf{x}) \)计算

## 知识点3：高斯混合模型（GMM）的基本原理
- 贝叶斯网络结构：\( z \to \mathbf{x} \)
  - \( z \sim \text{Categorical}_{\pi^*}(1, \dots, K) \)
  - \( p(\mathbf{x}|z = k) = \mathcal{N}(\mathbf{x}; \mu_k^*, \Sigma_k^*) \)
- 生成过程：
  - 通过采样\( z \)选择混合成分\( k \)
  - 从该高斯分布采样生成数据点
- 聚类：后验概率\( p(z|\mathbf{x}) \)标识混合成分
- 无监督学习：从无标注数据中学习（不适定问题）

## 知识点4：高斯混合模型的数学表达
- 高斯密度：\( \mathcal{N}(\mathbf{x}; \mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^d \cdot \det(\Sigma)}} \exp\left( -\frac{1}{2}(\mathbf{x} - \mu)^\top \Sigma^{-1} (\mathbf{x} - \mu) \right) \)
- GMM联合分布：\( p(\mathbf{x}) = \sum_z p(\mathbf{x}, z) = \sum_z p(z) p(\mathbf{x}|z) = \sum_{k=1}^K p(z = k) \mathcal{N}(\mathbf{x}; \mu_k^*, \Sigma_k^*) = \sum_{k=1}^K \pi_k^* \mathcal{N}(\mathbf{x}; \mu_k^*, \Sigma_k^*) \)

## 知识点5：概率基本概念澄清
- 概率课程中会仔细区分：
  - 随机变量，如\( X : \Omega \to \mathbb{R}^d \)
  - 样本，如\( x_1, x_2, \dots, x_N \in \mathbb{R}^d \)
- 实际应用中会模糊这些对象的区分
- 课程中不会讨论形式化构造\( \Omega \)

## 知识点6：高斯混合模型的学习
- GMM的底层参数：\( \theta^* = \{\pi^*, \{\mu_k^*\}_{k=1}^K, \{\Sigma_k^*\}_{k=1}^K \} \)
- 最大似然估计（MLE）：
  \[ \hat{\theta} = \arg \max_{\theta} \frac{1}{N} \sum_{i=1}^N \log p_{\theta}(x_i) = \arg \max_{\theta} \frac{1}{N} \sum_{i=1}^N \log \left( \sum_{k=1}^K \pi_k \mathcal{N}(x; \mu_k, \Sigma_k) \right) \]
- 无解析解，是非凸优化问题
- 期望最大化（EM）算法是学习GMM的经典方法

## 知识点7：Jensen不等式
- 对于凹函数\( f \)和\( \alpha \in \Delta^{N-1} := \{\mathbf{v} \in \mathbb{R}^N : v_n \in [0,1] \forall n \in [N], \sum_{n=1}^N v_n = 1\} \)，有：
  \[ f\left( \sum_{n=1}^N \alpha_n x_n \right) \geq \sum_{n=1}^N \alpha_n f(x_n) \]
- 特别地，\( \log(\cdot) \)是凹函数
- 应用Jensen不等式（凹函数）：
  \[ \log\left( \mathbb{E}_{z \sim q(z)} [g(z)] \right) = \log\left( \sum_{z} q(z) g(z) \right) \geq \sum_{z} q(z) \log(g(z)) \]

## 知识点8：变分推断
- 传统机器学习目标：\( \max_{\theta} \sum_{i=1}^N \log\left( \int p_{\theta}(x_i, z) dz \right) \)
- 定理：对数似然可表示为：
  \[ \log p_{\theta}(x) = \max_{q(\cdot|x): q(\cdot|x) \geq 0, \int q(z|x)dz=1} \int q(z|x) \log\left( \frac{p_{\theta}(x, z)}{q(z|x)} \right) dz, \]
  最大化分布由\( p_{\theta}(\cdot|x) \)给出（可通过贝叶斯规则计算）
- 新机器学习目标：\( \max_{\theta} \sum_{i=1}^N \max_{q(\cdot|x_i)} \int q(z|x_i) \log\left( \frac{p_{\theta}(x_i,z)}{q(z|x_i)} \right) dz \)

## 知识点9：期望最大化（EM）算法
- 新机器学习目标：\( \max_\theta \sum_{i=1}^N \max_{q(\cdot|x_i)} \int q(z|x_i) \log \left( \frac{p_\theta(x_i,z)}{q(z|x_i)} \right) dz \)
- EM交替执行两个步骤：
  - E-step：\( q^{(\ell)}(\cdot|x_i) = p_{\theta^{(\ell)}}(\cdot|x_i) \)
  - M-step：\( \theta^{(\ell+1)} = \arg \max_\theta \sum_{i=1}^N \int q^{(\ell)}(z|x_i) \log(p_\theta(x_i,z))dz \)
- 与k-means算法的过程类似

## 知识点10：GMM的EM算法实现
- E-step：计算后验概率
  \[ q^{(\ell)}(z|x_i) = p_{\theta^{(\ell)}}(z|x_i) = \frac{p_{\theta^{(\ell)}}(x_i|z)p_{\theta^{(\ell)}}(z)}{p_{\theta^{(\ell)}}(x_i)} = \frac{\pi_z^{(\ell)} \mathcal{N}(x_i; \mu_z^{(\ell)}, \Sigma_z^{(\ell)})}{\sum_{k=1}^K \pi_k^{(\ell)} \mathcal{N}(x_i; \mu_k^{(\ell)}, \Sigma_k^{(\ell)})} \]
- M-step：更新参数
  \[ \theta^{(\ell+1)} = \arg \max_\theta \sum_{i=1}^N \sum_{k=1}^K q^{(\ell)}(k|x_i) \log(p_\theta(k)p_\theta(x_i|k)) \]

## 知识点11：隐马尔可夫模型（HMM）
- 离散隐藏状态的马尔可夫链\( \mathbf{h}_1 \to \mathbf{h}_2 \to \dots \)，包含\( K \)个可能状态
- 给定时间\( t \)的\( \mathbf{h}_t \)，观测\( x_t \)（\( \mathbb{R}^D \)中的随机向量）与所有其他观测/隐藏状态独立
- 初始状态分布\( \pi^* \in \Delta^{K-1} \)和转移矩阵\( \mathbf{T}^* \in \mathbb{R}^{K \times K} \)：
  \[ \mathbb{P}(\mathbf{h}_{t+1} = s_j | \mathbf{h}_t = s_i) = T_{ij}^* \]
- 观测矩阵\( \mathbf{O}^* \in \mathbb{R}^{D \times K} \)（第\( j \)列是\( \mathbb{E}[x_t | \mathbf{h}_t = s_j] \)）
- 实际应用：非侵入式负载监控

## 知识点12：单主题模型
- \( K \)：语料库中不同主题的数量；\( D \)：词汇表中不同单词的数量
- 文档生成过程：文档的主题\( z \)从离散分布（概率向量\( \mathbf{w} \in \Delta^{K-1} \)）中抽取
- 给定\( z \)，\( N \)个单词独立地从离散分布（概率向量\( \boldsymbol{\mu}_z^* \in \Delta^{d-1} \)）中抽取
- 单词表示为\( D \)维向量：\( x_n = \mathbf{e}_i \)当且仅当第\( n \)个单词是词汇表中的第\( i \)个单词
- 模型中：观测\( x_1, \dots, x_N \)（文档单词）给定固定\( z \)（文档主题）

## 知识点13：潜在狄利克雷分配（LDA）
- LDA是混合成员文档模型：每个文档对应**主题混合**（非单个主题）
- 主题混合遵循狄利克雷分布\( \text{Dir}(\boldsymbol{\alpha}^*) \)（参数\( \boldsymbol{\alpha}^* \in \mathbb{R}_{++}^K \)，严格正项）
  - 密度：\( p_{\boldsymbol{\alpha}^*}(\mathbf{h}) = \frac{\Gamma(\alpha_0^*)}{\prod_{k=1}^K \Gamma(\alpha_k^*)} \cdot \prod_{k=1}^K h_k^{\alpha_k^* - 1}, \quad \mathbf{h} \in \Delta^{K-1} \)
  - 其中\( \alpha_0^* = \sum_{k=1}^K \alpha_k^* \)
  - \( \Gamma(s) := \int_0^\infty x^{s-1} e^{-x} dx \)（性质：\( \Gamma(s+1) = s\Gamma(s) \)；对于\( s \in \mathbb{N}_+ \)，\( \Gamma(s+1) = s! \)）
- \( K \)个主题由概率向量\( \boldsymbol{\mu}_1^*, \dots, \boldsymbol{\mu}_K^* \in \Delta^{K-1} \)定义
- 文档生成过程：
  1. 抽取主题混合\( \mathbf{h} = [h_1, \dots, h_K] \sim \text{Dir}(\boldsymbol{\alpha}^*) \)
  2. 对每个单词\( x_n \)：
     - 按\( \mathbf{h} \)抽样主题\( j \)
     - 按\( \boldsymbol{\mu}_j^* \)抽样\( x_n \)
     - 因此\( x_n \)遵循\( \sum_{k=1}^K h_k \boldsymbol{\mu}_k^* \)
