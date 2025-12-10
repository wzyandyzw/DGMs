# 知识点13：潜在狄利克雷分配（LDA）
"""
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
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.special import gamma, digamma

class LatentDirichletAllocation:
    """潜在狄利克雷分配（LDA）的实现"""
    
    def __init__(self, vocab_size, num_topics, alpha=0.1, beta=0.1):
        """初始化LDA模型
        
        参数:
        vocab_size: 词汇表大小
        num_topics: 主题数量
        alpha: 狄利克雷先验参数（文档-主题分布）
        beta: 狄利克雷先验参数（主题-单词分布）
        """
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        
        # 模型参数
        self.document_topic_probs = None  # 文档-主题分布，shape=(num_docs, num_topics)
        self.topic_word_probs = None  # 主题-单词分布，shape=(num_topics, vocab_size)
    
    def initialize_parameters(self, num_docs):
        """初始化模型参数
        
        参数:
        num_docs: 文档数量
        """
        # 1. 文档-主题分布（狄利克雷先验）
        self.document_topic_probs = np.random.dirichlet(np.ones(self.num_topics) * self.alpha, size=num_docs)
        
        # 2. 主题-单词分布（狄利克雷先验）
        self.topic_word_probs = np.random.dirichlet(np.ones(self.vocab_size) * self.beta, size=self.num_topics)
        
        print(f"初始化LDA模型参数：")
        print(f"   词汇表大小：{self.vocab_size}")
        print(f"   主题数量：{self.num_topics}")
        print(f"   狄利克雷先验alpha：{self.alpha}")
        print(f"   狄利克雷先验beta：{self.beta}")
        print(f"   文档数量：{num_docs}")
    
    def generate_document(self, doc_length):
        """生成一篇文档
        
        参数:
        doc_length: 文档长度（单词数）
        
        返回:
        document: 文档的单词列表
        topic_assignments: 每个单词的主题分配
        """
        # 1. 从狄利克雷分布中抽取主题混合
        topic_mixture = np.random.dirichlet(np.ones(self.num_topics) * self.alpha, size=1)[0]
        
        document = []
        topic_assignments = []
        
        # 2. 生成每个单词
        for _ in range(doc_length):
            # a. 从主题混合中抽取主题
            topic = np.random.choice(self.num_topics, p=topic_mixture)
            
            # b. 从主题-单词分布中抽取单词
            word = np.random.choice(self.vocab_size, p=self.topic_word_probs[topic])
            
            document.append(word)
            topic_assignments.append(topic)
        
        return document, topic_assignments
    
    def generate_corpus(self, num_docs, avg_doc_length):
        """生成语料库
        
        参数:
        num_docs: 文档数量
        avg_doc_length: 平均文档长度
        
        返回:
        corpus: 语料库，每个元素是一篇文档的单词列表
        topic_assignments: 每个文档中每个单词的主题分配
        """
        corpus = []
        all_topic_assignments = []
        
        for i in range(num_docs):
            # 随机确定文档长度（泊松分布）
            doc_length = np.random.poisson(avg_doc_length)
            doc_length = max(1, doc_length)  # 至少1个单词
            
            # 生成文档
            document, doc_topic_assignments = self.generate_document(doc_length)
            
            corpus.append(document)
            all_topic_assignments.append(doc_topic_assignments)
        
        print(f"\n生成的语料库：")
        print(f"   文档数量：{num_docs}")
        print(f"   平均文档长度：{avg_doc_length}")
        
        return corpus, all_topic_assignments
    
    def gibbs_sampling(self, corpus, num_iterations=100):
        """使用吉布斯采样学习LDA模型
        
        参数:
        corpus: 语料库
        num_iterations: 吉布斯采样迭代次数
        
        返回:
        document_topic_counts: 文档-主题计数，shape=(num_docs, num_topics)
        topic_word_counts: 主题-单词计数，shape=(num_topics, vocab_size)
        topic_counts: 主题计数，shape=(num_topics,)
        """
        num_docs = len(corpus)
        
        # 初始化计数
        document_topic_counts = np.zeros((num_docs, self.num_topics))
        topic_word_counts = np.zeros((self.num_topics, self.vocab_size))
        topic_counts = np.zeros(self.num_topics)
        
        # 初始化每个单词的主题分配
        all_topic_assignments = []
        
        for i in range(num_docs):
            doc = corpus[i]
            doc_topic_assignments = []
            
            for word in doc:
                # 随机分配主题
                topic = np.random.choice(self.num_topics)
                doc_topic_assignments.append(topic)
                
                # 更新计数
                document_topic_counts[i, topic] += 1
                topic_word_counts[topic, word] += 1
                topic_counts[topic] += 1
            
            all_topic_assignments.append(doc_topic_assignments)
        
        # 吉布斯采样迭代
        for iteration in range(num_iterations):
            for i in range(num_docs):
                doc = corpus[i]
                
                for n in range(len(doc)):
                    word = doc[n]
                    topic = all_topic_assignments[i][n]
                    
                    # 减少计数
                    document_topic_counts[i, topic] -= 1
                    topic_word_counts[topic, word] -= 1
                    topic_counts[topic] -= 1
                    
                    # 计算每个主题的条件概率
                    topic_probs = np.zeros(self.num_topics)
                    
                    for k in range(self.num_topics):
                        term1 = (document_topic_counts[i, k] + self.alpha) / (sum(document_topic_counts[i, :]) + self.num_topics * self.alpha)
                        term2 = (topic_word_counts[k, word] + self.beta) / (topic_counts[k] + self.vocab_size * self.beta)
                        topic_probs[k] = term1 * term2
                    
                    # 归一化
                    topic_probs /= np.sum(topic_probs)
                    
                    # 采样新的主题
                    new_topic = np.random.choice(self.num_topics, p=topic_probs)
                    
                    # 更新主题分配
                    all_topic_assignments[i][n] = new_topic
                    
                    # 更新计数
                    document_topic_counts[i, new_topic] += 1
                    topic_word_counts[new_topic, word] += 1
                    topic_counts[new_topic] += 1
            
            if (iteration + 1) % 10 == 0:
                print(f"吉布斯采样迭代 {iteration+1}/{num_iterations}")
        
        # 估计文档-主题分布和主题-单词分布
        self.document_topic_probs = (document_topic_counts + self.alpha) / (np.sum(document_topic_counts, axis=1, keepdims=True) + self.num_topics * self.alpha)
        self.topic_word_probs = (topic_word_counts + self.beta) / (np.sum(topic_word_counts, axis=1, keepdims=True) + self.vocab_size * self.beta)
        
        return document_topic_counts, topic_word_counts, topic_counts
    
    def get_topic_words(self, topic_id, num_words=10):
        """获取主题的前num_words个单词
        
        参数:
        topic_id: 主题ID
        num_words: 要返回的单词数量
        
        返回:
        top_words: 主题的前num_words个单词的索引
        word_probs: 这些单词的概率
        """
        # 获取主题的单词分布
        word_probs = self.topic_word_probs[topic_id]
        
        # 获取概率最高的单词
        top_indices = np.argsort(word_probs)[::-1][:num_words]
        top_probs = word_probs[top_indices]
        
        return top_indices, top_probs
    
    def plot_topic_distributions(self):
        """绘制主题分布和主题-单词分布"""
        plt.figure(figsize=(12, 8))
        
        # 1. 主题-单词分布（前几个主题）
        for k in range(min(3, self.num_topics)):
            plt.subplot(2, 2, k+1)
            top_words, word_probs = self.get_topic_words(k, num_words=15)
            plt.bar(range(len(top_words)), word_probs)
            plt.title(f'主题 {k+1} 的单词分布')
            plt.xlabel('单词索引')
            plt.ylabel('概率')
            plt.xticks(range(len(top_words)), top_words, rotation=90)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../figure/lda_topic_distributions.png')
        plt.close()
    
    def plot_document_topic_distributions(self, document_indices):
        """绘制文档的主题分布
        
        参数:
        document_indices: 要绘制的文档索引列表
        """
        num_docs_to_plot = len(document_indices)
        
        plt.figure(figsize=(12, 3*num_docs_to_plot))
        
        for i, doc_idx in enumerate(document_indices):
            plt.subplot(num_docs_to_plot, 1, i+1)
            
            topic_probs = self.document_topic_probs[doc_idx]
            plt.bar(range(self.num_topics), topic_probs)
            plt.title(f'文档 {doc_idx+1} 的主题分布')
            plt.xlabel('主题')
            plt.ylabel('概率')
            plt.xticks(range(self.num_topics))
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../figure/lda_document_topic_distributions.png')
        plt.close()

# 计算狄利克雷分布的密度
def dirichlet_density(theta, alpha):
    """计算狄利克雷分布的密度
    
    参数:
    theta: 狄利克雷分布的参数，shape=(num_topics,)
    alpha: 狄利克雷分布的先验参数，shape=(num_topics,)
    
    返回:
    density: 狄利克雷分布的密度
    """
    alpha_sum = np.sum(alpha)
    numerator = gamma(alpha_sum) * np.prod(theta ** (alpha - 1))
    denominator = np.prod(gamma(alpha))
    
    return numerator / denominator

# 狄利克雷分布示例
def dirichlet_example():
    """狄利克雷分布示例"""
    print("=== 狄利克雷分布示例 ===")
    
    # 2个主题的狄利克雷分布
    alpha_values = [
        np.array([0.1, 0.1]),   # 稀疏分布
        np.array([1.0, 1.0]),   # 均匀分布
        np.array([5.0, 5.0]),   # 集中分布
        np.array([2.0, 0.5])    # 偏斜分布
    ]
    
    plt.figure(figsize=(12, 8))
    
    for i, alpha in enumerate(alpha_values):
        plt.subplot(2, 2, i+1)
        
        # 生成网格点
        theta1 = np.linspace(0.01, 0.99, 100)
        theta2 = 1 - theta1
        
        # 计算密度
        density = np.zeros_like(theta1)
        for j in range(len(theta1)):
            theta = np.array([theta1[j], theta2[j]])
            density[j] = dirichlet_density(theta, alpha)
        
        plt.plot(theta1, density)
        plt.title(f'狄利克雷分布，alpha={alpha}')
        plt.xlabel('theta1')
        plt.ylabel('密度')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figure/dirichlet_distributions.png')
    plt.close()

# LDA文档生成示例
def lda_document_generation_example():
    """LDA文档生成示例"""
    print("=== LDA文档生成示例 ===")
    
    # 创建LDA模型
    vocab_size = 100  # 词汇表大小
    num_topics = 3    # 主题数量
    
    lda = LatentDirichletAllocation(vocab_size, num_topics, alpha=0.1, beta=0.1)
    
    # 手动设置主题-单词分布（使每个主题有明显的特征）
    lda.topic_word_probs = np.zeros((num_topics, vocab_size))
    
    # 主题0：科技主题，包含单词0-33的概率较高
    lda.topic_word_probs[0, :34] = 0.9 / 34  # 单词0-33
    lda.topic_word_probs[0, 34:] = 0.1 / 66  # 单词34-99
    
    # 主题1：体育主题，包含单词34-66的概率较高
    lda.topic_word_probs[1, :34] = 0.1 / 34  # 单词0-33
    lda.topic_word_probs[1, 34:67] = 0.9 / 33  # 单词34-66
    lda.topic_word_probs[1, 67:] = 0.1 / 33  # 单词67-99
    
    # 主题2：艺术主题，包含单词67-99的概率较高
    lda.topic_word_probs[2, :67] = 0.1 / 67  # 单词0-66
    lda.topic_word_probs[2, 67:] = 0.9 / 33  # 单词67-99
    
    # 生成语料库
    num_docs = 50
    avg_doc_length = 10
    
    corpus, all_topic_assignments = lda.generate_corpus(num_docs, avg_doc_length)
    
    # 学习模型参数
    print(f"\n使用吉布斯采样学习LDA模型...")
    document_topic_counts, topic_word_counts, topic_counts = lda.gibbs_sampling(corpus, num_iterations=50)
    
    # 绘制结果
    lda.plot_topic_distributions()
    lda.plot_document_topic_distributions([0, 1, 2])
    
    # 显示主题示例
    print(f"\n主题示例：")
    for k in range(num_topics):
        top_words, word_probs = lda.get_topic_words(k, num_words=5)
        print(f"   主题 {k+1} 的前5个单词：{top_words}, 概率：{word_probs}")

# 主函数
def main():
    print("=== 潜在狄利克雷分配（LDA）演示 ===")
    
    # 运行狄利克雷分布示例
    dirichlet_example()
    
    # 运行LDA文档生成示例
    lda_document_generation_example()
    
    print("\nLDA演示完成！")
    print("\n要点总结:")
    print("- LDA是一种混合成员文档模型，每篇文档可以包含多个主题")
    print("- LDA使用狄利克雷分布作为先验来建模文档-主题分布和主题-单词分布")
    print("- 文档生成过程：先抽取主题混合，然后为每个单词分配主题并生成单词")
    print("- 可以使用吉布斯采样来学习LDA模型的参数")
    print("- LDA可以自动发现文档集合中的潜在主题结构")
    print("- LDA广泛应用于文本分析、信息检索、推荐系统等领域")

if __name__ == "__main__":
    main()
