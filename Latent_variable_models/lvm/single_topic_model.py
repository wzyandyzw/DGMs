# 知识点12：单主题模型
"""
- \( K \)：语料库中不同主题的数量；\( D \)：词汇表中不同单词的数量
- 文档生成过程：文档的主题\( z \)从离散分布（概率向量\( \mathbf{w} \in \Delta^{K-1} \)）中抽取
- 给定\( z \)，\( N \)个单词独立地从离散分布（概率向量\( oldsymbol{\mu}_z^* \in \Delta^{d-1} \)）中抽取
- 单词表示为\( D \)维向量：\( x_n = \mathbf{e}_i \)当且仅当第\( n \)个单词是词汇表中的第\( i \)个单词
- 模型中：观测\( x_1, \dots, x_N \)（文档单词）给定固定\( z \)（文档主题）
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class SingleTopicModel:
    """单主题模型的实现"""
    
    def __init__(self, vocab_size, num_topics):
        """初始化单主题模型
        
        参数:
        vocab_size: 词汇表大小
        num_topics: 主题数量
        """
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        
        # 模型参数
        self.topic_probs = None  # 主题分布，shape=(num_topics,)
        self.topic_word_probs = None  # 主题-单词分布，shape=(num_topics, vocab_size)
    
    def initialize_parameters(self):
        """初始化模型参数"""
        # 1. 主题分布（Dirichlet先验）
        self.topic_probs = np.random.dirichlet(np.ones(self.num_topics), size=1)[0]
        
        # 2. 主题-单词分布（每个主题的单词分布，Dirichlet先验）
        self.topic_word_probs = np.zeros((self.num_topics, self.vocab_size))
        for k in range(self.num_topics):
            self.topic_word_probs[k] = np.random.dirichlet(np.ones(self.vocab_size), size=1)[0]
        
        print(f"初始化单主题模型参数：")
        print(f"   词汇表大小：{self.vocab_size}")
        print(f"   主题数量：{self.num_topics}")
        print(f"   主题分布：{self.topic_probs}")
    
    def generate_document(self, doc_length):
        """生成一篇文档
        
        参数:
        doc_length: 文档长度（单词数）
        
        返回:
        topic: 文档的主题
        words: 文档的单词（词汇表索引）
        """
        # 1. 从主题分布中抽取主题
        topic = np.random.choice(self.num_topics, p=self.topic_probs)
        
        # 2. 从主题-单词分布中抽取单词
        words = np.random.choice(self.vocab_size, size=doc_length, p=self.topic_word_probs[topic])
        
        return topic, words
    
    def generate_corpus(self, num_docs, avg_doc_length):
        """生成语料库
        
        参数:
        num_docs: 文档数量
        avg_doc_length: 平均文档长度
        
        返回:
        topics: 每个文档的主题，shape=(num_docs,)
        corpus: 语料库，每个元素是一篇文档的单词列表
        """
        topics = []
        corpus = []
        
        for i in range(num_docs):
            # 随机确定文档长度（泊松分布）
            doc_length = np.random.poisson(avg_doc_length)
            doc_length = max(1, doc_length)  # 至少1个单词
            
            # 生成文档
            topic, words = self.generate_document(doc_length)
            
            topics.append(topic)
            corpus.append(words)
        
        print(f"\n生成的语料库：")
        print(f"   文档数量：{num_docs}")
        print(f"   平均文档长度：{avg_doc_length}")
        print(f"   文档主题分布：{np.bincount(topics, minlength=self.num_topics) / num_docs}")
        
        return topics, corpus
    
    def estimate_parameters(self, corpus, topics):
        """从语料库中估计模型参数
        
        参数:
        corpus: 语料库
        topics: 每个文档的主题
        """
        num_docs = len(corpus)
        
        # 1. 估计主题分布
        topic_counts = Counter(topics)
        self.topic_probs = np.zeros(self.num_topics)
        for k in range(self.num_topics):
            self.topic_probs[k] = (topic_counts[k] + 0.1) / (num_docs + self.num_topics * 0.1)  # 添加平滑
        
        # 2. 估计主题-单词分布
        self.topic_word_probs = np.zeros((self.num_topics, self.vocab_size))
        
        for k in range(self.num_topics):
            # 获取主题为k的所有文档
            topic_k_docs = [corpus[i] for i in range(num_docs) if topics[i] == k]
            
            # 计算每个单词的出现次数
            word_counts = Counter()
            for doc in topic_k_docs:
                word_counts.update(doc)
            
            # 计算单词总数
            total_words = sum(word_counts.values())
            
            # 计算主题-单词分布（添加平滑）
            for w in range(self.vocab_size):
                self.topic_word_probs[k, w] = (word_counts[w] + 0.1) / (total_words + self.vocab_size * 0.1)
        
        print(f"\n估计的模型参数：")
        print(f"   主题分布：{self.topic_probs}")
    
    def infer_topic(self, document):
        """推断文档的主题
        
        参数:
        document: 文档的单词列表
        
        返回:
        inferred_topic: 推断的主题
        topic_probs: 每个主题的概率
        """
        # 计算每个主题的概率
        topic_probs = np.zeros(self.num_topics)
        
        for k in range(self.num_topics):
            # 主题的先验概率
            prior = self.topic_probs[k]
            
            # 计算似然 P(document | topic=k)
            likelihood = 1.0
            for word in document:
                likelihood *= self.topic_word_probs[k, word]
            
            # 后验概率（未归一化）
            topic_probs[k] = prior * likelihood
        
        # 归一化
        topic_probs /= np.sum(topic_probs)
        
        # 选择概率最大的主题
        inferred_topic = np.argmax(topic_probs)
        
        return inferred_topic, topic_probs
    
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
        
        # 1. 主题分布
        plt.subplot(2, 2, 1)
        plt.bar(range(self.num_topics), self.topic_probs)
        plt.title('主题分布')
        plt.xlabel('主题')
        plt.ylabel('概率')
        plt.xticks(range(self.num_topics))
        plt.grid(True, alpha=0.3)
        
        # 2. 主题-单词分布（前几个主题）
        for k in range(min(3, self.num_topics)):
            plt.subplot(2, 2, k+2)
            top_words, word_probs = self.get_topic_words(k, num_words=15)
            plt.bar(range(len(top_words)), word_probs)
            plt.title(f'主题 {k+1} 的单词分布')
            plt.xlabel('单词索引')
            plt.ylabel('概率')
            plt.xticks(range(len(top_words)), top_words, rotation=90)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../figure/single_topic_model_distributions.png')
        plt.close()
    
    def evaluate_model(self, corpus, true_topics):
        """评估模型的性能
        
        参数:
        corpus: 语料库
        true_topics: 每个文档的真实主题
        
        返回:
        accuracy: 主题推断准确率
        """
        correct = 0
        total = len(corpus)
        
        for i in range(total):
            document = corpus[i]
            true_topic = true_topics[i]
            
            # 推断主题
            inferred_topic, _ = self.infer_topic(document)
            
            if inferred_topic == true_topic:
                correct += 1
        
        accuracy = correct / total
        print(f"\n模型评估：")
        print(f"   主题推断准确率：{accuracy:.4f}")
        
        return accuracy

# 模拟文档生成示例
def simulate_document_generation():
    """模拟文档生成过程"""
    print("=== 单主题模型文档生成示例 ===")
    
    # 创建单主题模型
    vocab_size = 100  # 词汇表大小
    num_topics = 3    # 主题数量
    
    stm = SingleTopicModel(vocab_size, num_topics)
    
    # 手动设置主题-单词分布（使每个主题有明显的特征）
    stm.topic_probs = np.array([0.3, 0.4, 0.3])  # 主题分布
    
    # 主题0：科技主题，包含单词0-33的概率较高
    stm.topic_word_probs = np.zeros((num_topics, vocab_size))
    stm.topic_word_probs[0, :34] = 0.9 / 34  # 单词0-33
    stm.topic_word_probs[0, 34:] = 0.1 / 66  # 单词34-99
    
    # 主题1：体育主题，包含单词34-66的概率较高
    stm.topic_word_probs[1, :34] = 0.1 / 34  # 单词0-33
    stm.topic_word_probs[1, 34:67] = 0.9 / 33  # 单词34-66
    stm.topic_word_probs[1, 67:] = 0.1 / 33  # 单词67-99
    
    # 主题2：艺术主题，包含单词67-99的概率较高
    stm.topic_word_probs[2, :67] = 0.1 / 67  # 单词0-66
    stm.topic_word_probs[2, 67:] = 0.9 / 33  # 单词67-99
    
    print(f"\n手动设置的主题-单词分布：")
    print(f"   主题0（科技）：单词0-33的概率较高")
    print(f"   主题1（体育）：单词34-66的概率较高")
    print(f"   主题2（艺术）：单词67-99的概率较高")
    
    # 生成语料库
    num_docs = 50
    avg_doc_length = 10
    
    true_topics, corpus = stm.generate_corpus(num_docs, avg_doc_length)
    
    # 评估模型
    stm.evaluate_model(corpus, true_topics)
    
    # 推断文档主题示例
    print(f"\n文档主题推断示例：")
    for i in range(5):
        document = corpus[i]
        true_topic = true_topics[i]
        inferred_topic, topic_probs = stm.infer_topic(document)
        
        print(f"   文档 {i+1}: 真实主题={true_topic}, 推断主题={inferred_topic}, 主题概率={topic_probs}")
    
    # 绘制结果
    stm.plot_topic_distributions()

# 主函数
def main():
    print("=== 单主题模型演示 ===")
    
    # 运行模拟文档生成示例
    simulate_document_generation()
    
    print("\n单主题模型演示完成！")
    print("\n要点总结:")
    print("- 单主题模型假设每篇文档只包含一个主题")
    print("- 文档生成过程：先从主题分布中选择一个主题，然后从该主题的单词分布中生成单词")
    print("- 模型参数包括主题分布和主题-单词分布")
    print("- 可以使用最大似然估计或贝叶斯估计从语料库中学习模型参数")
    print("- 可以通过贝叶斯推断确定新文档的主题")
    print("- 单主题模型是更复杂的主题模型（如LDA）的基础")

if __name__ == "__main__":
    main()
