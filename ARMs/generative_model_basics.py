#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识点1：生成模型基础与概率分布

这个文件包含生成模型的基础概念和基本概率分布的实现，包括：
1. 生成模型的基本概念
2. 伯努利分布
3. 分类分布
4. 联合分布与条件独立性
5. 链式法则与贝叶斯法则
"""

import numpy as np
import matplotlib.pyplot as plt

class BernoulliDistribution:
    """
    伯努利分布类，用于二值随机变量
    """
    def __init__(self, p):
        """
        初始化伯努利分布
        
        参数:
            p: 成功概率 (0 <= p <= 1)
        """
        if not 0 <= p <= 1:
            raise ValueError("概率p必须在[0, 1]范围内")
        self.p = p
    
    def pmf(self, x):
        """
        计算概率质量函数
        
        参数:
            x: 随机变量值 (0 或 1)
        
        返回:
            概率值
        """
        if x not in [0, 1]:
            raise ValueError("伯努利分布的输入必须是0或1")
        return self.p ** x * (1 - self.p) ** (1 - x)
    
    def sample(self, size=1):
        """
        从伯努利分布采样
        
        参数:
            size: 采样数量
        
        返回:
            采样结果数组
        """
        return np.random.binomial(1, self.p, size)
    
    def mean(self):
        """
        计算均值
        """
        return self.p
    
    def variance(self):
        """
        计算方差
        """
        return self.p * (1 - self.p)


class CategoricalDistribution:
    """
    分类分布类，用于多类别随机变量
    """
    def __init__(self, p):
        """
        初始化分类分布
        
        参数:
            p: 概率数组，必须满足所有元素非负且和为1
        """
        p = np.array(p)
        if not np.all(p >= 0):
            raise ValueError("概率必须非负")
        if not np.isclose(np.sum(p), 1.0):
            raise ValueError("概率和必须为1")
        self.p = p
        self.K = len(p)  # 类别数
    
    def pmf(self, x):
        """
        计算概率质量函数
        
        参数:
            x: 随机变量值 (0 到 K-1)
        
        返回:
            概率值
        """
        if x < 0 or x >= self.K:
            raise ValueError(f"分类分布的输入必须在[0, {self.K-1}]范围内")
        return self.p[x]
    
    def sample(self, size=1):
        """
        从分类分布采样
        
        参数:
            size: 采样数量
        
        返回:
            采样结果数组
        """
        return np.random.choice(self.K, size, p=self.p)
    
    def mean(self):
        """
        计算均值
        """
        return np.sum(np.arange(self.K) * self.p)
    
    def variance(self):
        """
        计算方差
        """
        mean = self.mean()
        return np.sum((np.arange(self.K) - mean) ** 2 * self.p)


def joint_distribution_example():
    """
    联合分布示例：两个独立伯努利变量的联合分布
    """
    # 创建两个独立的伯努利分布
    bern1 = BernoulliDistribution(0.6)
    bern2 = BernoulliDistribution(0.3)
    
    print("两个独立伯努利变量的联合分布:")
    print("x1 | x2 | P(x1, x2)")
    print("--------------------")
    
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            # 由于独立性，联合概率等于边缘概率的乘积
            p_joint = bern1.pmf(x1) * bern2.pmf(x2)
            print(f"{x1}  | {x2}  | {p_joint:.4f}")


def chain_rule_example():
    """
    链式法则示例
    """
    # 假设有三个变量x1, x2, x3，其中x2依赖x1，x3依赖x1和x2
    print("\n链式法则示例:")
    
    # P(x1)
    p_x1 = {0: 0.6, 1: 0.4}
    
    # P(x2 | x1)
    p_x2_given_x1 = {
        0: {0: 0.7, 1: 0.3},
        1: {0: 0.2, 1: 0.8}
    }
    
    # P(x3 | x1, x2)
    p_x3_given_x1x2 = {
        (0, 0): {0: 0.9, 1: 0.1},
        (0, 1): {0: 0.6, 1: 0.4},
        (1, 0): {0: 0.5, 1: 0.5},
        (1, 1): {0: 0.3, 1: 0.7}
    }
    
    # 计算联合概率P(x1=0, x2=1, x3=1)
    p_joint = p_x1[0] * p_x2_given_x1[0][1] * p_x3_given_x1x2[(0, 1)][1]
    print(f"P(x1=0, x2=1, x3=1) = P(x1=0) * P(x2=1|x1=0) * P(x3=1|x1=0,x2=1)")
    print(f"                      = {p_x1[0]} * {p_x2_given_x1[0][1]} * {p_x3_given_x1x2[(0, 1)][1]}")
    print(f"                      = {p_joint:.4f}")


def bayes_rule_example():
    """
    贝叶斯法则示例
    """
    print("\n贝叶斯法则示例:")
    
    # 假设有两个类别：垃圾邮件(1)和正常邮件(0)
    # 先验概率
    p_spam = 0.2
    p_not_spam = 1 - p_spam
    
    # 条件概率：邮件包含单词"免费"的概率
    p_free_given_spam = 0.9
    p_free_given_not_spam = 0.1
    
    # 计算后验概率：包含"免费"的邮件是垃圾邮件的概率
    p_spam_given_free = (p_free_given_spam * p_spam) / \
                       (p_free_given_spam * p_spam + p_free_given_not_spam * p_not_spam)
    
    print(f"先验概率 P(垃圾邮件) = {p_spam:.2f}")
    print(f"条件概率 P(包含'免费'|垃圾邮件) = {p_free_given_spam:.2f}")
    print(f"条件概率 P(包含'免费'|正常邮件) = {p_free_given_not_spam:.2f}")
    print(f"后验概率 P(垃圾邮件|包含'免费') = {p_spam_given_free:.4f}")


def visualize_distributions():
    """
    可视化伯努利分布和分类分布
    """
    # 伯努利分布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    p_values = np.linspace(0, 1, 100)
    means = [BernoulliDistribution(p).mean() for p in p_values]
    variances = [BernoulliDistribution(p).variance() for p in p_values]
    
    ax1.plot(p_values, means, label='均值', color='blue')
    ax1.plot(p_values, variances, label='方差', color='red')
    ax1.set_xlabel('成功概率 p')
    ax1.set_ylabel('值')
    ax1.set_title('伯努利分布的均值和方差')
    ax1.legend()
    ax1.grid(True)
    
    # 分类分布示例
    categories = [0, 1, 2, 3]
    probabilities = [0.1, 0.3, 0.4, 0.2]
    cat_dist = CategoricalDistribution(probabilities)
    
    ax2.bar(categories, probabilities, color=['blue', 'green', 'red', 'purple'])
    ax2.set_xlabel('类别')
    ax2.set_ylabel('概率')
    ax2.set_title('分类分布示例')
    ax2.set_xticks(categories)
    
    plt.tight_layout()
    plt.savefig('figure/bernoulli_categorical_distributions.png')
    print("\n分布可视化已保存到 figure/bernoulli_categorical_distributions.png")


if __name__ == "__main__":
    print("===== 生成模型基础与概率分布 =====")
    
    # 测试伯努利分布
    print("\n1. 伯努利分布测试:")
    bern = BernoulliDistribution(0.6)
    print(f"P(X=1) = {bern.pmf(1):.4f}")
    print(f"P(X=0) = {bern.pmf(0):.4f}")
    print(f"采样10次: {bern.sample(10)}")
    
    # 测试分类分布
    print("\n2. 分类分布测试:")
    cat = CategoricalDistribution([0.1, 0.3, 0.4, 0.2])
    print(f"P(X=2) = {cat.pmf(2):.4f}")
    print(f"均值: {cat.mean():.4f}")
    print(f"方差: {cat.variance():.4f}")
    print(f"采样10次: {cat.sample(10)}")
    
    # 展示联合分布
    joint_distribution_example()
    
    # 展示链式法则
    chain_rule_example()
    
    # 展示贝叶斯法则
    bayes_rule_example()
    
    # 可视化分布
    visualize_distributions()
