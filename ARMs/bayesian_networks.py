#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识点2：贝叶斯网络及其应用

这个文件包含贝叶斯网络的实现和应用，包括：
1. 贝叶斯网络的基本概念
2. 贝叶斯网络的定义（有向无环图）
3. 条件概率分布（CPD）
4. 朴素贝叶斯分类器
5. 逻辑回归模型
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class BayesianNetwork:
    """
    贝叶斯网络类，基于有向无环图（DAG）实现
    """
    def __init__(self):
        """
        初始化贝叶斯网络
        """
        self.nodes = set()  # 所有节点
        self.edges = set()  # 边集合 (parent, child)
        self.cpds = {}  # 条件概率分布字典 {node: cpd}
    
    def add_node(self, node):
        """
        添加节点到贝叶斯网络
        
        参数:
            node: 节点名称
        """
        self.nodes.add(node)
    
    def add_edge(self, parent, child):
        """
        添加边到贝叶斯网络
        
        参数:
            parent: 父节点名称
            child: 子节点名称
        """
        # 确保两个节点都存在
        if parent not in self.nodes:
            self.add_node(parent)
        if child not in self.nodes:
            self.add_node(child)
        
        # 添加边
        self.edges.add((parent, child))
    
    def set_cpd(self, node, cpd):
        """
        设置节点的条件概率分布
        
        参数:
            node: 节点名称
            cpd: 条件概率分布对象
        """
        if node not in self.nodes:
            raise ValueError(f"节点 {node} 不存在于网络中")
        self.cpds[node] = cpd
    
    def get_parents(self, node):
        """
        获取节点的所有父节点
        
        参数:
            node: 节点名称
        
        返回:
            父节点列表
        """
        return [parent for parent, child in self.edges if child == node]
    
    def get_children(self, node):
        """
        获取节点的所有子节点
        
        参数:
            node: 节点名称
        
        返回:
            子节点列表
        """
        return [child for parent, child in self.edges if parent == node]


class CPD:
    """
    条件概率分布类
    """
    def __init__(self, node, parents, probabilities):
        """
        初始化条件概率分布
        
        参数:
            node: 节点名称
            parents: 父节点列表
            probabilities: 条件概率表，格式为字典
        """
        self.node = node
        self.parents = parents
        self.probabilities = probabilities
    
    def get_probability(self, node_value, parent_values):
        """
        获取给定父节点值的条件概率
        
        参数:
            node_value: 节点值
            parent_values: 父节点值的字典
        
        返回:
            条件概率
        """
        # 创建键，按照父节点顺序排列
        if not self.parents:
            # 边缘概率
            return self.probabilities[node_value]
        
        key = tuple(parent_values[parent] for parent in self.parents)
        return self.probabilities[key][node_value]


class NaiveBayesClassifier:
    """
    朴素贝叶斯分类器类
    """
    def __init__(self):
        """
        初始化朴素贝叶斯分类器
        """
        self.classes = None
        self.class_priors = None
        self.feature_likelihoods = None
    
    def fit(self, X, y):
        """
        训练朴素贝叶斯分类器
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
            y: 标签向量，形状为 (n_samples,)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # 计算类先验概率
        self.class_priors = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.class_priors[i] = np.sum(y == c) / n_samples
        
        # 计算特征似然（假设伯努利分布）
        self.feature_likelihoods = []
        for feature_idx in range(n_features):
            # 每个特征的似然：P(x_i | y)
            feature_values = np.unique(X[:, feature_idx])
            likelihood = np.zeros((n_classes, len(feature_values)))
            
            for i, c in enumerate(self.classes):
                for j, v in enumerate(feature_values):
                    # 计算 P(x_i = v | y = c)
                    numerator = np.sum((X[:, feature_idx] == v) & (y == c))
                    denominator = np.sum(y == c)
                    # 使用拉普拉斯平滑
                    likelihood[i, j] = (numerator + 1) / (denominator + len(feature_values))
            
            self.feature_likelihoods.append((feature_values, likelihood))
    
    def predict(self, X):
        """
        预测标签
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
        
        返回:
            预测标签向量
        """
        n_samples = X.shape[0]
        predictions = []
        
        for i in range(n_samples):
            sample = X[i]
            class_scores = []
            
            for class_idx, c in enumerate(self.classes):
                # 先验概率
                score = np.log(self.class_priors[class_idx])
                
                # 加上每个特征的似然（使用对数以避免下溢）
                for feature_idx, (feature_values, likelihood) in enumerate(self.feature_likelihoods):
                    feature_value = sample[feature_idx]
                    # 找到特征值的索引
                    if feature_value in feature_values:
                        value_idx = list(feature_values).index(feature_value)
                        score += np.log(likelihood[class_idx, value_idx])
                    else:
                        # 未知特征值，使用均匀分布
                        score += np.log(1 / len(feature_values))
                
                class_scores.append(score)
            
            # 选择得分最高的类
            predictions.append(self.classes[np.argmax(class_scores)])
        
        return np.array(predictions)


def sigmoid(z):
    """
    Sigmoid激活函数
    
    参数:
        z: 输入值
    
    返回:
        sigmoid(z) 的值
    """
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    """
    逻辑回归类
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        初始化逻辑回归模型
        
        参数:
            learning_rate: 学习率
            n_iterations: 迭代次数
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        训练逻辑回归模型
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
            y: 标签向量，形状为 (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for _ in range(self.n_iterations):
            # 计算线性模型
            linear_model = np.dot(X, self.weights) + self.bias
            # 应用sigmoid函数
            y_predicted = sigmoid(linear_model)
            
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """
        预测标签
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
        
        返回:
            预测标签向量
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        # 阈值为0.5
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)


class NeuralLogisticRegression:
    """
    神经逻辑回归类（带隐藏层）
    """
    def __init__(self, input_dim, hidden_dim, learning_rate=0.01, n_iterations=1000):
        """
        初始化神经逻辑回归模型
        
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            learning_rate: 学习率
            n_iterations: 迭代次数
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
        
        返回:
            预测概率和隐藏层输出
        """
        # 隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)  # 使用tanh激活函数
        
        # 输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y):
        """
        反向传播
        
        参数:
            X: 特征矩阵
            y: 标签向量
        """
        n_samples = X.shape[0]
        
        # 计算损失导数
        dz2 = self.a2 - y.reshape(-1, 1)
        
        # 计算输出层权重和偏置的梯度
        dW2 = (1 / n_samples) * np.dot(self.a1.T, dz2)
        db2 = (1 / n_samples) * np.sum(dz2, axis=0, keepdims=True)
        
        # 计算隐藏层权重和偏置的梯度
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (1 - np.power(self.a1, 2))  # tanh导数
        dW1 = (1 / n_samples) * np.dot(X.T, dz1)
        db1 = (1 / n_samples) * np.sum(dz1, axis=0)
        
        # 更新参数
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def fit(self, X, y):
        """
        训练神经逻辑回归模型
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
            y: 标签向量，形状为 (n_samples,)
        """
        for _ in range(self.n_iterations):
            # 前向传播
            self.forward(X)
            # 反向传播
            self.backward(X, y)
    
    def predict(self, X):
        """
        预测标签
        
        参数:
            X: 特征矩阵，形状为 (n_samples, n_features)
        
        返回:
            预测标签向量
        """
        y_pred_proba = self.forward(X)
        y_pred = [1 if i > 0.5 else 0 for i in y_pred_proba.flatten()]
        return np.array(y_pred)


# 测试函数
def test_naive_bayes():
    """
    测试朴素贝叶斯分类器
    """
    print("\n=== 朴素贝叶斯分类器测试 ===")
    
    # 使用Iris数据集（简化为二分类）
    iris = load_iris()
    X = iris.data
    y = (iris.target == 0).astype(int)  # 只区分山鸢尾(0)和其他花(1)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 二值化特征（为了适应伯努利朴素贝叶斯）
    X_train_binary = (X_train > X_train.mean(axis=0)).astype(int)
    X_test_binary = (X_test > X_train.mean(axis=0)).astype(int)
    
    # 训练和预测
    nb = NaiveBayesClassifier()
    nb.fit(X_train_binary, y_train)
    y_pred = nb.predict(X_test_binary)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"朴素贝叶斯分类器准确率: {accuracy:.4f}")


def test_logistic_regression():
    """
    测试逻辑回归模型
    """
    print("\n=== 逻辑回归测试 ===")
    
    # 使用Iris数据集
    iris = load_iris()
    X = iris.data
    y = (iris.target == 0).astype(int)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练和预测
    lr = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"逻辑回归准确率: {accuracy:.4f}")


def test_neural_logistic_regression():
    """
    测试神经逻辑回归模型
    """
    print("\n=== 神经逻辑回归测试 ===")
    
    # 使用Iris数据集
    iris = load_iris()
    X = iris.data
    y = (iris.target == 0).astype(int)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练和预测
    nlr = NeuralLogisticRegression(input_dim=4, hidden_dim=10, learning_rate=0.01, n_iterations=1000)
    nlr.fit(X_train, y_train)
    y_pred = nlr.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"神经逻辑回归准确率: {accuracy:.4f}")


def create_student_network():
    """
    创建学生网络示例
    """
    print("\n=== 学生网络示例 ===")
    
    # 创建贝叶斯网络
    student_net = BayesianNetwork()
    
    # 添加节点
    student_net.add_node("Difficulty")  # 课程难度
    student_net.add_node("Intelligence")  # 学生智力
    student_net.add_node("Grade")  # 学生成绩
    student_net.add_node("SAT")  # SAT分数
    student_net.add_node("Letter")  # 推荐信
    
    # 添加边
    student_net.add_edge("Difficulty", "Grade")
    student_net.add_edge("Intelligence", "Grade")
    student_net.add_edge("Intelligence", "SAT")
    student_net.add_edge("Grade", "Letter")
    
    # 设置CPD
    # Difficulty: 简单(0)或困难(1)
    student_net.set_cpd("Difficulty", CPD("Difficulty", [], {0: 0.6, 1: 0.4}))
    
    # Intelligence: 低(0)或高(1)
    student_net.set_cpd("Intelligence", CPD("Intelligence", [], {0: 0.7, 1: 0.3}))
    
    # Grade: 差(0), 中(1), 好(2) | Difficulty, Intelligence
    grade_cpd = CPD("Grade", ["Difficulty", "Intelligence"], {
        (0, 0): {0: 0.3, 1: 0.4, 2: 0.3},  # 简单课程, 低智力
        (0, 1): {0: 0.1, 1: 0.3, 2: 0.6},  # 简单课程, 高智力
        (1, 0): {0: 0.7, 1: 0.2, 2: 0.1},  # 困难课程, 低智力
        (1, 1): {0: 0.2, 1: 0.4, 2: 0.4}   # 困难课程, 高智力
    })
    student_net.set_cpd("Grade", grade_cpd)
    
    # SAT: 低(0)或高(1) | Intelligence
    sat_cpd = CPD("SAT", ["Intelligence"], {
        0: {0: 0.9, 1: 0.1},  # 低智力
        1: {0: 0.2, 1: 0.8}   # 高智力
    })
    student_net.set_cpd("SAT", sat_cpd)
    
    # Letter: 差(0)或好(1) | Grade
    letter_cpd = CPD("Letter", ["Grade"], {
        0: {0: 0.9, 1: 0.1},  # 差成绩
        1: {0: 0.6, 1: 0.4},  # 中成绩
        2: {0: 0.1, 1: 0.9}   # 好成绩
    })
    student_net.set_cpd("Letter", letter_cpd)
    
    print("学生网络已创建完成")
    print(f"节点: {student_net.nodes}")
    print(f"边: {student_net.edges}")
    
    # 示例查询
    grade = grade_cpd.get_probability(2, {"Difficulty": 0, "Intelligence": 1})
    print(f"P(Grade=好 | Difficulty=简单, Intelligence=高) = {grade:.4f}")


if __name__ == "__main__":
    print("===== 贝叶斯网络及其应用 =====")
    
    # 创建学生网络示例
    create_student_network()
    
    # 测试分类器
    test_naive_bayes()
    test_logistic_regression()
    test_neural_logistic_regression()
