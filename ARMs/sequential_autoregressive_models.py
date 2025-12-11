#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识点6：基于序列的自回归模型

这个文件包含基于序列的自回归模型的实现，主要是循环神经网络(RNN)及其应用：
1. 基本RNN单元
2. 字符级RNN语言模型
3. RNN的训练和采样
4. RNN的常见问题和改进
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def sigmoid(x):
    """
    Sigmoid激活函数
    
    参数:
        x: 输入值
    
    返回:
        sigmoid(x) 的值
    """
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """
    tanh激活函数
    
    参数:
        x: 输入值
    
    返回:
        tanh(x) 的值
    """
    return np.tanh(x)


def softmax(x):
    """
    Softmax激活函数
    
    参数:
        x: 输入值（向量或矩阵）
    
    返回:
        softmax(x) 的值
    """
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)


class RNNCell:
    """
    基本RNN单元
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化RNN单元
        
        参数:
            input_size: 输入维度
            hidden_size: 隐藏状态维度
            output_size: 输出维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # 输入到隐藏层权重
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏层到隐藏层权重
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # 隐藏层到输出层权重
        
        # 初始化偏置
        self.bh = np.zeros((hidden_size, 1))  # 隐藏层偏置
        self.by = np.zeros((output_size, 1))  # 输出层偏置
    
    def forward(self, x, h_prev):
        """
        前向传播
        
        参数:
            x: 当前时间步输入 (input_size, 1)
            h_prev: 前一时间步隐藏状态 (hidden_size, 1)
        
        返回:
            y: 当前时间步输出 (output_size, 1)
            h_next: 当前时间步隐藏状态 (hidden_size, 1)
        """
        # 计算隐藏状态
        h_next = tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
        
        # 计算输出
        y = np.dot(self.Why, h_next) + self.by
        
        return y, h_next


class CharacterRNN:
    """
    字符级RNN语言模型
    """
    def __init__(self, chars, hidden_size=100, seq_length=25, learning_rate=1e-1):
        """
        初始化字符级RNN
        
        参数:
            chars: 字符集列表
            hidden_size: 隐藏状态维度
            seq_length: 训练时的序列长度
            learning_rate: 学习率
        """
        self.chars = chars
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        
        # 创建字符到索引和索引到字符的映射
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        
        # 初始化RNN单元
        self.rnn_cell = RNNCell(
            input_size=self.vocab_size,
            hidden_size=hidden_size,
            output_size=self.vocab_size
        )
        
        # 初始化梯度
        self.dWxh = np.zeros_like(self.rnn_cell.Wxh)
        self.dWhh = np.zeros_like(self.rnn_cell.Whh)
        self.dWhy = np.zeros_like(self.rnn_cell.Why)
        self.dbh = np.zeros_like(self.rnn_cell.bh)
        self.dby = np.zeros_like(self.rnn_cell.by)
    
    def one_hot_encode(self, char):
        """
        字符的one-hot编码
        
        参数:
            char: 单个字符
        
        返回:
            one-hot编码向量 (vocab_size, 1)
        """
        vec = np.zeros((self.vocab_size, 1))
        idx = self.char_to_idx[char]
        vec[idx] = 1
        return vec
    
    def one_hot_decode(self, vec):
        """
        one-hot向量解码为字符
        
        参数:
            vec: one-hot编码向量 (vocab_size, 1) 或概率分布 (vocab_size, 1)
        
        返回:
            解码后的字符
        """
        idx = np.argmax(vec)
        return self.idx_to_char[idx]
    
    def sample(self, h_prev, seed_char, n):
        """
        从模型采样序列
        
        参数:
            h_prev: 初始隐藏状态 (hidden_size, 1)
            seed_char: 起始字符
            n: 采样字符数量
        
        返回:
            采样得到的字符串
        """
        # 将起始字符转换为one-hot编码
        x = self.one_hot_encode(seed_char)
        
        sampled_chars = [seed_char]
        h = h_prev.copy()
        
        for _ in range(n - 1):
            # 前向传播
            y, h = self.rnn_cell.forward(x, h)
            
            # 转换为概率分布
            p = softmax(y)
            
            # 根据概率分布采样字符索引
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            
            # 解码为字符
            char = self.idx_to_char[idx]
            sampled_chars.append(char)
            
            # 更新输入为当前字符的one-hot编码
            x = self.one_hot_encode(char)
        
        return ''.join(sampled_chars)
    
    def forward_pass(self, inputs, targets, h_prev):
        """
        前向传播并计算损失
        
        参数:
            inputs: 输入字符索引列表
            targets: 目标字符索引列表
            h_prev: 初始隐藏状态 (hidden_size, 1)
        
        返回:
            loss: 交叉熵损失
            hs: 隐藏状态列表
            ys: 输出列表
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0
        
        # 对每个时间步进行前向传播
        for t in range(len(inputs)):
            # 输入的one-hot编码
            xs[t] = self.one_hot_encode(inputs[t])
            
            # 前向传播
            ys[t], hs[t] = self.rnn_cell.forward(xs[t], hs[t-1])
            
            # 计算概率分布
            ps[t] = softmax(ys[t])
            
            # 计算交叉熵损失
            loss += -np.log(ps[t][self.char_to_idx[targets[t]], 0])
        
        return loss, hs, xs, ys, ps
    
    def backward_pass(self, hs, xs, ys, ps, targets):
        """
        反向传播计算梯度
        
        参数:
            hs: 隐藏状态列表
            xs: 输入列表
            ys: 输出列表
            ps: 概率分布列表
            targets: 目标字符索引列表
        """
        # 初始化梯度
        dhnext = np.zeros_like(hs[0])
        
        # 对每个时间步进行反向传播（从后往前）
        for t in reversed(range(len(xs))):
            # 输出层梯度
            dy = np.copy(ps[t])
            dy[self.char_to_idx[targets[t]]] -= 1  # 交叉熵损失的梯度
            
            # 隐藏层到输出层权重和偏置的梯度
            self.dWhy += np.dot(dy, hs[t].T)
            self.dby += dy
            
            # 隐藏层梯度
            dh = np.dot(self.rnn_cell.Why.T, dy) + dhnext  # 包括来自下一时间步的梯度
            
            # tanh的导数
            dhraw = (1 - hs[t] * hs[t]) * dh
            
            # 偏置梯度
            self.dbh += dhraw
            
            # 输入到隐藏层权重梯度
            self.dWxh += np.dot(dhraw, xs[t].T)
            
            # 隐藏层到隐藏层权重梯度
            self.dWhh += np.dot(dhraw, hs[t-1].T)
            
            # 传播到前一时间步的隐藏状态梯度
            dhnext = np.dot(self.rnn_cell.Whh.T, dhraw)
    
    def update_parameters(self):
        """
        使用梯度下降更新参数
        """
        # 梯度裁剪，防止梯度爆炸
        for dparam in [self.dWxh, self.dWhh, self.dWhy, self.dbh, self.dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        # 更新参数
        self.rnn_cell.Wxh -= self.learning_rate * self.dWxh
        self.rnn_cell.Whh -= self.learning_rate * self.dWhh
        self.rnn_cell.Why -= self.learning_rate * self.dWhy
        self.rnn_cell.bh -= self.learning_rate * self.dbh
        self.rnn_cell.by -= self.learning_rate * self.dby
        
        # 重置梯度
        self.dWxh.fill(0)
        self.dWhh.fill(0)
        self.dWhy.fill(0)
        self.dbh.fill(0)
        self.dby.fill(0)
    
    def train(self, text, epochs=10):
        """
        训练字符级RNN
        
        参数:
            text: 训练文本
            epochs: 训练轮数
        """
        n = len(text)
        losses = []
        start_time = time.time()
        
        print("开始训练字符级RNN...")
        
        for epoch in range(epochs):
            # 初始化隐藏状态
            h_prev = np.zeros((self.hidden_size, 1))
            
            # 随机选择起始位置
            p = np.random.randint(0, self.seq_length)
            
            for i in range(p, n - self.seq_length, self.seq_length):
                # 准备输入和目标序列
                inputs = text[i:i+self.seq_length]
                targets = text[i+1:i+self.seq_length+1]
                
                # 前向传播
                loss, hs, xs, ys, ps = self.forward_pass(inputs, targets, h_prev)
                losses.append(loss)
                
                # 反向传播
                self.backward_pass(hs, xs, ys, ps, targets)
                
                # 更新参数
                self.update_parameters()
                
                # 更新初始隐藏状态
                h_prev = hs[len(inputs)-1]
            
            # 每个epoch结束后采样并打印结果
            if (epoch + 1) % 1 == 0:
                sample_text = self.sample(h_prev, text[p], 200)
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Loss: {loss/self.seq_length:.4f}")
                print(f"Sample: {sample_text}")
        
        training_time = time.time() - start_time
        print(f"\n训练完成，耗时 {training_time:.2f} 秒")
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 4))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.savefig('figure/rnn_loss_curve.png')
        print("损失曲线已保存到 figure/rnn_loss_curve.png")


class LSTMCell:
    """
    LSTM单元 (Long Short-Term Memory)
    用于解决RNN的梯度消失问题
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化LSTM单元
        
        参数:
            input_size: 输入维度
            hidden_size: 隐藏状态维度
            output_size: 输出维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01  # 遗忘门权重
        self.bf = np.zeros((hidden_size, 1))  # 遗忘门偏置
        
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01  # 输入门权重
        self.bi = np.zeros((hidden_size, 1))  # 输入门偏置
        
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01  # 细胞状态更新权重
        self.bc = np.zeros((hidden_size, 1))  # 细胞状态更新偏置
        
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01  # 输出门权重
        self.bo = np.zeros((hidden_size, 1))  # 输出门偏置
        
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # 隐藏层到输出层权重
        self.by = np.zeros((output_size, 1))  # 输出层偏置
    
    def forward(self, x, h_prev, c_prev):
        """
        LSTM前向传播
        
        参数:
            x: 当前时间步输入 (input_size, 1)
            h_prev: 前一时间步隐藏状态 (hidden_size, 1)
            c_prev: 前一时间步细胞状态 (hidden_size, 1)
        
        返回:
            y: 当前时间步输出 (output_size, 1)
            h_next: 当前时间步隐藏状态 (hidden_size, 1)
            c_next: 当前时间步细胞状态 (hidden_size, 1)
        """
        # 拼接输入和隐藏状态
        concat = np.concatenate((h_prev, x), axis=0)
        
        # 遗忘门
        f = sigmoid(np.dot(self.Wf, concat) + self.bf)
        
        # 输入门
        i = sigmoid(np.dot(self.Wi, concat) + self.bi)
        
        # 细胞状态更新
        c_tilde = tanh(np.dot(self.Wc, concat) + self.bc)
        
        # 新的细胞状态
        c_next = f * c_prev + i * c_tilde
        
        # 输出门
        o = sigmoid(np.dot(self.Wo, concat) + self.bo)
        
        # 新的隐藏状态
        h_next = o * tanh(c_next)
        
        # 输出
        y = np.dot(self.Why, h_next) + self.by
        
        return y, h_next, c_next


# 示例函数
def load_sample_text(filename="sample_text.txt"):
    """
    加载示例文本
    
    参数:
        filename: 文件名
    
    返回:
        文本内容
    """
    # 检查文件是否存在
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        # 如果文件不存在，使用默认文本
        text = "This is a sample text for training the character RNN. " * 10
        print(f"{filename} not found, using default sample text.")
    
    return text


def rnn_basic_demo():
    """
    基本RNN演示
    """
    print("\n=== 基本RNN演示 ===")
    
    # 创建RNN单元
    rnn_cell = RNNCell(input_size=5, hidden_size=3, output_size=5)
    
    # 生成随机输入
    x = np.random.randn(5, 1)
    h_prev = np.random.randn(3, 1)
    
    # 前向传播
    y, h_next = rnn_cell.forward(x, h_prev)
    
    print(f"输入形状: {x.shape}")
    print(f"前一隐藏状态形状: {h_prev.shape}")
    print(f"输出形状: {y.shape}")
    print(f"新隐藏状态形状: {h_next.shape}")
    print(f"输出值: {y.ravel()}")


def character_rnn_demo():
    """
    字符级RNN演示
    """
    print("\n=== 字符级RNN语言模型演示 ===")
    
    # 加载示例文本
    text = load_sample_text()
    
    # 获取字符集
    chars = sorted(list(set(text)))
    print(f"字符集大小: {len(chars)}")
    print(f"字符集: {chars}")
    
    # 创建字符RNN模型
    rnn = CharacterRNN(
        chars=chars,
        hidden_size=50,
        seq_length=10,
        learning_rate=1e-1
    )
    
    # 训练模型（只训练1轮以节省时间）
    rnn.train(text, epochs=2)
    
    # 采样
    h_prev = np.zeros((rnn.hidden_size, 1))
    sample_text = rnn.sample(h_prev, text[0], 100)
    print(f"\n最终采样结果: {sample_text}")


def lstm_demo():
    """
    LSTM演示
    """
    print("\n=== LSTM演示 ===")
    
    # 创建LSTM单元
    lstm_cell = LSTMCell(input_size=5, hidden_size=3, output_size=5)
    
    # 生成随机输入
    x = np.random.randn(5, 1)
    h_prev = np.random.randn(3, 1)
    c_prev = np.random.randn(3, 1)
    
    # 前向传播
    y, h_next, c_next = lstm_cell.forward(x, h_prev, c_prev)
    
    print(f"输入形状: {x.shape}")
    print(f"前一隐藏状态形状: {h_prev.shape}")
    print(f"前一细胞状态形状: {c_prev.shape}")
    print(f"输出形状: {y.shape}")
    print(f"新隐藏状态形状: {h_next.shape}")
    print(f"新细胞状态形状: {c_next.shape}")


def rnn_problems_demo():
    """
    演示RNN的常见问题
    """
    print("\n=== RNN常见问题演示 ===")
    
    # 创建一个简单的RNN来演示梯度消失
    rnn_cell = RNNCell(input_size=2, hidden_size=2, output_size=2)
    
    # 生成简单序列
    sequence_length = 100
    x_sequence = [np.random.randn(2, 1) for _ in range(sequence_length)]
    
    # 前向传播
    h = np.zeros((2, 1))
    hs = [h]
    
    for x in x_sequence:
        y, h = rnn_cell.forward(x, h)
        hs.append(h)
    
    # 计算梯度（模拟）
    gradients = []
    h_grad = np.ones((2, 1))
    
    for t in reversed(range(len(x_sequence))):
        # 计算梯度（简化版）
        h_grad = h_grad * (1 - hs[t+1]**2)  # tanh的导数
        gradients.append(np.linalg.norm(h_grad))
    
    gradients.reverse()
    
    # 绘制梯度范数随时间步的变化
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(gradients)), gradients)
    plt.title('Gradient Norm Over Time Steps')
    plt.xlabel('Time Step')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.savefig('figure/rnn_gradient_vanishing.png')
    print("梯度消失演示图已保存到 figure/rnn_gradient_vanishing.png")


if __name__ == "__main__":
    print("===== 基于序列的自回归模型 =====")
    
    # 创建figure目录（如果不存在）
    import os
    os.makedirs('figure', exist_ok=True)
    
    # 基本RNN演示
    rnn_basic_demo()
    
    # LSTM演示
    lstm_demo()
    
    # 字符级RNN演示（注意：训练需要一定时间）
    # character_rnn_demo()  # 取消注释以运行完整演示
    
    # RNN问题演示
    rnn_problems_demo()
    
    print("\n所有演示完成！")
