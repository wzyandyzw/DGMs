#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识点7：现代自回归模型

这个文件包含现代自回归模型的实现，主要是注意力机制和Transformer：
1. 注意力机制
2. 多头注意力
3. Transformer解码器
4. GPT风格的自回归模型
5. 条件生成模型
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def softmax(x, axis=-1):
    """
    Softmax激活函数
    
    参数:
        x: 输入值（向量或矩阵）
        axis: 计算softmax的轴
    
    返回:
        softmax(x) 的值
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, gamma=1.0, beta=0.0, eps=1e-6):
    """
    层归一化
    
    参数:
        x: 输入张量
        gamma: 缩放参数
        beta: 偏移参数
        eps: 防止除零的小值
    
    返回:
        归一化后的张量
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(variance + eps)
    return gamma * x_norm + beta


def gelu(x):
    """
    GELU激活函数
    
    参数:
        x: 输入值
    
    返回:
        gelu(x) 的值
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    缩放点积注意力
    
    参数:
        q: 查询张量 (..., seq_len_q, depth)
        k: 键张量 (..., seq_len_k, depth)
        v: 值张量 (..., seq_len_v, depth_v)
        mask: 注意力掩码 (..., seq_len_q, seq_len_k)
    
    返回:
        注意力输出和注意力权重
    """
    # 计算注意力分数
    matmul_qk = np.matmul(q, k.swapaxes(-1, -2))  # (..., seq_len_q, seq_len_k)
    
    # 缩放
    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    
    # 应用掩码（如果提供）
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    # 计算注意力权重
    attention_weights = softmax(scaled_attention_logits, axis=-1)
    
    # 计算注意力输出
    output = np.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    
    return output, attention_weights


class MultiHeadAttention:
    """
    多头注意力层
    """
    def __init__(self, d_model, num_heads):
        """
        初始化多头注意力
        
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
        """
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        # 线性投影层权重
        self.wq = np.random.randn(d_model, d_model) * 0.01
        self.wk = np.random.randn(d_model, d_model) * 0.01
        self.wv = np.random.randn(d_model, d_model) * 0.01
        self.wo = np.random.randn(d_model, d_model) * 0.01
    
    def split_heads(self, x, batch_size):
        """
        将输入分割为多个头
        
        参数:
            x: 输入张量 (batch_size, seq_len, d_model)
            batch_size: 批次大小
        
        返回:
            分割后的张量 (batch_size, num_heads, seq_len, depth)
        """
        x = np.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return np.transpose(x, axes=(0, 2, 1, 3))
    
    def forward(self, v, k, q, mask):
        """
        前向传播
        
        参数:
            v: 值张量 (batch_size, seq_len_v, d_model)
            k: 键张量 (batch_size, seq_len_k, d_model)
            q: 查询张量 (batch_size, seq_len_q, d_model)
            mask: 注意力掩码 (batch_size, seq_len_q, seq_len_k)
        
        返回:
            注意力输出和注意力权重
        """
        batch_size = q.shape[0]
        
        # 线性投影
        q = np.matmul(q, self.wq)
        k = np.matmul(k, self.wk)
        v = np.matmul(v, self.wv)
        
        # 分割为多个头
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # 计算缩放点积注意力
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        # 合并头
        scaled_attention = np.transpose(scaled_attention, axes=(0, 2, 1, 3))
        concat_attention = np.reshape(scaled_attention, 
                                     (batch_size, -1, self.d_model))
        
        # 线性投影
        output = np.matmul(concat_attention, self.wo)
        
        return output, attention_weights


class FeedForwardNetwork:
    """
    前馈神经网络
    """
    def __init__(self, d_model, dff):
        """
        初始化前馈神经网络
        
        参数:
            d_model: 模型维度
            dff: 前馈网络隐藏层维度
        """
        self.w1 = np.random.randn(d_model, dff) * 0.01
        self.b1 = np.zeros(dff)
        self.w2 = np.random.randn(dff, d_model) * 0.01
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 (batch_size, seq_len, d_model)
        
        返回:
            前馈网络输出
        """
        x = np.matmul(x, self.w1) + self.b1
        x = gelu(x)
        x = np.matmul(x, self.w2) + self.b2
        return x


class TransformerDecoderLayer:
    """
    Transformer解码器层
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        初始化Transformer解码器层
        
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            dff: 前馈网络隐藏层维度
            rate:  dropout率
        """
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, dff)
        
        self.layernorm1 = lambda x: layer_norm(x)
        self.layernorm2 = lambda x: layer_norm(x)
        
        self.rate = rate
    
    def forward(self, x, look_ahead_mask):
        """
        前向传播
        
        参数:
            x: 输入张量 (batch_size, target_seq_len, d_model)
            look_ahead_mask: 前瞻掩码 (batch_size, target_seq_len, target_seq_len)
        
        返回:
            解码器层输出和注意力权重
        """
        # 多头自注意力
        attn_output, attn_weights_block1 = self.mha(x, x, x, look_ahead_mask)
        out1 = self.layernorm1(x + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2, attn_weights_block1


class Decoder:
    """
    Transformer解码器
    """
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_pos_encoding, rate=0.1):
        """
        初始化Transformer解码器
        
        参数:
            num_layers: 解码器层数
            d_model: 模型维度
            num_heads: 注意力头数
            dff: 前馈网络隐藏层维度
            vocab_size: 词汇表大小
            max_pos_encoding: 最大位置编码长度
            rate: dropout率
        """
        self.d_model = d_model
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01
        
        # 位置编码
        self.pos_encoding = self.positional_encoding(max_pos_encoding, d_model)
        
        # 解码器层
        self.dec_layers = [TransformerDecoderLayer(d_model, num_heads, dff, rate)
                          for _ in range(num_layers)]
        
        # 输出层
        self.final_layer = np.random.randn(d_model, vocab_size) * 0.01
    
    def positional_encoding(self, position, d_model):
        """
        位置编码
        
        参数:
            position: 位置数量
            d_model: 模型维度
        
        返回:
            位置编码矩阵 (position, d_model)
        """
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                    np.arange(d_model)[np.newaxis, :],
                                    d_model)
        
        # 偶数列使用sin
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # 奇数列使用cos
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return pos_encoding
    
    def get_angles(self, pos, i, d_model):
        """
        计算位置编码角度
        
        参数:
            pos: 位置索引
            i: 维度索引
            d_model: 模型维度
        
        返回:
            角度值
        """
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def create_look_ahead_mask(self, size):
        """
        创建前瞻掩码
        
        参数:
            size: 序列长度
        
        返回:
            前瞻掩码 (size, size)
        """
        mask = 1 - np.tril(np.ones((size, size)))
        return mask  # (size, size)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 (batch_size, target_seq_len)
        
        返回:
            解码器输出和注意力权重
        """
        batch_size = x.shape[0]
        target_seq_len = x.shape[1]
        attention_weights = {}
        
        # 嵌入
        x = self.embedding[x]
        x *= np.sqrt(self.d_model)
        
        # 添加位置编码
        x += self.pos_encoding[:, :target_seq_len, :]
        
        # 创建前瞻掩码
        look_ahead_mask = self.create_look_ahead_mask(target_seq_len)
        look_ahead_mask = np.tile(look_ahead_mask[np.newaxis, :, :], (batch_size, 1, 1))
        
        # 解码器层
        for i in range(self.num_layers):
            x, block1 = self.dec_layers[i].forward(x, look_ahead_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
        
        # 输出层
        final_output = np.matmul(x, self.final_layer)  # (batch_size, target_seq_len, vocab_size)
        
        return final_output, attention_weights
    
    def sample(self, start_token, max_length):
        """
        自回归采样
        
        参数:
            start_token: 起始标记索引
            max_length: 最大生成长度
        
        返回:
            采样序列
        """
        # 初始化序列
        sequence = [start_token]
        
        for _ in range(max_length - 1):
            # 创建输入张量
            x = np.array([sequence])
            
            # 前向传播
            predictions, _ = self.forward(x)
            
            # 获取最后一个时间步的预测
            next_token_logits = predictions[0, -1, :]
            
            # 使用softmax获取概率分布
            next_token_probs = softmax(next_token_logits)
            
            # 采样下一个标记
            next_token = np.random.choice(len(next_token_probs), p=next_token_probs)
            
            # 添加到序列
            sequence.append(next_token)
        
        return sequence


class GPT:
    """
    GPT风格的自回归模型
    """
    def __init__(self, vocab_size, d_model=768, num_heads=12, dff=3072, num_layers=12, max_seq_len=1024):
        """
        初始化GPT模型
        
        参数:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            dff: 前馈网络隐藏层维度
            num_layers: 解码器层数
            max_seq_len: 最大序列长度
        """
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=vocab_size,
            max_pos_encoding=max_seq_len
        )
        
        self.vocab_size = vocab_size
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量 (batch_size, seq_len)
        
        返回:
            模型输出和注意力权重
        """
        return self.decoder.forward(x)
    
    def sample(self, start_token, max_length):
        """
        自回归采样
        
        参数:
            start_token: 起始标记索引
            max_length: 最大生成长度
        
        返回:
            采样序列
        """
        return self.decoder.sample(start_token, max_length)
    
    def generate(self, context, max_length, temperature=1.0):
        """
        基于上下文生成文本
        
        参数:
            context: 上下文序列
            max_length: 最大生成长度
            temperature: 采样温度
        
        返回:
            生成的序列
        """
        sequence = context.copy()
        
        for _ in range(max_length - len(context)):
            # 创建输入张量
            x = np.array([sequence])
            
            # 前向传播
            predictions, _ = self.forward(x)
            
            # 获取最后一个时间步的预测
            next_token_logits = predictions[0, -1, :]
            
            # 应用温度缩放
            next_token_logits = next_token_logits / temperature
            
            # 使用softmax获取概率分布
            next_token_probs = softmax(next_token_logits)
            
            # 采样下一个标记
            next_token = np.random.choice(len(next_token_probs), p=next_token_probs)
            
            # 添加到序列
            sequence.append(next_token)
        
        return sequence


class ConditionalGPT:
    """
    条件生成GPT模型
    """
    def __init__(self, vocab_size, d_model=768, num_heads=12, dff=3072, num_layers=12, max_seq_len=1024):
        """
        初始化条件生成GPT模型
        
        参数:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            dff: 前馈网络隐藏层维度
            num_layers: 解码器层数
            max_seq_len: 最大序列长度
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 创建解码器（与GPT相同）
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=vocab_size,
            max_pos_encoding=max_seq_len
        )
        
        # 条件嵌入层
        self.condition_embedding = np.random.randn(vocab_size, d_model) * 0.01
    
    def forward(self, x, condition):
        """
        条件生成前向传播
        
        参数:
            x: 输入序列 (batch_size, seq_len)
            condition: 条件序列 (batch_size, condition_len)
        
        返回:
            模型输出
        """
        # 合并条件和输入序列
        combined_seq = np.concatenate([condition, x], axis=1)
        
        # 前向传播
        output, attention_weights = self.decoder.forward(combined_seq)
        
        # 只返回与输入序列对应的输出
        return output[:, condition.shape[1]:, :], attention_weights
    
    def generate(self, condition, max_length, temperature=1.0):
        """
        基于条件生成文本
        
        参数:
            condition: 条件序列
            max_length: 最大生成长度
            temperature: 采样温度
        
        返回:
            生成的序列
        """
        condition_len = len(condition)
        sequence = condition.copy()
        
        for _ in range(max_length):
            # 创建输入张量
            x = np.array([sequence])
            
            # 前向传播
            predictions, _ = self.decoder.forward(x)
            
            # 获取最后一个时间步的预测
            next_token_logits = predictions[0, -1, :]
            
            # 应用温度缩放
            next_token_logits = next_token_logits / temperature
            
            # 使用softmax获取概率分布
            next_token_probs = softmax(next_token_logits)
            
            # 采样下一个标记
            next_token = np.random.choice(len(next_token_probs), p=next_token_probs)
            
            # 添加到序列
            sequence.append(next_token)
        
        # 返回不包含条件部分的生成结果
        return sequence[condition_len:]


# 可视化函数
def plot_attention_weights(attention, sentence, predicted_sentence):
    """
    绘制注意力权重
    
    参数:
        attention: 注意力权重
        sentence: 输入句子
        predicted_sentence: 预测句子
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    ax.set_xticklabels([''] + predicted_sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + sentence, fontdict=fontdict)
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    
    plt.savefig('figure/attention_weights.png')
    print("注意力权重图已保存到 figure/attention_weights.png")


def plot_attention_heads(attention_weights, num_heads=4):
    """
    绘制多头注意力
    
    参数:
        attention_weights: 注意力权重
        num_heads: 要显示的头数
    """
    fig = plt.figure(figsize=(15, 15))
    
    for h in range(num_heads):
        ax = fig.add_subplot(2, 2, h+1)
        ax.matshow(attention_weights[0, h, :, :], cmap='viridis')
        ax.set_title(f'Head {h+1}')
    
    plt.tight_layout()
    plt.savefig('figure/multihead_attention.png')
    print("多头注意力图已保存到 figure/multihead_attention.png")


# 示例函数
def scaled_dot_product_attention_demo():
    """
    缩放点积注意力演示
    """
    print("\n=== 缩放点积注意力演示 ===")
    
    # 创建示例张量
    q = np.random.randn(1, 3, 64)
    k = np.random.randn(1, 3, 64)
    v = np.random.randn(1, 3, 64)
    
    # 创建掩码
    mask = np.array([[[0, 1, 1],
                     [0, 0, 1],
                     [0, 0, 0]]])
    
    # 计算注意力
    output, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    
    print(f"查询张量形状: {q.shape}")
    print(f"键张量形状: {k.shape}")
    print(f"值张量形状: {v.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"注意力权重:\n{attention_weights[0]}")


def multi_head_attention_demo():
    """
    多头注意力演示
    """
    print("\n=== 多头注意力演示 ===")
    
    # 创建多头注意力层
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    
    # 创建示例输入
    x = np.random.randn(64, 10, 512)
    
    # 创建掩码
    mask = np.ones((64, 10, 10))
    
    # 前向传播
    output, attn_weights = mha.forward(x, x, x, mask)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 绘制多头注意力
    plot_attention_heads(attn_weights)


def decoder_demo():
    """
    Transformer解码器演示
    """
    print("\n=== Transformer解码器演示 ===")
    
    # 创建解码器
    decoder = Decoder(
        num_layers=2,
        d_model=64,
        num_heads=2,
        dff=256,
        vocab_size=1000,
        max_pos_encoding=50
    )
    
    # 创建示例输入
    x = np.random.randint(0, 1000, (32, 10))
    
    # 前向传播
    output, attention_weights = decoder.forward(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重数量: {len(attention_weights)}")
    
    # 采样
    sample_sequence = decoder.sample(start_token=1, max_length=15)
    print(f"采样序列长度: {len(sample_sequence)}")
    print(f"采样序列: {sample_sequence}")


def gpt_demo():
    """
    GPT模型演示
    """
    print("\n=== GPT模型演示 ===")
    
    # 创建GPT模型（简化版）
    gpt = GPT(
        vocab_size=1000,
        d_model=64,
        num_heads=2,
        dff=256,
        num_layers=2,
        max_seq_len=50
    )
    
    # 创建示例输入
    x = np.random.randint(0, 1000, (32, 10))
    
    # 前向传播
    output, attention_weights = gpt.forward(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 采样
    sample_sequence = gpt.sample(start_token=1, max_length=15)
    print(f"采样序列长度: {len(sample_sequence)}")
    print(f"采样序列: {sample_sequence}")


def conditional_gpt_demo():
    """
    条件GPT模型演示
    """
    print("\n=== 条件GPT模型演示 ===")
    
    # 创建条件GPT模型（简化版）
    cond_gpt = ConditionalGPT(
        vocab_size=1000,
        d_model=64,
        num_heads=2,
        dff=256,
        num_layers=2,
        max_seq_len=50
    )
    
    # 创建示例输入和条件
    x = np.random.randint(0, 1000, (32, 5))
    condition = np.random.randint(0, 1000, (32, 2))
    
    # 前向传播
    output, attention_weights = cond_gpt.forward(x, condition)
    
    print(f"输入形状: {x.shape}")
    print(f"条件形状: {condition.shape}")
    print(f"输出形状: {output.shape}")
    
    # 条件生成
    condition_seq = [1, 2]  # 示例条件
    generated_seq = cond_gpt.generate(condition_seq, max_length=10)
    print(f"条件序列: {condition_seq}")
    print(f"生成序列长度: {len(generated_seq)}")
    print(f"生成序列: {generated_seq}")


if __name__ == "__main__":
    print("===== 现代自回归模型 =====")
    
    # 创建figure目录（如果不存在）
    import os
    os.makedirs('figure', exist_ok=True)
    
    # 缩放点积注意力演示
    scaled_dot_product_attention_demo()
    
    # 多头注意力演示
    multi_head_attention_demo()
    
    # Transformer解码器演示
    decoder_demo()
    
    # GPT模型演示
    gpt_demo()
    
    # 条件GPT模型演示
    conditional_gpt_demo()
    
    print("\n所有演示完成！")
