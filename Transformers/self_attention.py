# 知识点3. 自注意力机制与多头注意力

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SelfAttentionMechanism(nn.Module):
    """
    自注意力机制的详细实现
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # 线性变换层用于生成Q、K、V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        """
        自注意力前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_length, d_model)
            mask: 注意力掩码，形状为 (batch_size, seq_length, seq_length)，
                 将不希望被注意的位置设置为-∞
        
        Returns:
            output: 自注意力输出，形状为 (batch_size, seq_length, d_model)
            attention_weights: 注意力权重，形状为 (batch_size, seq_length, seq_length)
        """
        # 步骤1: 生成Q、K、V矩阵
        Q = self.W_q(x)  # (batch_size, seq_length, d_model)
        K = self.W_k(x)  # (batch_size, seq_length, d_model)
        V = self.W_v(x)  # (batch_size, seq_length, d_model)
        
        # 步骤2: 计算注意力分数
        # 分数 = Q * K^T / sqrt(d_model)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_model)
        
        # 步骤3: 应用掩码（如果提供）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 步骤4: 应用softmax获取注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 步骤5: 将注意力权重应用于V
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

class MultiHeadAttentionMechanism(nn.Module):
    """
    多头注意力机制的详细实现
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def split_heads(self, x):
        """
        将输入分割到多个头
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_length, d_model)
            
        Returns:
            分割后的张量，形状为 (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """
        将多个头的输出合并
        
        Args:
            x: 输入张量，形状为 (batch_size, num_heads, seq_length, d_k)
            
        Returns:
            合并后的张量，形状为 (batch_size, seq_length, d_model)
        """
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, x, mask=None):
        """
        多头注意力前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, seq_length, d_model)
            mask: 注意力掩码，形状为 (batch_size, seq_length, seq_length)
            
        Returns:
            output: 多头注意力输出，形状为 (batch_size, seq_length, d_model)
            attention_weights: 注意力权重，形状为 (batch_size, num_heads, seq_length, seq_length)
        """
        batch_size = x.size(0)
        
        # 步骤1: 生成Q、K、V矩阵
        Q = self.W_q(x)  # (batch_size, seq_length, d_model)
        K = self.W_k(x)  # (batch_size, seq_length, d_model)
        V = self.W_v(x)  # (batch_size, seq_length, d_model)
        
        # 步骤2: 分割到多个头
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_length, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_length, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_length, d_k)
        
        # 步骤3: 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # 步骤4: 应用掩码
        if mask is not None:
            # 扩展掩码维度以匹配多头注意力的要求
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 步骤5: 应用softmax获取注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 步骤6: 将注意力权重应用于V
        scaled_attention = torch.matmul(attention_weights, V)
        
        # 步骤7: 合并多个头的输出
        output = self.combine_heads(scaled_attention)
        
        # 步骤8: 应用最终线性变换
        output = self.W_o(output)
        
        return output, attention_weights

def visualize_attention(attention_weights, tokens):
    """
    可视化注意力权重矩阵
    
    Args:
        attention_weights: 注意力权重，形状为 (seq_length, seq_length)
        tokens: 标记列表，用于标注坐标轴
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights.detach().numpy(), cmap='viridis')
    plt.colorbar()
    
    # 设置坐标轴标签
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    
    plt.title('Self-Attention Weights')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    
    plt.tight_layout()
    plt.savefig('attention_visualization.png')
    plt.close()
    print("注意力可视化已保存为 'attention_visualization.png'")

# 示例：自注意力机制的应用
if __name__ == "__main__":
    # 示例句子
    sentence = "The animal didn't cross the street because it was too tired"
    tokens = sentence.split()
    seq_length = len(tokens)
    
    # 初始化参数
    d_model = 512
    num_heads = 8
    
    # 创建随机嵌入（实际应用中应使用预训练嵌入或学习嵌入）
    embeddings = torch.randn(1, seq_length, d_model)
    
    # 测试自注意力
    self_attention = SelfAttentionMechanism(d_model)
    sa_output, sa_weights = self_attention(embeddings)
    
    print("自注意力机制：")
    print(f"  输入形状: {embeddings.shape}")
    print(f"  输出形状: {sa_output.shape}")
    print(f"  注意力权重形状: {sa_weights.shape}")
    print()
    
    # 可视化第一个头的自注意力权重
    visualize_attention(sa_weights[0], tokens)
    
    # 测试多头注意力
    multihead_attention = MultiHeadAttentionMechanism(d_model, num_heads)
    mha_output, mha_weights = multihead_attention(embeddings)
    
    print("多头注意力机制：")
    print(f"  输入形状: {embeddings.shape}")
    print(f"  输出形状: {mha_output.shape}")
    print(f"  注意力权重形状: {mha_weights.shape}")
    print(f"  注意力头数量: {num_heads}")
    print()
    
    # 分析注意力
    print("注意力分析：")
    print(f"  句子: {sentence}")
    print()
    
    # 查看"it"单词对其他单词的注意力
    it_index = tokens.index("it")
    print(f"  'it' 对其他单词的注意力权重:")
    for i, token in enumerate(tokens):
        print(f"    {token}: {sa_weights[0, it_index, i].item():.4f}")
    
    # 找出"it"注意力最高的单词
    most_attended = tokens[torch.argmax(sa_weights[0, it_index]).item()]
    print(f"  'it' 最关注的单词: {most_attended}")
    
    print()
    print("多头注意力优势展示：")
    print("  - 不同头可以关注不同的依赖关系")
    print("  - 提高了模型捕捉不同位置信息的能力")
    print("  - 提供了多个表示子空间")
