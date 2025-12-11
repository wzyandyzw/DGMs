# 知识点2. Transformer核心组件

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """
    层归一化实现
    LayerNorm(z; γ, β) = γ * (z - μ_z) / σ_z + β
    其中 μ_z = sum(z_i) / d, σ_z = sqrt(sum((z_i - μ_z)^2) / d)
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # 学习的缩放参数
        self.beta = nn.Parameter(torch.zeros(d_model))  # 学习的偏置参数
        self.eps = eps  # 防止除零的小常数
    
    def forward(self, z):
        # 计算均值和方差（按特征维度归一化）
        mu_z = z.mean(dim=-1, keepdim=True)
        sigma_z = z.std(dim=-1, keepdim=True, unbiased=False)  # 使用总体标准差
        
        # 归一化并应用学习的缩放和偏置
        normalized = (z - mu_z) / (sigma_z + self.eps)
        output = self.gamma * normalized + self.beta
        return output

class SelfAttention(nn.Module):
    """
    自注意力机制实现
    """
    def __init__(self, d_model, k):
        super().__init__()
        # 权重矩阵定义
        self.W_q = nn.Linear(d_model, k, bias=False)  # 查询矩阵 W_Q ∈ R^(d×k)
        self.W_k = nn.Linear(d_model, k, bias=False)  # 键矩阵 W_K ∈ R^(d×k)
        self.W_v = nn.Linear(d_model, k, bias=False)  # 值矩阵 W_V ∈ R^(d×k)
        self.W_c = nn.Linear(k, d_model, bias=False)  # 输出投影矩阵 W_C ∈ R^(k×d)
        
        self.k = k  # 键向量维度
    
    def forward(self, X):
        """
        自注意力前向传播
        X: 输入矩阵 ∈ R^(n×d)
        """
        # 步骤1: 计算查询、键、值矩阵
        Q = self.W_q(X)  # Q ∈ R^(n×k)
        K = self.W_k(X)  # K ∈ R^(n×k)
        V = self.W_v(X)  # V ∈ R^(n×k)
        
        # 步骤2: 计算注意力分数和注意力矩阵
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.k)
        attention_matrix = F.softmax(attention_scores, dim=-1)  # 按行执行softmax
        
        # 步骤3: 计算注意力输出
        Z = torch.matmul(attention_matrix, V)  # Z ∈ R^(n×k)
        
        # 步骤4: 投影回原始维度
        U_tilde = self.W_c(Z)  # U_tilde ∈ R^(n×d)
        
        return U_tilde, attention_matrix

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现
    """
    def __init__(self, d_model, k, h):
        super().__init__()
        self.h = h  # 注意力头数量
        self.k = k  # 每个头的键向量维度
        
        # 为每个头创建自注意力层
        self.attention_heads = nn.ModuleList([
            SelfAttention(d_model, k) for _ in range(h)
        ])
        
        # 最终输出投影层
        self.W_o = nn.Linear(h * k, d_model, bias=False)
    
    def forward(self, X):
        """
        多头注意力前向传播
        X: 输入矩阵 ∈ R^(n×d)
        """
        # 每个注意力头独立计算
        head_outputs = []
        attention_matrices = []
        
        for head in self.attention_heads:
            output, attn_matrix = head(X)
            head_outputs.append(output)
            attention_matrices.append(attn_matrix)
        
        # 拼接所有头的输出
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # 投影到原始维度
        U_tilde = self.W_o(concatenated)
        
        return U_tilde, attention_matrices

class TransformerBlockWithComponents(nn.Module):
    """
    完整的Transformer块实现，包含核心组件
    """
    def __init__(self, d_model, k, h, m, dropout=0.1):
        super().__init__()
        # 多头注意力层
        self.multihead_attn = MultiHeadAttention(d_model, k, h)
        
        # 前馈神经网络
        self.linear1 = nn.Linear(d_model, m)  # W_1 ∈ R^(m×d)
        self.linear2 = nn.Linear(m, d_model)  # W_2 ∈ R^(d×m)
        
        # 层归一化层
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.d_model = d_model
        self.m = m  # 前馈神经网络内部维度
    
    def forward(self, X):
        """
        Transformer块前向传播
        X: 输入矩阵 ∈ R^(n×d)
        """
        # 步骤1: 多头注意力层
        U_tilde, attention_matrices = self.multihead_attn(X)
        
        # 步骤2: Add & Normalize 1
        U = self.norm1(X + self.dropout1(U_tilde))
        
        # 步骤3: 前馈神经网络
        # 对于X的每一行x_i: z_i_tilde = W_2 ReLU(W_1 u_i + b_1) + b_2
        Z_tilde = self.linear2(self.dropout2(F.relu(self.linear1(U))))
        
        # 步骤4: Add & Normalize 2
        Z = self.norm2(U + self.dropout3(Z_tilde))
        
        return Z, attention_matrices

# 示例：Transformer核心组件的使用
if __name__ == "__main__":
    # 初始化参数
    n = 10  # 序列长度
    d_model = 512  # Transformer块宽度
    k = 64  # 每个注意力头的键向量维度
    h = 8  # 注意力头数量
    m = 2048  # 前馈神经网络内部维度
    
    # 创建随机输入 (n×d_model)
    X = torch.randn(n, d_model)
    
    # 测试层归一化
    layer_norm = LayerNorm(d_model)
    norm_output = layer_norm(X)
    print(f"LayerNorm - Input shape: {X.shape}, Output shape: {norm_output.shape}")
    print(f"LayerNorm - Output mean: {norm_output.mean(dim=-1)[:3]}")  # 前3个序列的均值
    print(f"LayerNorm - Output std: {norm_output.std(dim=-1)[:3]}")  # 前3个序列的标准差
    print()
    
    # 测试自注意力
    self_attention = SelfAttention(d_model, k)
    sa_output, sa_attn = self_attention(X)
    print(f"SelfAttention - Input shape: {X.shape}, Output shape: {sa_output.shape}")
    print(f"SelfAttention - Attention matrix shape: {sa_attn.shape}")
    print(f"SelfAttention - Attention matrix row sum: {sa_attn.sum(dim=-1)[:3]}")  # 验证softmax行和为1
    print()
    
    # 测试多头注意力
    multihead_attn = MultiHeadAttention(d_model, k, h)
    mha_output, mha_attn = multihead_attn(X)
    print(f"MultiHeadAttention - Input shape: {X.shape}, Output shape: {mha_output.shape}")
    print(f"MultiHeadAttention - Attention heads: {len(mha_attn)}")
    print(f"MultiHeadAttention - First head attention shape: {mha_attn[0].shape}")
    print()
    
    # 测试完整Transformer块
    transformer_block = TransformerBlockWithComponents(d_model, k, h, m)
    tb_output, tb_attn = transformer_block(X)
    print(f"TransformerBlock - Input shape: {X.shape}, Output shape: {tb_output.shape}")
    print(f"TransformerBlock - Output mean: {tb_output.mean():.4f}, Output std: {tb_output.std():.4f}")
