# 知识点1. Transformer基本定义与架构

import numpy as np
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """
    Transformer块定义
    A Transformer block is a family of functions b : R^(n×d) → R^(n×d)
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Transformer块前向传播
        """
        # 自注意力层
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # 残差连接与层归一化
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # 前馈神经网络
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        # 残差连接与层归一化
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Transformer(nn.Module):
    """
    Transformer定义
    A Transformer is a stack of Transformer blocks: f = b^L ∘ ... ∘ b^1 : R^(n×d) → R^(n×d)
    """
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 创建Transformer块栈
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Transformer前向传播
        """
        # 依次通过每个Transformer块
        output = src
        for mod in self.encoder_layers:
            output = mod(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output

class PrefixTransformer(nn.Module):
    """
    使用Transformer参数化NADE模型
    使用掩码定义前缀函数
    """
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = Transformer(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        """
        前向传播
        """
        # 词嵌入
        embedded = self.embedding(x)
        # 通过Transformer
        transformer_output = self.transformer(embedded, src_mask=mask)
        # 输出层
        logits = self.fc(transformer_output)
        return logits
    
    def prefix_function(self, x, t):
        """
        前缀函数定义：f_t(x_1, ..., x_{t-1}) = f(m_t ⊙ x)
        """
        # 创建掩码
        n = x.size(1)
        m_t = torch.zeros((n, n), device=x.device)
        m_t[:, :t] = 1  # 只保留前t个位置
        
        # 扩展掩码维度以匹配多头注意力的要求
        mask = m_t.unsqueeze(0).unsqueeze(0)
        mask = (mask == 0).float() * -1e9  # 转换为注意力掩码格式
        
        # 应用前缀函数
        output = self.forward(x, mask=mask)
        return output[:, t-1, :]  # 返回第t个位置的输出
    
    def conditional_prob(self, x, t):
        """
        条件概率：p_t(x_t|x_1, ..., x_{t-1}) = Cat(softmax(f_t(x_1, ..., x_{t-1})))
        """
        logits = self.prefix_function(x, t)
        prob = torch.softmax(logits, dim=-1)
        return prob

# 示例：Transformer基本架构的简单使用
if __name__ == "__main__":
    # 初始化模型参数
    vocab_size = 1000
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    
    # 创建模型实例
    model = PrefixTransformer(vocab_size, d_model, nhead, num_encoder_layers)
    
    # 随机输入序列 (batch_size, seq_length)
    batch_size = 2
    seq_length = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # 计算条件概率 p(x_5|x_1, x_2, x_3, x_4)
    t = 5
    prob = model.conditional_prob(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Conditional probability shape for t={t}: {prob.shape}")
    print(f"Sum of probabilities (should be 1): {prob.sum(dim=-1)}")
