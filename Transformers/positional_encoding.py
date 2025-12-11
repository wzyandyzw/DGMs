# 知识点4. 位置编码

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    """
    正弦余弦位置编码
    Transformer论文中使用的位置编码方法
    """
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # 计算角度频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # 偶数位置使用正弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位置使用余弦函数
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度，并将位置编码注册为缓冲区（不参与梯度更新）
        pe = pe.unsqueeze(0)  # (1, max_seq_length, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入嵌入张量，形状为 (batch_size, seq_length, d_model)
            
        Returns:
            x: 嵌入与位置编码相加后的结果，形状与输入相同
        """
        # 只使用输入序列长度对应的位置编码部分
        seq_length = x.size(1)
        x = x + self.pe[:, :seq_length, :]
        return x

class LearnablePositionalEncoding(nn.Module):
    """
    可学习的位置编码
    位置编码作为模型参数进行学习
    """
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 创建可学习的位置编码参数
        self.pe = nn.Parameter(torch.randn(1, max_seq_length, d_model))
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入嵌入张量，形状为 (batch_size, seq_length, d_model)
            
        Returns:
            x: 嵌入与位置编码相加后的结果，形状与输入相同
        """
        seq_length = x.size(1)
        x = x + self.pe[:, :seq_length, :]
        return x

class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE)
    一种相对位置编码方法，通过旋转操作将位置信息注入到Q和K中
    """
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 计算旋转角度
        theta = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('theta', theta)
    
    def forward(self, q, k):
        """
        前向传播
        
        Args:
            q: 查询张量，形状为 (batch_size, seq_length, d_model)
            k: 键张量，形状为 (batch_size, seq_length, d_model)
            
        Returns:
            q_rope: 应用旋转位置编码后的查询张量
            k_rope: 应用旋转位置编码后的键张量
        """
        batch_size, seq_length, d_model = q.size()
        
        # 创建位置索引
        position = torch.arange(seq_length, dtype=torch.float, device=q.device).unsqueeze(1)
        
        # 计算旋转角度矩阵
        angle = position * self.theta
        
        # 创建旋转矩阵的元素
        cos_angle = torch.cos(angle).repeat(1, 2)  # (seq_length, d_model)
        sin_angle = torch.sin(angle).repeat(1, 2)  # (seq_length, d_model)
        
        # 为每个批次复制旋转矩阵
        cos_angle = cos_angle.unsqueeze(0).repeat(batch_size, 1, 1)
        sin_angle = sin_angle.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 应用旋转操作
        q_rope = self._rotate_half(q) * sin_angle + q * cos_angle
        k_rope = self._rotate_half(k) * sin_angle + k * cos_angle
        
        return q_rope, k_rope
    
    def _rotate_half(self, x):
        """
        将张量的后半部分维度旋转180度
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

def plot_positional_encoding(pe, seq_length=100, d_model=512):
    """
    可视化位置编码
    
    Args:
        pe: 位置编码矩阵，形状为 (1, seq_length, d_model)
        seq_length: 要显示的序列长度
        d_model: 嵌入维度
    """
    pe = pe[0, :seq_length, :].detach().numpy()
    
    plt.figure(figsize=(12, 8))
    plt.imshow(pe, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')
    plt.title('Positional Encoding Matrix')
    plt.savefig('positional_encoding_matrix.png')
    plt.close()
    
    # 绘制几个位置的编码曲线
    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.plot(pe[i, :], label=f'Position {i+1}')
    plt.legend()
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Positional Encoding Value')
    plt.title('Positional Encoding for Different Positions')
    plt.savefig('positional_encoding_curves.png')
    plt.close()
    
    print("位置编码可视化已保存")

def positional_distance_analysis(pe, position1, position2):
    """
    分析两个位置之间的位置编码距离
    
    Args:
        pe: 位置编码矩阵
        position1: 第一个位置
        position2: 第二个位置
        
    Returns:
        distance: 两个位置编码之间的欧几里得距离
    """
    pe1 = pe[0, position1, :]
    pe2 = pe[0, position2, :]
    distance = torch.norm(pe1 - pe2).item()
    return distance

# 示例：位置编码的使用
if __name__ == "__main__":
    # 初始化参数
    d_model = 512
    max_seq_length = 100
    seq_length = 20
    batch_size = 2
    
    # 创建随机嵌入
    embeddings = torch.randn(batch_size, seq_length, d_model)
    
    # 测试正弦余弦位置编码
    pos_encoding = PositionalEncoding(d_model, max_seq_length)
    pe_output = pos_encoding(embeddings)
    
    print("正弦余弦位置编码：")
    print(f"  输入形状: {embeddings.shape}")
    print(f"  输出形状: {pe_output.shape}")
    print()
    
    # 可视化位置编码
    plot_positional_encoding(pos_encoding.pe, seq_length=max_seq_length, d_model=d_model)
    
    # 测试可学习的位置编码
    learnable_pe = LearnablePositionalEncoding(d_model, max_seq_length)
    lpe_output = learnable_pe(embeddings)
    
    print("可学习的位置编码：")
    print(f"  输入形状: {embeddings.shape}")
    print(f"  输出形状: {lpe_output.shape}")
    print()
    
    # 测试旋转位置编码
    rope = RotaryPositionalEmbedding(d_model)
    q = torch.randn(batch_size, seq_length, d_model)
    k = torch.randn(batch_size, seq_length, d_model)
    q_rope, k_rope = rope(q, k)
    
    print("旋转位置编码：")
    print(f"  查询张量形状: {q.shape}")
    print(f"  键张量形状: {k.shape}")
    print(f"  应用RoPE后的查询形状: {q_rope.shape}")
    print(f"  应用RoPE后的键形状: {k_rope.shape}")
    print()
    
    # 位置编码距离分析
    print("位置编码距离分析：")
    for i in range(5):
        for j in range(i+1, i+3):
            if j < max_seq_length:
                distance = positional_distance_analysis(pos_encoding.pe, i, j)
                print(f"  位置 {i} 和位置 {j} 之间的欧几里得距离: {distance:.4f}")
    
    print()
    print("位置编码特点：")
    print("  - 正弦余弦位置编码：固定的数学模式，便于外推到更长序列")
    print("  - 可学习位置编码：模型自动学习最优的位置表示")
    print("  - 旋转位置编码：相对位置编码，具有良好的外推性能")
    print("  - 位置编码为模型提供了序列中元素的相对位置信息")
    print("  - 位置编码的维度与嵌入维度相同，便于直接相加")
