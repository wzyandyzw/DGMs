# 知识点5. 解码器结构

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    包含三个子层：掩码自注意力、编码器-解码器注意力、前馈神经网络
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 掩码自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 编码器-解码器注意力层
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        解码器层前向传播
        
        Args:
            tgt: 解码器输入，形状为 (seq_length, batch_size, d_model)
            memory: 编码器输出，形状为 (seq_length, batch_size, d_model)
            tgt_mask: 目标序列的掩码，形状为 (tgt_seq_length, tgt_seq_length)
            memory_mask: 编码器输出的掩码
            tgt_key_padding_mask: 目标序列的填充掩码
            memory_key_padding_mask: 编码器输出的填充掩码
            
        Returns:
            tgt: 解码器层输出，形状与输入相同
        """
        # 步骤1: 掩码自注意力层
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 步骤2: 编码器-解码器注意力层
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 步骤3: 前馈神经网络
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

class TransformerDecoder(nn.Module):
    """
    Transformer解码器
    由多个解码器层堆叠而成
    """
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        解码器前向传播
        
        Args:
            tgt: 解码器输入，形状为 (seq_length, batch_size, d_model)
            memory: 编码器输出，形状为 (seq_length, batch_size, d_model)
            tgt_mask: 目标序列的掩码，形状为 (tgt_seq_length, tgt_seq_length)
            memory_mask: 编码器输出的掩码
            tgt_key_padding_mask: 目标序列的填充掩码
            memory_key_padding_mask: 编码器输出的填充掩码
            
        Returns:
            output: 解码器输出，形状与输入相同
        """
        output = tgt
        
        # 依次通过每个解码器层
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
        
        # 应用最终的层归一化（如果提供）
        if self.norm is not None:
            output = self.norm(output)
        
        return output

def generate_square_subsequent_mask(sz):
    """
    生成下三角掩码，用于防止解码器看到未来的信息
    
    Args:
        sz: 序列长度
        
    Returns:
        mask: 下三角掩码，形状为 (sz, sz)
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class OutputLayer(nn.Module):
    """
    解码器输出层
    将解码器输出转换为词汇表概率分布
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        输出层前向传播
        
        Args:
            x: 解码器输出，形状为 (seq_length, batch_size, d_model)
            
        Returns:
            logits: 对数概率，形状为 (seq_length, batch_size, vocab_size)
            prob: 概率分布，形状为 (seq_length, batch_size, vocab_size)
        """
        logits = self.linear(x)
        prob = F.softmax(logits, dim=-1)
        return logits, prob

def greedy_decoding(decoder, output_layer, memory, start_token, max_length, device):
    """
    贪婪解码算法
    
    Args:
        decoder: Transformer解码器
        output_layer: 输出层
        memory: 编码器输出
        start_token: 起始标记
        max_length: 最大解码长度
        device: 设备
        
    Returns:
        decoded_sequence: 解码后的序列
    """
    batch_size = memory.size(1)
    
    # 初始化解码序列，只包含起始标记
    decoded_sequence = torch.full((1, batch_size), start_token, device=device, dtype=torch.long)
    
    # 对于每个解码步骤
    for _ in range(max_length - 1):
        # 生成掩码
        tgt_mask = generate_square_subsequent_mask(decoded_sequence.size(0)).to(device)
        
        # 解码器前向传播
        decoder_output = decoder(decoded_sequence, memory, tgt_mask=tgt_mask)
        
        # 输出层
        logits, prob = output_layer(decoder_output)
        
        # 选择概率最高的标记
        next_token = torch.argmax(prob[-1, :, :], dim=-1).unsqueeze(0)
        
        # 将新标记添加到解码序列
        decoded_sequence = torch.cat([decoded_sequence, next_token], dim=0)
    
    return decoded_sequence

# 示例：解码器结构的使用
if __name__ == "__main__":
    # 初始化参数
    d_model = 512
    nhead = 8
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    vocab_size = 10000
    
    # 创建示例输入
    batch_size = 2
    src_seq_length = 15
    tgt_seq_length = 10
    
    # 假设我们有编码器输出 (memory)
    memory = torch.randn(src_seq_length, batch_size, d_model)
    
    # 目标序列输入
    tgt = torch.randint(0, vocab_size, (tgt_seq_length, batch_size))
    
    # 创建嵌入层
    tgt_embedding = nn.Embedding(vocab_size, d_model)
    
    # 创建位置编码
    from positional_encoding import PositionalEncoding
    pos_encoding = PositionalEncoding(d_model, max_seq_length=5000)
    
    # 生成目标序列掩码
    tgt_mask = generate_square_subsequent_mask(tgt_seq_length)
    
    # 嵌入和位置编码
    tgt_embedded = tgt_embedding(tgt) * np.sqrt(d_model)
    tgt_embedded = pos_encoding(tgt_embedded)
    
    # 创建解码器层
    decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward, dropout)
    
    # 创建解码器
    decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
    
    # 解码器前向传播
    decoder_output = decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
    
    print("解码器结构：")
    print(f"  目标序列输入形状: {tgt.shape}")
    print(f"  嵌入后形状: {tgt_embedded.shape}")
    print(f"  编码器输出形状: {memory.shape}")
    print(f"  解码器输出形状: {decoder_output.shape}")
    print()
    
    # 创建输出层
    output_layer = OutputLayer(d_model, vocab_size)
    
    # 输出层前向传播
    logits, prob = output_layer(decoder_output)
    
    print("输出层：")
    print(f"  对数概率形状: {logits.shape}")
    print(f"  概率分布形状: {prob.shape}")
    print(f"  词汇表大小: {vocab_size}")
    print()
    
    # 贪婪解码示例
    print("贪婪解码示例：")
    device = torch.device("cpu")
    start_token = 1  # 假设1是起始标记
    max_length = 20
    
    # 转换为适当的设备
    memory = memory.to(device)
    decoder = decoder.to(device)
    output_layer = output_layer.to(device)
    
    # 执行贪婪解码
    decoded_sequence = greedy_decoding(decoder, output_layer, memory, start_token, max_length, device)
    
    print(f"  解码序列形状: {decoded_sequence.shape}")
    print(f"  解码序列: {decoded_sequence.squeeze().tolist()}")
    print()
    
    print("解码器结构特点：")
    print("  - 使用掩码自注意力防止未来信息泄露")
    print("  - 通过编码器-解码器注意力关注输入序列的相关部分")
    print("  - 堆叠多个解码器层以增加模型容量")
    print("  - 输出层将解码器输出转换为词汇表概率分布")
    print("  - 通常使用贪婪解码或束搜索生成最终输出")
