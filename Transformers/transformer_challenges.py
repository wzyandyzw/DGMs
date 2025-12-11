# 知识点8. Transformer的问题与解决方案

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FlashAttention(nn.Module):
    """
    快速注意力机制
    解决标准注意力机制内存效率低的问题
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # 线性层用于计算Q、K、V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # 输出线性层
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        快速注意力机制前向传播
        
        Args:
            query: 查询张量，形状为 (seq_length, batch_size, d_model)
            key: 键张量，形状为 (seq_length, batch_size, d_model)
            value: 值张量，形状为 (seq_length, batch_size, d_model)
            attn_mask: 注意力掩码，形状为 (seq_length, seq_length)
            key_padding_mask: 键填充掩码，形状为 (batch_size, seq_length)
            
        Returns:
            output: 注意力输出，形状为 (seq_length, batch_size, d_model)
        """
        # 输入形状：(seq_length, batch_size, d_model)
        B, Nt, E = query.shape
        _, Ns, _ = key.shape
        
        # 转换为 (batch_size, seq_length, d_model)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        # 计算Q、K、V
        q = self.q_proj(query) * (self.head_dim ** -0.5)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 将Q、K、V分割为多个注意力头
        # 形状变为 (batch_size, seq_length, nhead, head_dim)
        q = q.reshape(B, Nt, self.nhead, self.head_dim)
        k = k.reshape(B, Ns, self.nhead, self.head_dim)
        v = v.reshape(B, Ns, self.nhead, self.head_dim)
        
        # 交换维度，变为 (batch_size, nhead, seq_length, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 使用PyTorch的SDPA（Scaled Dot-Product Attention）实现快速注意力
        # 注意：需要PyTorch 2.0+才能使用此API
        if hasattr(F, 'scaled_dot_product_attention'):
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # 回退到标准注意力实现
            output = self._standard_attention(q, k, v, attn_mask, key_padding_mask)
        
        # 将注意力头合并
        # 形状变为 (batch_size, seq_length, d_model)
        output = output.transpose(1, 2).reshape(B, Nt, E)
        
        # 输出线性层
        output = self.out_proj(output)
        
        # 转换回 (seq_length, batch_size, d_model)
        output = output.transpose(0, 1)
        
        return output
    
    def _standard_attention(self, q, k, v, attn_mask, key_padding_mask):
        """
        标准注意力实现（回退方案）
        
        Args:
            q: 查询张量，形状为 (batch_size, nhead, seq_length_q, head_dim)
            k: 键张量，形状为 (batch_size, nhead, seq_length_k, head_dim)
            v: 值张量，形状为 (batch_size, nhead, seq_length_v, head_dim)
            attn_mask: 注意力掩码
            key_padding_mask: 键填充掩码
            
        Returns:
            output: 注意力输出
        """
        # 计算注意力分数
        attn_output_weights = torch.matmul(q, k.transpose(-2, -1))
        
        # 应用注意力掩码
        if attn_mask is not None:
            attn_output_weights += attn_mask
        
        # 应用键填充掩码
        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # 归一化
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        
        # 计算注意力输出
        output = torch.matmul(attn_output_weights, v)
        
        return output

class ReformerAttention(nn.Module):
    """
    Reformer注意力机制
    使用局部敏感哈希(LSH)减少注意力计算复杂度
    """
    def __init__(self, d_model, nhead, lsh_hash_bucket_size=32, n_hashes=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.lsh_hash_bucket_size = lsh_hash_bucket_size
        self.n_hashes = n_hashes
        
        # 线性层用于计算Q、K、V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # 输出线性层
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # LSH投影矩阵
        self.lsh_proj = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim)
            for _ in range(n_hashes)
        ])
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Reformer注意力机制前向传播
        
        Args:
            query: 查询张量，形状为 (seq_length, batch_size, d_model)
            key: 键张量，形状为 (seq_length, batch_size, d_model)
            value: 值张量，形状为 (seq_length, batch_size, d_model)
            attn_mask: 注意力掩码
            key_padding_mask: 键填充掩码
            
        Returns:
            output: 注意力输出，形状为 (seq_length, batch_size, d_model)
        """
        # 输入形状：(seq_length, batch_size, d_model)
        B, Nt, E = query.shape
        _, Ns, _ = key.shape
        
        # 转换为 (batch_size, seq_length, d_model)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        # 计算Q、K、V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 将Q、K、V分割为多个注意力头
        q = q.reshape(B, Nt, self.nhead, self.head_dim).transpose(1, 2)
        k = k.reshape(B, Ns, self.nhead, self.head_dim).transpose(1, 2)
        v = v.reshape(B, Ns, self.nhead, self.head_dim).transpose(1, 2)
        
        # 应用局部敏感哈希注意力
        output = self._lsh_attention(q, k, v)
        
        # 将注意力头合并
        output = output.transpose(1, 2).reshape(B, Nt, E)
        
        # 输出线性层
        output = self.out_proj(output)
        
        # 转换回 (seq_length, batch_size, d_model)
        output = output.transpose(0, 1)
        
        return output
    
    def _lsh_attention(self, q, k, v):
        """
        局部敏感哈希注意力实现
        
        Args:
            q: 查询张量，形状为 (batch_size, nhead, seq_length_q, head_dim)
            k: 键张量，形状为 (batch_size, nhead, seq_length_k, head_dim)
            v: 值张量，形状为 (batch_size, nhead, seq_length_v, head_dim)
            
        Returns:
            output: 注意力输出
        """
        B, H, T, D = q.shape
        
        # 对于每个哈希函数
        outputs = []
        for i in range(self.n_hashes):
            # 哈希投影
            q_proj = self.lsh_proj[i](q)
            k_proj = self.lsh_proj[i](k)
            
            # 计算哈希值
            q_hash = torch.argmax(F.softmax(q_proj, dim=-1), dim=-1)
            k_hash = torch.argmax(F.softmax(k_proj, dim=-1), dim=-1)
            
            # 简单实现：只关注同一哈希桶内的键值对
            # 注意：这是一个简化版本，实际Reformer实现更复杂
            output = torch.zeros_like(q)
            
            for b in range(B):
                for h in range(H):
                    # 对于每个哈希桶
                    for bucket in range(self.lsh_hash_bucket_size):
                        # 找到查询和键中属于当前哈希桶的位置
                        q_indices = torch.where(q_hash[b, h] == bucket)[0]
                        k_indices = torch.where(k_hash[b, h] == bucket)[0]
                        
                        if len(q_indices) > 0 and len(k_indices) > 0:
                            # 计算这些位置之间的注意力
                            q_bucket = q[b, h, q_indices]
                            k_bucket = k[b, h, k_indices]
                            v_bucket = v[b, h, k_indices]
                            
                            # 计算注意力分数
                            attn_scores = torch.matmul(q_bucket, k_bucket.transpose(-2, -1)) / np.sqrt(D)
                            attn_probs = F.softmax(attn_scores, dim=-1)
                            attn_probs = self.dropout(attn_probs)
                            
                            # 计算注意力输出
                            output_bucket = torch.matmul(attn_probs, v_bucket)
                            
                            # 将结果放回对应的位置
                            output[b, h, q_indices] = output_bucket
            
            outputs.append(output)
        
        # 平均多个哈希函数的结果
        output = torch.mean(torch.stack(outputs), dim=0)
        
        return output

class LinearAttention(nn.Module):
    """
    线性注意力机制
    将注意力计算复杂度从O(n^2)降低到O(n)
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # 线性层用于计算Q、K、V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # 输出线性层
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 用于线性注意力的激活函数
        self.activation = nn.Softmax(dim=-1)
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        线性注意力机制前向传播
        
        Args:
            query: 查询张量，形状为 (seq_length, batch_size, d_model)
            key: 键张量，形状为 (seq_length, batch_size, d_model)
            value: 值张量，形状为 (seq_length, batch_size, d_model)
            attn_mask: 注意力掩码
            key_padding_mask: 键填充掩码
            
        Returns:
            output: 注意力输出，形状为 (seq_length, batch_size, d_model)
        """
        # 输入形状：(seq_length, batch_size, d_model)
        B, Nt, E = query.shape
        _, Ns, _ = key.shape
        
        # 转换为 (batch_size, seq_length, d_model)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        # 计算Q、K、V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 将Q、K、V分割为多个注意力头
        q = q.reshape(B, Nt, self.nhead, self.head_dim).transpose(1, 2)
        k = k.reshape(B, Ns, self.nhead, self.head_dim).transpose(1, 2)
        v = v.reshape(B, Ns, self.nhead, self.head_dim).transpose(1, 2)
        
        # 应用线性注意力
        output = self._linear_attention(q, k, v)
        
        # 将注意力头合并
        output = output.transpose(1, 2).reshape(B, Nt, E)
        
        # 输出线性层
        output = self.out_proj(output)
        
        # 转换回 (seq_length, batch_size, d_model)
        output = output.transpose(0, 1)
        
        return output
    
    def _linear_attention(self, q, k, v):
        """
        线性注意力实现
        
        Args:
            q: 查询张量，形状为 (batch_size, nhead, seq_length_q, head_dim)
            k: 键张量，形状为 (batch_size, nhead, seq_length_k, head_dim)
            v: 值张量，形状为 (batch_size, nhead, seq_length_v, head_dim)
            
        Returns:
            output: 注意力输出
        """
        # 应用激活函数到Q和K
        q = self.activation(q)
        k = self.activation(k)
        
        # 计算K和V的乘积
        k_t = k.transpose(-2, -1)
        kv = torch.matmul(k_t, v)
        
        # 计算Q和KV的乘积
        qkv = torch.matmul(q, kv)
        
        # 归一化
        Z = torch.matmul(q, k_t.sum(dim=-1, keepdim=True))
        output = qkv / Z
        
        return output

class AdaptiveInput(nn.Module):
    """
    自适应输入嵌入
    解决大词汇表导致的内存问题
    """
    def __init__(self, vocab_size, d_model, cutoff=[20000, 40000, 200000], div_value=2.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.cutoff = cutoff
        self.div_value = div_value
        
        # 创建嵌入层列表
        self.embeddings = nn.ModuleList()
        
        # 第一个嵌入层：最常用的单词
        self.embeddings.append(nn.Embedding(cutoff[0], d_model))
        
        # 其他嵌入层：词汇量逐渐增加，维度逐渐减少
        for i in range(len(cutoff) - 1):
            prev_cutoff = cutoff[i]
            curr_cutoff = cutoff[i + 1]
            curr_dim = int(d_model / (div_value ** (i + 1)))
            self.embeddings.append(nn.Embedding(curr_cutoff - prev_cutoff, curr_dim))
        
        # 最后一个嵌入层：剩余单词
        if vocab_size > cutoff[-1]:
            curr_dim = int(d_model / (div_value ** len(cutoff)))
            self.embeddings.append(nn.Embedding(vocab_size - cutoff[-1], curr_dim))
        
        # 投影层：将不同维度的嵌入投影到相同维度
        self.projections = nn.ModuleList()
        for i in range(len(self.embeddings)):
            curr_dim = int(d_model / (div_value ** i)) if i < len(self.embeddings) - 1 else \
                      int(d_model / (div_value ** len(cutoff)))
            if curr_dim != d_model:
                self.projections.append(nn.Linear(curr_dim, d_model, bias=False))
            else:
                self.projections.append(nn.Identity())
    
    def forward(self, input_ids):
        """
        自适应输入嵌入前向传播
        
        Args:
            input_ids: 输入标记ID，形状为 (batch_size, seq_length)
            
        Returns:
            embeddings: 嵌入向量，形状为 (batch_size, seq_length, d_model)
        """
        batch_size, seq_length = input_ids.shape
        embeddings = torch.zeros(batch_size, seq_length, self.d_model, device=input_ids.device)
        
        # 处理每个嵌入层
        prev_cutoff = 0
        for i, (embedding, projection) in enumerate(zip(self.embeddings, self.projections)):
            curr_cutoff = self.cutoff[i] if i < len(self.cutoff) else self.vocab_size
            
            # 找到属于当前嵌入层的标记
            mask = (input_ids >= prev_cutoff) & (input_ids < curr_cutoff)
            
            if mask.any():
                # 获取当前嵌入层的标记
                curr_ids = input_ids[mask] - prev_cutoff
                
                # 计算嵌入
                curr_embeddings = embedding(curr_ids)
                curr_embeddings = projection(curr_embeddings)
                
                # 将结果放回对应的位置
                embeddings[mask] = curr_embeddings
            
            prev_cutoff = curr_cutoff
        
        return embeddings

class SparseTransformer(nn.Module):
    """
    稀疏Transformer
    使用局部注意力和滑动窗口注意力解决长序列问题
    """
    def __init__(self, d_model, nhead, num_layers=6, window_size=50, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.window_size = window_size
        
        # 创建稀疏Transformer层
        self.layers = nn.ModuleList([
            SparseTransformerLayer(d_model, nhead, window_size, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        稀疏Transformer前向传播
        
        Args:
            src: 输入序列，形状为 (seq_length, batch_size, d_model)
            src_mask: 源序列掩码
            src_key_padding_mask: 源序列填充掩码
            
        Returns:
            output: 输出序列，形状与输入相同
        """
        output = src
        
        for layer in self.layers:
            output = layer(output, src_mask, src_key_padding_mask)
        
        return output

class SparseTransformerLayer(nn.Module):
    """
    稀疏Transformer层
    """
    def __init__(self, d_model, nhead, window_size, dropout=0.1):
        super().__init__()
        # 滑动窗口注意力
        self.window_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(4 * d_model, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.window_size = window_size
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        稀疏Transformer层前向传播
        
        Args:
            src: 输入序列，形状为 (seq_length, batch_size, d_model)
            src_mask: 源序列掩码
            src_key_padding_mask: 源序列填充掩码
            
        Returns:
            output: 输出序列
        """
        # 滑动窗口注意力
        src2 = self._sliding_window_attention(src, src_mask, src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src
    
    def _sliding_window_attention(self, src, src_mask, src_key_padding_mask):
        """
        滑动窗口注意力实现
        
        Args:
            src: 输入序列
            src_mask: 源序列掩码
            src_key_padding_mask: 源序列填充掩码
            
        Returns:
            output: 注意力输出
        """
        seq_length = src.size(0)
        output = torch.zeros_like(src)
        
        # 滑动窗口
        for i in range(0, seq_length, self.window_size):
            # 当前窗口的起始和结束位置
            start = i
            end = min(i + self.window_size, seq_length)
            
            # 获取当前窗口的输入
            window_src = src[start:end]
            
            # 计算窗口内的注意力
            window_output = self.window_attn(
                window_src, window_src, window_src,
                attn_mask=src_mask[start:end, start:end] if src_mask is not None else None,
                key_padding_mask=src_key_padding_mask[:, start:end] if src_key_padding_mask is not None else None
            )[0]
            
            # 将结果放回对应的位置
            output[start:end] = window_output
        
        return output

# 示例：Transformer问题与解决方案的使用
if __name__ == "__main__":
    # 初始化参数
    d_model = 512
    nhead = 8
    seq_length = 1000  # 长序列示例
    batch_size = 2
    vocab_size = 10000
    
    print("=== Transformer的问题与解决方案示例 ===")
    
    print("\n1. 内存效率问题：FlashAttention")
    flash_attention = FlashAttention(d_model, nhead)
    print(f"  FlashAttention参数数量: {sum(p.numel() for p in flash_attention.parameters()):,}")
    
    # 创建长序列输入
    query = torch.randn(seq_length, batch_size, d_model)
    key = torch.randn(seq_length, batch_size, d_model)
    value = torch.randn(seq_length, batch_size, d_model)
    
    # FlashAttention前向传播
    output = flash_attention(query, key, value)
    print(f"  输入序列长度: {seq_length}")
    print(f"  输出形状: {output.shape}")
    
    print("\n2. 长序列问题：线性注意力")
    linear_attention = LinearAttention(d_model, nhead)
    print(f"  LinearAttention参数数量: {sum(p.numel() for p in linear_attention.parameters()):,}")
    
    # LinearAttention前向传播
    output = linear_attention(query, key, value)
    print(f"  输入序列长度: {seq_length}")
    print(f"  输出形状: {output.shape}")
    
    print("\n3. 大词汇表问题：自适应输入嵌入")
    cutoff = [20000, 40000, 200000]
    adaptive_input = AdaptiveInput(vocab_size, d_model, cutoff)
    print(f"  AdaptiveInput参数数量: {sum(p.numel() for p in adaptive_input.parameters()):,}")
    
    # 创建输入ID
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # AdaptiveInput前向传播
    embeddings = adaptive_input(input_ids)
    print(f"  词汇表大小: {vocab_size}")
    print(f"  嵌入形状: {embeddings.shape}")
    
    print("\n4. 稀疏注意力：滑动窗口注意力")
    sparse_transformer = SparseTransformer(d_model, nhead, num_layers=2, window_size=50)
    print(f"  SparseTransformer参数数量: {sum(p.numel() for p in sparse_transformer.parameters()):,}")
    
    # SparseTransformer前向传播
    output = sparse_transformer(query)
    print(f"  输入序列长度: {seq_length}")
    print(f"  输出形状: {output.shape}")
    print(f"  滑动窗口大小: {sparse_transformer.window_size}")
    
    print("\n5. Reformer注意力：局部敏感哈希")
    reformer_attention = ReformerAttention(d_model, nhead)
    print(f"  ReformerAttention参数数量: {sum(p.numel() for p in reformer_attention.parameters()):,}")
    
    # 注意：为了演示，我们使用较短的序列
    short_query = torch.randn(100, batch_size, d_model)
    short_key = torch.randn(100, batch_size, d_model)
    short_value = torch.randn(100, batch_size, d_model)
    
    # ReformerAttention前向传播
    output = reformer_attention(short_query, short_key, short_value)
    print(f"  输入序列长度: 100")
    print(f"  输出形状: {output.shape}")
    
    print("\n=== Transformer主要问题与解决方案总结 ===")
    print("  1. 内存效率问题：")
    print("     - FlashAttention：使用分块和重新排序优化内存访问")
    print("     - 降低注意力计算的内存复杂度")
    
    print("  2. 长序列问题：")
    print("     - 线性注意力：将注意力计算复杂度从O(n^2)降低到O(n)")
    print("     - SparseTransformer：使用局部注意力和滑动窗口注意力")
    print("     - Reformer：使用局部敏感哈希(LSH)减少计算量")
    
    print("  3. 大词汇表问题：")
    print("     - 自适应输入嵌入：为不同频率的单词分配不同维度的嵌入")
    print("     - 减少大词汇表导致的内存占用")
    
    print("  4. 计算效率问题：")
    print("     - 蒸馏：使用较小的模型学习较大模型的知识")
    print("     - 剪枝：移除不重要的模型参数")
    print("     - 量化：降低模型参数的精度")
    
    print("  5. 过拟合问题：")
    print("     - Dropout：在训练过程中随机丢弃神经元")
    print("     - 数据增强：增加训练数据的多样性")
    print("     - 正则化：L1/L2正则化")
