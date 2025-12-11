# 知识点7. Transformer变体（BERT与GPT）

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BERTEmbedding(nn.Module):
    """
    BERT嵌入层
    组合了词嵌入、位置嵌入和段嵌入
    """
    def __init__(self, vocab_size, d_model, max_seq_length, dropout=0.1):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)  # 0和1两个段
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
    
    def forward(self, input_ids, segment_ids=None):
        """
        BERT嵌入层前向传播
        
        Args:
            input_ids: 输入标记ID，形状为 (batch_size, seq_length)
            segment_ids: 段ID，形状为 (batch_size, seq_length)
            
        Returns:
            embeddings: 组合嵌入，形状为 (batch_size, seq_length, d_model)
        """
        batch_size, seq_length = input_ids.shape
        
        # 位置索引
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # 如果没有提供段ID，则默认为0
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids, device=input_ids.device)
        
        # 获取嵌入
        word_embeddings = self.word_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        segment_embeddings = self.segment_embedding(segment_ids)
        
        # 组合嵌入
        embeddings = word_embeddings + position_embeddings + segment_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BERTLayer(nn.Module):
    """
    BERT层
    包含多头自注意力和前馈网络
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        BERT层前向传播
        
        Args:
            src: 输入序列，形状为 (seq_length, batch_size, d_model)
            src_mask: 源序列的掩码，形状为 (seq_length, seq_length)
            src_key_padding_mask: 源序列的填充掩码，形状为 (batch_size, seq_length)
            
        Returns:
            src: 输出序列，形状与输入相同
        """
        # 多头自注意力
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class BERT(nn.Module):
    """
    BERT模型
    由多个BERT层堆叠而成
    """
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=12,
                 dim_feedforward=3072, dropout=0.1, max_seq_length=512):
        super().__init__()
        # 嵌入层
        self.embedding = BERTEmbedding(vocab_size, d_model, max_seq_length, dropout)
        
        # BERT层
        self.layers = nn.ModuleList([BERTLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        
        # 池化层（用于分类任务）
        self.pooler = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()
    
    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        """
        BERT模型前向传播
        
        Args:
            input_ids: 输入标记ID，形状为 (batch_size, seq_length)
            segment_ids: 段ID，形状为 (batch_size, seq_length)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_length)
            
        Returns:
            last_hidden_state: 最后一层的隐藏状态，形状为 (batch_size, seq_length, d_model)
            pooled_output: 池化后的输出，形状为 (batch_size, d_model)
        """
        batch_size, seq_length = input_ids.shape
        
        # 嵌入层
        embeddings = self.embedding(input_ids, segment_ids)
        
        # 转换为 (seq_length, batch_size, d_model) 格式以适应MultiheadAttention
        embeddings = embeddings.transpose(0, 1)
        
        # 注意力掩码
        if attention_mask is not None:
            # 将注意力掩码转换为适合MultiheadAttention的格式
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None
        
        # 通过BERT层
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=key_padding_mask)
        
        # 转换回 (batch_size, seq_length, d_model) 格式
        last_hidden_state = hidden_states.transpose(0, 1)
        
        # 池化层：取[CLS]标记的隐藏状态
        pooled_output = self.pooler(last_hidden_state[:, 0, :])
        pooled_output = self.activation(pooled_output)
        
        return last_hidden_state, pooled_output

class BERTForMaskedLM(nn.Module):
    """
    BERT用于掩码语言模型（MLM）
    """
    def __init__(self, bert_model, vocab_size):
        super().__init__()
        self.bert = bert_model
        
        # MLM输出层
        self.mlm_head = nn.Sequential(
            nn.Linear(bert_model.embedding.d_model, bert_model.embedding.d_model),
            nn.ReLU(),
            nn.LayerNorm(bert_model.embedding.d_model),
            nn.Linear(bert_model.embedding.d_model, vocab_size)
        )
    
    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        """
        BERT MLM前向传播
        
        Args:
            input_ids: 输入标记ID，形状为 (batch_size, seq_length)
            segment_ids: 段ID，形状为 (batch_size, seq_length)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_length)
            
        Returns:
            prediction_scores: 预测分数，形状为 (batch_size, seq_length, vocab_size)
        """
        last_hidden_state, _ = self.bert(input_ids, segment_ids, attention_mask)
        prediction_scores = self.mlm_head(last_hidden_state)
        
        return prediction_scores

class BERTForNextSentencePrediction(nn.Module):
    """
    BERT用于下一句预测（NSP）
    """
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        
        # NSP输出层
        self.nsp_head = nn.Linear(bert_model.embedding.d_model, 2)  # 0或1
    
    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        """
        BERT NSP前向传播
        
        Args:
            input_ids: 输入标记ID，形状为 (batch_size, seq_length)
            segment_ids: 段ID，形状为 (batch_size, seq_length)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_length)
            
        Returns:
            seq_relationship_scores: 序列关系分数，形状为 (batch_size, 2)
        """
        _, pooled_output = self.bert(input_ids, segment_ids, attention_mask)
        seq_relationship_scores = self.nsp_head(pooled_output)
        
        return seq_relationship_scores

class GPTEmbedding(nn.Module):
    """
    GPT嵌入层
    组合了词嵌入和位置嵌入
    """
    def __init__(self, vocab_size, d_model, max_seq_length, dropout=0.1):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
    
    def forward(self, input_ids):
        """
        GPT嵌入层前向传播
        
        Args:
            input_ids: 输入标记ID，形状为 (batch_size, seq_length)
            
        Returns:
            embeddings: 组合嵌入，形状为 (batch_size, seq_length, d_model)
        """
        batch_size, seq_length = input_ids.shape
        
        # 位置索引
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # 获取嵌入
        word_embeddings = self.word_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        
        # 组合嵌入
        embeddings = word_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        return embeddings

class GPTLayer(nn.Module):
    """
    GPT层
    包含因果多头自注意力和前馈网络
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # 因果多头自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, causal_mask=None):
        """
        GPT层前向传播
        
        Args:
            src: 输入序列，形状为 (seq_length, batch_size, d_model)
            causal_mask: 因果掩码，形状为 (seq_length, seq_length)
            
        Returns:
            src: 输出序列，形状与输入相同
        """
        # 多头自注意力（使用因果掩码防止看到未来信息）
        src2 = self.self_attn(src, src, src, attn_mask=causal_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class GPT(nn.Module):
    """
    GPT模型
    由多个GPT层堆叠而成
    """
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=12,
                 dim_feedforward=3072, dropout=0.1, max_seq_length=512):
        super().__init__()
        # 嵌入层
        self.embedding = GPTEmbedding(vocab_size, d_model, max_seq_length, dropout)
        
        # GPT层
        self.layers = nn.ModuleList([GPTLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        
        # 输出层
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, attention_mask=None):
        """
        GPT模型前向传播
        
        Args:
            input_ids: 输入标记ID，形状为 (batch_size, seq_length)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_length)
            
        Returns:
            logits: 对数概率，形状为 (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = input_ids.shape
        
        # 嵌入层
        embeddings = self.embedding(input_ids)
        
        # 转换为 (seq_length, batch_size, d_model) 格式以适应MultiheadAttention
        embeddings = embeddings.transpose(0, 1)
        
        # 生成因果掩码
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=input_ids.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        
        # 注意力掩码
        if attention_mask is not None:
            # 将注意力掩码转换为适合MultiheadAttention的格式
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None
        
        # 通过GPT层
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask=causal_mask)
        
        # 转换回 (batch_size, seq_length, d_model) 格式
        hidden_states = hidden_states.transpose(0, 1)
        
        # 输出层
        logits = self.lm_head(hidden_states)
        
        return logits

def gpt_generate(model, input_ids, max_length, temperature=1.0, top_k=50, top_p=0.9, device=None):
    """
    GPT生成文本
    
    Args:
        model: GPT模型
        input_ids: 输入标记ID，形状为 (batch_size, seq_length)
        max_length: 生成的最大长度
        temperature: 温度参数，控制生成的随机性
        top_k: 只考虑概率最高的k个标记
        top_p: 只考虑累积概率超过p的标记
        device: 设备
        
    Returns:
        generated_ids: 生成的标记ID，形状为 (batch_size, max_length)
    """
    if device is None:
        device = input_ids.device
    
    model.eval()
    generated_ids = input_ids
    
    for _ in range(max_length - input_ids.size(1)):
        # 前向传播
        logits = model(generated_ids)
        
        # 只考虑最后一个标记的输出
        next_token_logits = logits[:, -1, :] / temperature
        
        # Top-k采样
        if top_k > 0:
            next_token_logits = torch.topk(next_token_logits, top_k)[0]
        
        # Top-p采样（核采样）
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[:, indices_to_remove] = float('-inf')
        
        # 概率分布
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        
        # 采样下一个标记
        next_token = torch.multinomial(next_token_probs, num_samples=1)
        
        # 添加到生成序列
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    
    return generated_ids

# 示例：Transformer变体的使用
if __name__ == "__main__":
    # 初始化参数
    vocab_size = 10000
    d_model = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    max_seq_length = 256
    batch_size = 2
    
    print("=== BERT模型示例 ===")
    
    # 创建BERT模型
    bert = BERT(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, max_seq_length)
    print(f"  BERT模型参数数量: {sum(p.numel() for p in bert.parameters()):,}")
    
    # 创建示例输入
    input_ids = torch.randint(0, vocab_size, (batch_size, max_seq_length))
    segment_ids = torch.randint(0, 2, (batch_size, max_seq_length))
    attention_mask = torch.ones(batch_size, max_seq_length)
    
    # BERT前向传播
    last_hidden_state, pooled_output = bert(input_ids, segment_ids, attention_mask)
    
    print(f"  输入ID形状: {input_ids.shape}")
    print(f"  最后一层隐藏状态形状: {last_hidden_state.shape}")
    print(f"  池化输出形状: {pooled_output.shape}")
    
    # BERT用于掩码语言模型
    bert_mlm = BERTForMaskedLM(bert, vocab_size)
    mlm_output = bert_mlm(input_ids, segment_ids, attention_mask)
    print(f"  MLM输出形状: {mlm_output.shape}")
    
    # BERT用于下一句预测
    bert_nsp = BERTForNextSentencePrediction(bert)
    nsp_output = bert_nsp(input_ids, segment_ids, attention_mask)
    print(f"  NSP输出形状: {nsp_output.shape}")
    
    print("\n=== GPT模型示例 ===")
    
    # 创建GPT模型
    gpt = GPT(vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, max_seq_length)
    print(f"  GPT模型参数数量: {sum(p.numel() for p in gpt.parameters()):,}")
    
    # 创建示例输入
    input_ids_gpt = torch.randint(0, vocab_size, (batch_size, 10))
    
    # GPT前向传播
    logits = gpt(input_ids_gpt)
    
    print(f"  输入ID形状: {input_ids_gpt.shape}")
    print(f"  输出对数概率形状: {logits.shape}")
    
    # GPT生成文本
    print("  生成文本示例：")
    generated_ids = gpt_generate(gpt, input_ids_gpt, max_length=30, temperature=0.7, top_k=30, top_p=0.85)
    print(f"  生成的标记ID形状: {generated_ids.shape}")
    print(f"  输入序列长度: {input_ids_gpt.size(1)}")
    print(f"  生成的序列长度: {generated_ids.size(1)}")
    
    print("\n=== BERT与GPT的主要区别 ===")
    print("  1. 训练目标:")
    print("     - BERT: 掩码语言模型(MLM) + 下一句预测(NSP)")
    print("     - GPT: 因果语言模型(CLM)")
    print("  2. 架构:")
    print("     - BERT: 仅使用Transformer编码器")
    print("     - GPT: 仅使用Transformer解码器")
    print("  3. 上下文理解:")
    print("     - BERT: 双向上下文理解")
    print("     - GPT: 单向（自回归）上下文理解")
    print("  4. 应用场景:")
    print("     - BERT: 适合分类、问答、命名实体识别等任务")
    print("     - GPT: 适合文本生成、对话等任务")
