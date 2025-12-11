# 知识点6. 超参数选择

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class HyperparameterConfig:
    """
    Transformer超参数配置类
    提供不同规模的超参数配置
    """
    def __init__(self, config_name="base"):
        """
        初始化超参数配置
        
        Args:
            config_name: 配置名称 (base, small, large, tiny)
        """
        self.config = self._get_config(config_name)
    
    def _get_config(self, config_name):
        """
        获取预定义的超参数配置
        
        Args:
            config_name: 配置名称
            
        Returns:
            config: 超参数字典
        """
        configs = {
            "tiny": {
                "d_model": 128,
                "nhead": 4,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "dim_feedforward": 512,
                "dropout": 0.1,
                "learning_rate": 1e-4,
                "batch_size": 16,
                "max_seq_length": 128
            },
            "small": {
                "d_model": 256,
                "nhead": 4,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "dim_feedforward": 1024,
                "dropout": 0.1,
                "learning_rate": 5e-5,
                "batch_size": 32,
                "max_seq_length": 256
            },
            "base": {
                "d_model": 512,
                "nhead": 8,
                "num_encoder_layers": 6,
                "num_decoder_layers": 6,
                "dim_feedforward": 2048,
                "dropout": 0.1,
                "learning_rate": 2e-5,
                "batch_size": 64,
                "max_seq_length": 512
            },
            "large": {
                "d_model": 1024,
                "nhead": 16,
                "num_encoder_layers": 12,
                "num_decoder_layers": 12,
                "dim_feedforward": 4096,
                "dropout": 0.1,
                "learning_rate": 1e-5,
                "batch_size": 32,
                "max_seq_length": 512
            }
        }
        
        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
        
        return configs[config_name]
    
    def __getattr__(self, name):
        """
        允许通过属性访问超参数
        
        Args:
            name: 超参数名称
            
        Returns:
            value: 超参数值
        """
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"Hyperparameter {name} not found")
    
    def to_dict(self):
        """
        将配置转换为字典
        
        Returns:
            config_dict: 超参数字典
        """
        return self.config.copy()

def calculate_model_size(hyperparams, vocab_size):
    """
    估计Transformer模型的参数量
    
    Args:
        hyperparams: 超参数配置
        vocab_size: 词汇表大小
        
    Returns:
        total_params: 总参数量
        param_details: 参数详情字典
    """
    # 嵌入层参数
    embed_params = vocab_size * hyperparams.d_model
    
    # 位置编码参数（如果是可学习的）
    pos_embed_params = hyperparams.max_seq_length * hyperparams.d_model
    
    # 注意力层参数
    # 每个注意力头的参数: 3 * d_model^2 (Q, K, V 线性层) + d_model (输出线性层)
    attention_params_per_head = 3 * hyperparams.d_model * hyperparams.d_model + hyperparams.d_model
    total_attention_params = attention_params_per_head * hyperparams.nhead
    
    # 前馈网络参数
    ffn_params = 2 * hyperparams.d_model * hyperparams.dim_feedforward
    
    # 每层的参数（编码器和解码器层结构相同）
    layer_params = total_attention_params + ffn_params + 2 * hyperparams.d_model  # 层归一化参数
    
    # 编码器参数
    encoder_params = hyperparams.num_encoder_layers * layer_params
    
    # 解码器参数（额外包含编码器-解码器注意力层）
    decoder_layer_params = layer_params + total_attention_params  # 额外的注意力层
    decoder_params = hyperparams.num_decoder_layers * decoder_layer_params
    
    # 输出层参数
    output_params = hyperparams.d_model * vocab_size
    
    # 总参数
    total_params = embed_params + pos_embed_params + encoder_params + decoder_params + output_params
    
    param_details = {
        "embedding": embed_params,
        "positional_embedding": pos_embed_params,
        "encoder": encoder_params,
        "decoder": decoder_params,
        "output": output_params,
        "total": total_params
    }
    
    return total_params, param_details

def get_optimizer(model, hyperparams, optimizer_name="adam"):
    """
    获取优化器
    
    Args:
        model: Transformer模型
        hyperparams: 超参数配置
        optimizer_name: 优化器名称 (adam, adamw, sgd, adagrad)
        
    Returns:
        optimizer: 优化器
    """
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=hyperparams.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=hyperparams.learning_rate,
            momentum=0.9
        )
    elif optimizer_name == "adagrad":
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=hyperparams.learning_rate
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer

def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """
    获取学习率调度器
    
    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        
    Returns:
        scheduler: 学习率调度器
    """
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step):
        """
        学习率衰减函数
        
        Args:
            current_step: 当前步数
            
        Returns:
            lr: 学习率缩放因子
        """
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 衰减阶段
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler

def get_dropout_layers(model, dropout_rate):
    """
    递归获取模型中的所有Dropout层
    
    Args:
        model: PyTorch模型
        dropout_rate: 新的Dropout率
        
    Returns:
        dropout_layers: Dropout层列表
    """
    dropout_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            dropout_layers.append((name, module))
    
    return dropout_layers

def update_dropout_rate(model, new_dropout_rate):
    """
    更新模型中所有Dropout层的Dropout率
    
    Args:
        model: PyTorch模型
        new_dropout_rate: 新的Dropout率
        
    Returns:
        model: 更新后的模型
    """
    dropout_layers = get_dropout_layers(model, new_dropout_rate)
    
    for name, layer in dropout_layers:
        layer.p = new_dropout_rate
    
    return model

def hyperparameter_search_space():
    """
    定义超参数搜索空间
    
    Returns:
        search_space: 超参数搜索空间
    """
    search_space = {
        "d_model": [256, 512, 1024],
        "nhead": [4, 8, 16],
        "num_encoder_layers": [3, 6, 12],
        "num_decoder_layers": [3, 6, 12],
        "dim_feedforward": [1024, 2048, 4096],
        "dropout": [0.1, 0.2, 0.3],
        "learning_rate": [5e-6, 1e-5, 5e-5],
        "batch_size": [32, 64, 128]
    }
    
    return search_space

def sample_hyperparameters(search_space):
    """
    从搜索空间中随机采样超参数组合
    
    Args:
        search_space: 超参数搜索空间
        
    Returns:
        hyperparams: 采样的超参数字典
    """
    hyperparams = {}
    
    for param, values in search_space.items():
        hyperparams[param] = np.random.choice(values)
    
    return hyperparams

def evaluate_hyperparameters(hyperparams, model, data_loader, criterion, device):
    """
    评估超参数组合的性能
    
    Args:
        hyperparams: 超参数字典
        model: 模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 设备
        
    Returns:
        loss: 平均损失
        accuracy: 准确率
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)
            
            output = model(src, tgt[:-1])
            
            loss = criterion(output.view(-1, output.size(-1)), tgt[1:].view(-1))
            total_loss += loss.item()
            
            # 计算准确率
            predicted = torch.argmax(output, dim=-1)
            correct = (predicted == tgt[1:]).sum().item()
            total_correct += correct
            total_tokens += tgt[1:].numel()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_tokens
    
    return avg_loss, accuracy

# 示例：超参数选择的使用
if __name__ == "__main__":
    # 创建超参数配置
    print("超参数配置示例：")
    config = HyperparameterConfig("base")
    print(f"  模型尺寸: {config.d_model}")
    print(f"  注意力头数: {config.nhead}")
    print(f"  编码器层数: {config.num_encoder_layers}")
    print(f"  解码器层数: {config.num_decoder_layers}")
    print(f"  前馈网络尺寸: {config.dim_feedforward}")
    print(f"  Dropout率: {config.dropout}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  批量大小: {config.batch_size}")
    print(f"  最大序列长度: {config.max_seq_length}")
    print()
    
    # 计算模型参数量
    print("模型参数量估计：")
    vocab_size = 10000
    total_params, param_details = calculate_model_size(config, vocab_size)
    print(f"  总参数量: {total_params:,}")
    print(f"  嵌入层参数: {param_details['embedding']:,}")
    print(f"  位置编码参数: {param_details['positional_embedding']:,}")
    print(f"  编码器参数: {param_details['encoder']:,}")
    print(f"  解码器参数: {param_details['decoder']:,}")
    print(f"  输出层参数: {param_details['output']:,}")
    print()
    
    # 获取优化器和调度器
    print("优化器和调度器示例：")
    
    # 创建一个简单的Transformer模型
    from torch.nn import Transformer
    model = Transformer(
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout
    )
    
    optimizer = get_optimizer(model, config, optimizer_name="adamw")
    print(f"  优化器: {type(optimizer).__name__}")
    print(f"  初始学习率: {config.learning_rate}")
    
    # 计算训练步数
    num_warmup_steps = 1000
    num_training_steps = 10000
    scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)
    print(f"  调度器: {type(scheduler).__name__}")
    print(f"  预热步数: {num_warmup_steps}")
    print(f"  总训练步数: {num_training_steps}")
    print()
    
    # 超参数搜索空间
    print("超参数搜索空间示例：")
    search_space = hyperparameter_search_space()
    print(f"  可搜索的超参数: {list(search_space.keys())}")
    
    # 采样超参数
    sampled_hparams = sample_hyperparameters(search_space)
    print(f"  采样的超参数组合: {sampled_hparams}")
    print()
    
    # 更新Dropout率示例
    print("更新Dropout率示例：")
    print(f"  初始Dropout率: {model.dropout.p}")
    
    new_dropout_rate = 0.2
    model = update_dropout_rate(model, new_dropout_rate)
    
    # 验证Dropout率是否已更新
    dropout_layers = get_dropout_layers(model, new_dropout_rate)
    print(f"  模型中的Dropout层数量: {len(dropout_layers)}")
    print(f"  第一个Dropout层的新Dropout率: {dropout_layers[0][1].p}")
    print()
    
    print("超参数选择提示：")
    print("  1. 模型尺寸(d_model)：较大的尺寸通常带来更好的性能，但计算成本更高")
    print("  2. 注意力头数(nhead)：多头注意力有助于捕获不同子空间的信息")
    print("  3. 层数：增加层数可以提高模型的表达能力，但容易过拟合")
    print("  4. Dropout率：防止过拟合，但过高会导致欠拟合")
    print("  5. 学习率：使用较小的学习率和预热机制通常效果更好")
    print("  6. 批量大小：在内存允许的情况下，较大的批量大小可以提高训练稳定性")
