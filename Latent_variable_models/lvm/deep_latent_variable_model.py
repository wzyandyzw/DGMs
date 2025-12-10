# 知识点2：深度潜在变量模型
"""
- 使用神经网络建模条件分布
  - z ~ N(0, I)
  - p(x|z) = N(x; μ_θ(z), Σ_θ(z))，其中μ_θ和Σ_θ是神经网络
- 希望训练后z对应有意义的潜在变异因素
- 特征可通过p(z|x)计算
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

# 深度潜在变量模型类
class DeepLatentVariableModel(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        super(DeepLatentVariableModel, self).__init__()
        
        # 编码器：x -> z的均值和方差
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # 输出均值和对数方差
        )
        
        # 解码器：z -> x的均值和方差
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * input_dim)  # 输出均值和对数方差
        )
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """前向传播"""
        # 编码
        encoder_output = self.encoder(x)
        z_mu, z_logvar = torch.chunk(encoder_output, 2, dim=1)
        
        # 重参数化
        z = self.reparameterize(z_mu, z_logvar)
        
        # 解码
        decoder_output = self.decoder(z)
        x_mu, x_logvar = torch.chunk(decoder_output, 2, dim=1)
        
        return z_mu, z_logvar, z, x_mu, x_logvar

# 损失函数（VAE损失）
def vae_loss(x, x_mu, x_logvar, z_mu, z_logvar):
    """计算VAE的损失函数"""
    # 重构损失
    recon_loss = nn.functional.mse_loss(x_mu, x, reduction='sum')
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    
    return recon_loss + kl_loss

# 训练函数
def train_model(model, dataloader, epochs=100, lr=1e-3):
    """训练深度潜在变量模型"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].float()
            
            optimizer.zero_grad()
            z_mu, z_logvar, z, x_mu, x_logvar = model(x)
            loss = vae_loss(x, x_mu, x_logvar, z_mu, z_logvar)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# 主函数
if __name__ == "__main__":
    # 生成模拟数据
    X, _ = make_swiss_roll(n_samples=1000, noise=0.1)
    X = X[:, [0, 2]]  # 取前两个维度
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # 标准化
    
    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 初始化模型
    input_dim = X.shape[1]
    latent_dim = 2
    model = DeepLatentVariableModel(input_dim, latent_dim)
    
    # 训练模型
    print("训练深度潜在变量模型...")
    train_model(model, dataloader, epochs=100)
    
    # 可视化结果
    model.eval()
    with torch.no_grad():
        z_mu, _, _, x_mu, _ = model(torch.tensor(X, dtype=torch.float32))
    
    # 绘制原始数据和重构数据
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)
    plt.title('原始数据')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(x_mu[:, 0], x_mu[:, 1], c='red', alpha=0.5)
    plt.title('重构数据')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../figure/deep_latent_variable_example.png')
    plt.close()
    
    print("深度潜在变量模型示例创建完成")
