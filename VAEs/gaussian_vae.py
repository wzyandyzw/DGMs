# 知识点3：高斯VAE (Gaussian VAE) 实现
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义高斯VAE模型
class GaussianVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=20):
        super(GaussianVAE, self).__init__()
        
        # 编码器：输入 -> 隐藏层 -> 均值和方差
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # 均值
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 方差的对数
        
        # 解码器：潜在变量 -> 隐藏层 -> 输出
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 使用sigmoid将输出限制在0-1之间
        )
    
    def encode(self, x):
        """编码：输入x -> 潜在变量的均值和方差"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧：从N(mu, sigma^2)中采样"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # 从标准正态分布中采样
        return mu + eps * std
    
    def decode(self, z):
        """解码：潜在变量z -> 重构的x"""
        return self.decoder(z)
    
    def forward(self, x):
        """前向传播：x -> mu, logvar -> z -> x_recon"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# 定义损失函数（ELBO损失）
def vae_loss(x_recon, x, mu, logvar):
    """计算VAE的损失函数
    x_recon: 重构的x
    x: 原始x
    mu: 潜在变量的均值
    logvar: 潜在变量的方差的对数
    """
    # 重构损失：使用二元交叉熵
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL散度损失：KL(q(z|x) || p(z))
    # KL散度的解析解：0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失
    return recon_loss + kl_loss

# 训练VAE模型
def train_vae(model, train_loader, optimizer, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        train_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            # 数据预处理
            data = data.view(data.size(0), -1).to(device)  # 展平并移动到设备
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            x_recon, mu, logvar = model(data)
            
            # 计算损失
            loss = vae_loss(x_recon, data, mu, logvar)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item()
        
        # 打印每个epoch的损失
        avg_loss = train_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# 生成样本
def generate_samples(model, num_samples=10, latent_dim=20, device='cpu'):
    """从VAE的潜在空间生成样本"""
    model.eval()
    
    with torch.no_grad():
        # 从标准正态分布中采样潜在变量
        z = torch.randn(num_samples, latent_dim).to(device)
        
        # 解码生成样本
        generated = model.decode(z)
        
    return generated.view(-1, 28, 28).cpu().numpy()

# 主函数
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 超参数
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 10
    latent_dim = 20
    
    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = GaussianVAE(input_dim=784, hidden_dim=256, latent_dim=latent_dim).to(device)
    
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("开始训练VAE模型...")
    train_vae(model, train_loader, optimizer, device, num_epochs=num_epochs)
    
    # 生成样本
    print("\n生成样本...")
    generated_samples = generate_samples(model, num_samples=10, latent_dim=latent_dim, device=device)
    
    # 可视化生成的样本
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(generated_samples[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print("\n高斯VAE模型训练和生成完成！")