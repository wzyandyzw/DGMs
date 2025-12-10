# 知识点5：向量量化VAE (VQ-VAE) 实现
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 定义向量量化(VQ)层
class VectorQuantization(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost=0.25):
        """
        向量量化层
        :param embedding_dim: 嵌入维度
        :param num_embeddings: 码本大小
        :param commitment_cost: 承诺损失的权重
        """
        super(VectorQuantization, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # 创建可学习的码本
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        # 初始化码本权重
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z_e):
        """
        前向传播
        :param z_e: 编码器输出的连续潜在变量
        :return: z_q: 量化后的潜在变量
                 vq_loss: VQ损失
        """
        # 展开z_e为(batch_size, num_features, embedding_dim)
        # 这里假设z_e的形状是(batch_size, embedding_dim)
        batch_size = z_e.shape[0]
        z_e_flattened = z_e.view(-1, self.embedding_dim)
        
        # 计算z_e与所有嵌入向量的距离
        distances = (torch.sum(z_e_flattened**2, dim=1, keepdim=True) 
                     + torch.sum(self.embeddings.weight**2, dim=1) 
                     - 2 * torch.matmul(z_e_flattened, self.embeddings.weight.t()))
        
        # 找到最近的嵌入向量索引
        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = encoding_indices.view(batch_size, -1)  # 恢复原始形状
        
        # 生成one-hot编码
        one_hot_encodings = torch.zeros(encoding_indices.shape[0], encoding_indices.shape[1], self.num_embeddings, 
                                       device=z_e.device)
        one_hot_encodings.scatter_(2, encoding_indices.unsqueeze(2), 1)
        
        # 量化：z_q = E * one_hot
        z_q = torch.matmul(one_hot_encodings, self.embeddings.weight)
        z_q = z_q.view(z_e.shape)  # 恢复与z_e相同的形状
        
        # 计算VQ损失
        # 第一个损失项：||sg(z_e) - z_q||^2
        loss_1 = torch.mean((torch.detach(z_e) - z_q)**2)
        # 第二个损失项：||z_e - sg(z_q)||^2
        loss_2 = torch.mean((z_e - torch.detach(z_q))**2)
        # 总VQ损失
        vq_loss = loss_1 + self.commitment_cost * loss_2
        
        # 使用straight-through估计器
        z_q = z_e + torch.detach(z_q - z_e)
        
        return z_q, vq_loss

# 定义VQ-VAE模型
class VQVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, embedding_dim=20, num_embeddings=512):
        super(VQVAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # VQ层
        self.vq_layer = VectorQuantization(embedding_dim, num_embeddings)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """前向传播"""
        # 编码
        z_e = self.encoder(x)
        
        # 量化
        z_q, vq_loss = self.vq_layer(z_e)
        
        # 解码
        x_recon = self.decoder(z_q)
        
        return x_recon, vq_loss

# 定义损失函数
def vqvae_loss(x_recon, x, vq_loss):
    """计算VQ-VAE的总损失"""
    # 重构损失
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # 总损失
    return recon_loss + vq_loss

# 训练VQ-VAE模型
def train_vqvae(model, train_loader, optimizer, device, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        train_loss = 0
        recon_loss_epoch = 0
        vq_loss_epoch = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            # 数据预处理
            data = data.view(data.size(0), -1).to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            x_recon, vq_loss = model(data)
            
            # 计算重构损失
            recon_loss = nn.functional.binary_cross_entropy(x_recon, data, reduction='sum')
            
            # 总损失
            loss = recon_loss + vq_loss
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累计损失
            train_loss += loss.item()
            recon_loss_epoch += recon_loss.item()
            vq_loss_epoch += vq_loss.item()
        
        # 打印每个epoch的损失
        avg_loss = train_loss / len(train_loader.dataset)
        avg_recon_loss = recon_loss_epoch / len(train_loader.dataset)
        avg_vq_loss = vq_loss_epoch / len(train_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Total Loss: {avg_loss:.4f}, '
              f'Recon Loss: {avg_recon_loss:.4f}, '
              f'VQ Loss: {avg_vq_loss:.4f}')

# 生成样本
def generate_samples(model, num_samples=10, embedding_dim=20, device='cpu'):
    """从VQ-VAE生成样本"""
    model.eval()
    
    with torch.no_grad():
        # 随机采样码本中的嵌入向量
        # 1. 随机选择嵌入向量索引
        random_indices = torch.randint(0, model.vq_layer.num_embeddings, (num_samples,))
        
        # 2. 获取对应的嵌入向量
        z_q = model.vq_layer.embeddings(random_indices).to(device)
        
        # 3. 解码生成样本
        generated = model.decoder(z_q)
        
    return generated.view(-1, 28, 28).cpu().numpy()

# 主函数
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 超参数
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 10
    embedding_dim = 20
    num_embeddings = 512
    
    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = VQVAE(input_dim=784, hidden_dim=256, 
                 embedding_dim=embedding_dim, num_embeddings=num_embeddings)
    model.to(device)
    
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("开始训练VQ-VAE模型...")
    train_vqvae(model, train_loader, optimizer, device, num_epochs=num_epochs)
    
    # 生成样本
    print("\n生成样本...")
    generated_samples = generate_samples(model, num_samples=10, 
                                        embedding_dim=embedding_dim, device=device)
    
    # 可视化生成的样本
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(generated_samples[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print("\nVQ-VAE模型训练和生成完成！")