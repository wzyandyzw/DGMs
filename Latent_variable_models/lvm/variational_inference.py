# 知识点8：变分推断
"""
- 对数边际似然的下界：log p(x) ≥ E_{q(z|x)} [log p(x,z)] - E_{q(z|x)} [log q(z|x)] =: L(x, θ, φ)
- L(x, θ, φ)是ELBO（Evidence Lower Bound）或变分自由能
- 目标是最大化ELBO，等价于最小化KL(q(z|x) || p(z|x))
- 平均ELBO：L(θ, φ) = 1/M Σ_{m=1}^M L(x^{(m)}, θ, φ)
- 变分推断通过优化q(z|x)来近似后验分布p(z|x)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Bernoulli
from torch.utils.data import TensorDataset, DataLoader

# 1. 变分推断的基本原理
class VariationalInferenceDemo:
    """变分推断的基本演示"""
    
    def __init__(self):
        self.theta_true = 0.7  # 真实参数
        self.M = 1000  # 样本数量
        
    def generate_data(self):
        """生成伯努利分布的数据"""
        self.data = np.random.binomial(1, self.theta_true, size=self.M)
        print(f"生成数据：{self.M}个伯努利样本，真实参数theta={self.theta_true}")
        print(f"样本均值：{np.mean(self.data):.4f}")
        
    def compute_elbo(self, phi):
        """计算ELBO
        phi: Beta分布的参数 (alpha, beta)
        """
        alpha, beta = phi
        
        # 计算E_q[log p(x|z)] + E_q[log p(z)] - E_q[log q(z)]
        # 这里z是theta，x是观测数据
        
        # 第一项：E_q[log p(x|theta)] = sum_i log(theta^{x_i} (1-theta)^{1-x_i})
        sum_log_theta = np.sum(self.data) * (np.digamma(alpha) - np.digamma(alpha + beta))
        sum_log_1_theta = (self.M - np.sum(self.data)) * (np.digamma(beta) - np.digamma(alpha + beta))
        term1 = sum_log_theta + sum_log_1_theta
        
        # 第二项：E_q[log p(theta)] - E_q[log q(theta)] = log Gamma(a+b)/(Gamma(a)Gamma(b)) + (a-1)(digamma(a)-digamma(a+b)) + (b-1)(digamma(b)-digamma(a+b))
        # 减去 log Gamma(alpha+beta)/(Gamma(alpha)Gamma(beta)) - (alpha-1)digamma(alpha) - (beta-1)digamma(beta) + (alpha+beta-2)digamma(alpha+beta)
        term2 = (self.alpha_prior - 1) * (np.digamma(alpha) - np.digamma(alpha + beta))
        term2 += (self.beta_prior - 1) * (np.digamma(beta) - np.digamma(alpha + beta))
        term2 -= (alpha - 1) * np.digamma(alpha)
        term2 -= (beta - 1) * np.digamma(beta)
        term2 += (alpha + beta - 2) * np.digamma(alpha + beta)
        term2 += np.log(np.math.gamma(alpha + beta) / (np.math.gamma(alpha) * np.math.gamma(beta)))
        
        return term1 + term2
    
    def optimize_elbo(self):
        """优化ELBO"""
        self.alpha_prior = 1.0  # Beta先验的alpha参数
        self.beta_prior = 1.0   # Beta先验的beta参数
        
        # 初始参数
        phi = np.array([1.0, 1.0])
        
        # 优化参数
        learning_rate = 0.01
        num_iterations = 1000
        elbo_history = []
        
        for i in range(num_iterations):
            # 计算梯度
            alpha, beta = phi
            
            # 计算digamma函数
            digamma_a = np.digamma(alpha)
            digamma_b = np.digamma(beta)
            digamma_ab = np.digamma(alpha + beta)
            
            # 计算梯度
            grad_alpha = (np.sum(self.data) + self.alpha_prior - 1) * (digamma_a - digamma_ab)
            grad_alpha -= (alpha - 1) * np.polygamma(1, alpha)
            grad_alpha += (alpha + beta - 2) * np.polygamma(1, alpha + beta)
            grad_alpha += np.polygamma(1, alpha + beta) - np.polygamma(1, alpha)
            
            grad_beta = ((self.M - np.sum(self.data)) + self.beta_prior - 1) * (digamma_b - digamma_ab)
            grad_beta -= (beta - 1) * np.polygamma(1, beta)
            grad_beta += (alpha + beta - 2) * np.polygamma(1, alpha + beta)
            grad_beta += np.polygamma(1, alpha + beta) - np.polygamma(1, beta)
            
            # 更新参数
            phi += learning_rate * np.array([grad_alpha, grad_beta])
            
            # 确保参数为正
            phi = np.maximum(phi, 1e-8)
            
            # 计算ELBO
            elbo = self.compute_elbo(phi)
            elbo_history.append(elbo)
            
            if (i + 1) % 100 == 0:
                print(f"迭代 {i+1}: ELBO = {elbo:.4f}, phi = ({phi[0]:.4f}, {phi[1]:.4f})")
        
        self.phi_opt = phi
        self.elbo_history = elbo_history
        self.theta_estimate = phi[0] / (phi[0] + phi[1])
        
        print(f"\n优化完成：")
        print(f"最优参数 phi = ({phi[0]:.4f}, {phi[1]:.4f})")
        print(f"theta的后验均值估计：{self.theta_estimate:.4f}")
        print(f"真实值：{self.theta_true:.4f}")
    
    def plot_results(self):
        """绘制结果"""
        # 1. ELBO随迭代次数变化
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.elbo_history)), self.elbo_history)
        plt.xlabel('迭代次数')
        plt.ylabel('ELBO')
        plt.title('ELBO随迭代次数变化')
        plt.grid(True)
        
        # 2. 后验分布与真实值比较
        plt.subplot(1, 2, 2)
        
        # 绘制后验分布
        theta = np.linspace(0, 1, 100)
        posterior = np.power(theta, self.phi_opt[0]-1) * np.power(1-theta, self.phi_opt[1]-1)
        posterior /= np.trapz(posterior, theta)  # 归一化
        
        plt.plot(theta, posterior, label='后验分布 q(theta)')
        plt.axvline(self.theta_true, color='red', linestyle='--', label=f'真实theta={self.theta_true}')
        plt.axvline(self.theta_estimate, color='green', linestyle=':', label=f'估计均值={self.theta_estimate:.4f}')
        
        plt.xlabel('theta')
        plt.ylabel('概率密度')
        plt.title('后验分布与真实值比较')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../figure/variational_inference_demo.png')
        plt.close()

# 2. 深度变分自编码器（VAE）中的变分推断
class VAE(nn.Module):
    """深度变分自编码器"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 潜在变量的均值和方差
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """前向传播"""
        # 编码
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # 采样
        z = self.reparameterize(mu, logvar)
        
        # 解码
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar
    
    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        """计算VAE的损失函数
        recon_x: 重构的输入
        x: 原始输入
        mu: 潜在变量的均值
        logvar: 潜在变量的对数方差
        """
        # 重构损失
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL散度损失
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # ELBO = -BCE - KLD
        return BCE + KLD, BCE, KLD

# 3. 使用VAE进行演示
def vae_demo():
    """使用VAE进行变分推断演示"""
    print("\n=== 深度变分自编码器（VAE）演示 ===")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 超参数
    input_dim = 784  # MNIST图像大小
    hidden_dim = 256
    latent_dim = 2
    learning_rate = 1e-3
    num_epochs = 10
    batch_size = 128
    
    # 加载MNIST数据
    try:
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(input_dim))
        ])
        
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        
        print(f"加载MNIST数据成功，共{len(train_dataset)}个样本")
        
    except ImportError:
        # 如果没有安装torchvision，生成模拟数据
        print("torchvision未安装，生成模拟数据")
        
        # 生成2D高斯混合数据
        n_samples = 60000
        means = [[-2, -2], [2, 2], [-2, 2], [2, -2]]
        cov = [[0.5, 0], [0, 0.5]]
        
        X = []
        for i in range(n_samples):
            mean = means[np.random.choice(len(means))]
            sample = np.random.multivariate_normal(mean, cov)
            X.append(sample)
        
        X = np.array(X)
        
        # 将数据转换为[0,1]区间
        X = (X - X.min()) / (X.max() - X.min())
        X = torch.tensor(X, dtype=torch.float32)
        
        train_dataset = TensorDataset(X, X)  # 输入和输出相同
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        
        input_dim = 2  # 更新输入维度
    
    # 创建VAE模型
    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print("开始训练VAE...")
    for epoch in range(num_epochs):
        total_loss = 0
        total_bce = 0
        total_kld = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = model.loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_bce = total_bce / len(train_loader.dataset)
        avg_kld = total_kld / len(train_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, BCE: {avg_bce:.4f}, KLD: {avg_kld:.4f}")
    
    print("VAE训练完成！")

# 主函数
def main():
    print("=== 变分推断演示 ===")
    
    # 1. 基本变分推断演示
    print("\n1. 伯努利分布的变分推断")
    vi_demo = VariationalInferenceDemo()
    vi_demo.generate_data()
    vi_demo.optimize_elbo()
    vi_demo.plot_results()
    
    # 2. VAE演示
    try:
        vae_demo()
    except Exception as e:
        print(f"VAE演示失败: {e}")
    
    print("\n变分推断演示完成！")
    print("\n要点总结:")
    print("- 变分推断通过优化ELBO来近似后验分布")
    print("- ELBO = E_q[log p(x,z)] - E_q[log q(z|x)]")
    print("- 目标是最大化ELBO，等价于最小化KL(q(z|x) || p(z|x))")
    print("- VAE是变分推断在深度学习中的应用")

if __name__ == "__main__":
    main()
