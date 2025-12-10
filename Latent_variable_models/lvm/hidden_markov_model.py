# 知识点11：隐马尔可夫模型（HMM）
"""
- 离散隐藏状态的马尔可夫链\( \mathbf{h}_1 \to \mathbf{h}_2 \to \dots \)，包含\( K \)个可能状态
- 给定时间\( t \)的\( \mathbf{h}_t \)，观测\( x_t \)（\( \mathbb{R}^D \)中的随机向量）与所有其他观测/隐藏状态独立
- 初始状态分布\( \pi^* \in \Delta^{K-1} \)和转移矩阵\( \mathbf{T}^* \in \mathbb{R}^{K \times K} \)：
  \[ \mathbb{P}(\mathbf{h}_{t+1} = s_j | \mathbf{h}_t = s_i) = T_{ij}^* \]
- 观测矩阵\( \mathbf{O}^* \in \mathbb{R}^{D \times K} \)（第\( j \)列是\( \mathbb{E}[x_t | \mathbf{h}_t = s_j] \)）
- 实际应用：非侵入式负载监控
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

class HiddenMarkovModel:
    """隐马尔可夫模型（HMM）的实现"""
    
    def __init__(self, n_states, observation_dim):
        """初始化HMM模型
        
        参数:
        n_states: 隐藏状态的数量
        observation_dim: 观测向量的维度
        """
        self.n_states = n_states
        self.observation_dim = observation_dim
        
        # 模型参数
        self.pi = None  # 初始状态分布，shape=(n_states,)
        self.T = None   # 转移矩阵，shape=(n_states, n_states)
        self.O = None   # 观测矩阵，shape=(observation_dim, n_states)
        self.observation_cov = None  # 观测噪声协方差矩阵，shape=(observation_dim, observation_dim)
    
    def initialize_parameters(self):
        """初始化模型参数"""
        # 1. 初始状态分布
        self.pi = np.random.dirichlet(np.ones(self.n_states), size=1)[0]
        
        # 2. 转移矩阵
        # 使用Dirichlet分布初始化每个状态的转移概率
        self.T = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            self.T[i] = np.random.dirichlet(np.ones(self.n_states), size=1)[0]
        
        # 3. 观测矩阵
        # 每个状态的观测均值
        self.O = np.random.normal(0, 1, size=(self.observation_dim, self.n_states))
        
        # 4. 观测噪声协方差矩阵
        self.observation_cov = np.eye(self.observation_dim) * 0.1
        
        print(f"初始化HMM参数：")
        print(f"   隐藏状态数：{self.n_states}")
        print(f"   观测维度：{self.observation_dim}")
        print(f"   初始状态分布：{self.pi}")
        print(f"   转移矩阵：\n{self.T}")
        print(f"   观测矩阵：\n{self.O}")
        print(f"   观测噪声协方差矩阵：\n{self.observation_cov}")
    
    def generate_sequence(self, length):
        """生成HMM序列
        
        参数:
        length: 序列长度
        
        返回:
        hidden_states: 隐藏状态序列，shape=(length,)
        observations: 观测序列，shape=(length, observation_dim)
        """
        hidden_states = np.zeros(length, dtype=int)
        observations = np.zeros((length, self.observation_dim))
        
        # 生成初始状态
        hidden_states[0] = np.random.choice(self.n_states, p=self.pi)
        
        # 生成初始观测
        mean_obs = self.O[:, hidden_states[0]]
        observations[0] = np.random.multivariate_normal(mean_obs, self.observation_cov)
        
        # 生成后续状态和观测
        for t in range(1, length):
            # 从当前状态转移到下一个状态
            current_state = hidden_states[t-1]
            next_state = np.random.choice(self.n_states, p=self.T[current_state])
            hidden_states[t] = next_state
            
            # 生成观测
            mean_obs = self.O[:, next_state]
            observations[t] = np.random.multivariate_normal(mean_obs, self.observation_cov)
        
        return hidden_states, observations
    
    def forward_algorithm(self, observations):
        """前向算法：计算观测序列的概率
        
        参数:
        observations: 观测序列，shape=(length, observation_dim)
        
        返回:
        alpha: 前向概率，shape=(length, n_states)
        log_likelihood: 观测序列的对数似然
        """
        length = len(observations)
        alpha = np.zeros((length, self.n_states))
        
        # 初始化前向概率
        for i in range(self.n_states):
            mean_obs = self.O[:, i]
            obs_prob = multivariate_normal.pdf(observations[0], mean=mean_obs, cov=self.observation_cov)
            alpha[0, i] = self.pi[i] * obs_prob
        
        # 递推计算前向概率
        for t in range(1, length):
            for j in range(self.n_states):
                # 计算观测概率
                mean_obs = self.O[:, j]
                obs_prob = multivariate_normal.pdf(observations[t], mean=mean_obs, cov=self.observation_cov)
                
                # 计算前向概率
                alpha[t, j] = np.sum(alpha[t-1, :] * self.T[:, j]) * obs_prob
        
        # 计算观测序列的概率
        log_likelihood = np.log(np.sum(alpha[-1, :]))
        
        return alpha, log_likelihood
    
    def backward_algorithm(self, observations):
        """后向算法：计算后向概率
        
        参数:
        observations: 观测序列，shape=(length, observation_dim)
        
        返回:
        beta: 后向概率，shape=(length, n_states)
        """
        length = len(observations)
        beta = np.zeros((length, self.n_states))
        
        # 初始化后向概率
        beta[-1, :] = 1.0
        
        # 递推计算后向概率
        for t in range(length-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = 0.0
                for j in range(self.n_states):
                    # 计算观测概率
                    mean_obs = self.O[:, j]
                    obs_prob = multivariate_normal.pdf(observations[t+1], mean=mean_obs, cov=self.observation_cov)
                    
                    # 计算后向概率
                    beta[t, i] += self.T[i, j] * obs_prob * beta[t+1, j]
        
        return beta
    
    def viterbi_algorithm(self, observations):
        """维特比算法：找到最可能的隐藏状态序列
        
        参数:
        observations: 观测序列，shape=(length, observation_dim)
        
        返回:
        best_path: 最可能的隐藏状态序列，shape=(length,)
        """
        length = len(observations)
        delta = np.zeros((length, self.n_states))
        psi = np.zeros((length, self.n_states), dtype=int)
        
        # 初始化
        for i in range(self.n_states):
            mean_obs = self.O[:, i]
            obs_prob = multivariate_normal.pdf(observations[0], mean=mean_obs, cov=self.observation_cov)
            delta[0, i] = self.pi[i] * obs_prob
        
        # 递推计算delta和psi
        for t in range(1, length):
            for j in range(self.n_states):
                # 计算观测概率
                mean_obs = self.O[:, j]
                obs_prob = multivariate_normal.pdf(observations[t], mean=mean_obs, cov=self.observation_cov)
                
                # 计算delta和psi
                delta[t, j] = np.max(delta[t-1, :] * self.T[:, j]) * obs_prob
                psi[t, j] = np.argmax(delta[t-1, :] * self.T[:, j])
        
        # 回溯找到最佳路径
        best_path = np.zeros(length, dtype=int)
        best_path[-1] = np.argmax(delta[-1, :])
        
        for t in range(length-2, -1, -1):
            best_path[t] = psi[t+1, best_path[t+1]]
        
        return best_path
    
    def compute_posterior(self, observations):
        """计算隐藏状态的后验概率
        
        参数:
        observations: 观测序列，shape=(length, observation_dim)
        
        返回:
        gamma: 后验概率，shape=(length, n_states)
        """
        alpha, log_likelihood = self.forward_algorithm(observations)
        beta = self.backward_algorithm(observations)
        
        # 计算后验概率
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        
        return gamma
    
    def plot_sequence(self, hidden_states, observations):
        """绘制HMM序列
        
        参数:
        hidden_states: 隐藏状态序列，shape=(length,)
        observations: 观测序列，shape=(length, observation_dim)
        """
        length = len(hidden_states)
        
        plt.figure(figsize=(12, 5))
        
        # 1. 观测序列
        plt.subplot(2, 1, 1)
        for dim in range(self.observation_dim):
            plt.plot(range(length), observations[:, dim], label=f'观测维度 {dim+1}')
        plt.title('观测序列')
        plt.xlabel('时间')
        plt.ylabel('观测值')
        plt.legend()
        plt.grid(True)
        
        # 2. 隐藏状态序列
        plt.subplot(2, 1, 2)
        plt.step(range(length), hidden_states, where='post')
        plt.yticks(np.arange(self.n_states))
        plt.title('隐藏状态序列')
        plt.xlabel('时间')
        plt.ylabel('隐藏状态')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../figure/hmm_sequence.png')
        plt.close()
    
    def plot_posterior(self, observations, hidden_states):
        """绘制隐藏状态的后验概率
        
        参数:
        observations: 观测序列，shape=(length, observation_dim)
        hidden_states: 隐藏状态序列，shape=(length,)
        """
        length = len(observations)
        gamma = self.compute_posterior(observations)
        
        plt.figure(figsize=(12, 5))
        
        # 绘制后验概率
        for i in range(self.n_states):
            plt.plot(range(length), gamma[:, i], label=f'状态 {i} 的后验概率')
        
        # 绘制真实隐藏状态
        plt.step(range(length), hidden_states + 0.5, where='post', color='black', linewidth=2, label='真实隐藏状态')
        
        plt.title('隐藏状态的后验概率')
        plt.xlabel('时间')
        plt.ylabel('概率')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../figure/hmm_posterior.png')
        plt.close()

# 非侵入式负载监控示例
def non_intrusive_load_monitoring_example():
    """非侵入式负载监控示例"""
    print("\n=== 非侵入式负载监控示例 ===")
    
    # 创建HMM模型
    n_states = 3  # 3个电器（例如：灯泡、电视、冰箱）
    observation_dim = 1  # 观测为总功率
    
    hmm = HiddenMarkovModel(n_states, observation_dim)
    
    # 手动设置参数（非侵入式负载监控场景）
    hmm.pi = np.array([0.7, 0.2, 0.1])  # 初始状态分布
    hmm.T = np.array([[0.8, 0.1, 0.1],  # 转移矩阵
                     [0.2, 0.7, 0.1],
                     [0.1, 0.1, 0.8]])
    hmm.O = np.array([[50, 150, 300]])  # 观测矩阵（每个电器的功率）
    hmm.observation_cov = np.array([[25]])  # 观测噪声协方差矩阵
    
    print(f"\n负载监控HMM参数：")
    print(f"   初始状态分布：{hmm.pi}")
    print(f"   转移矩阵：\n{hmm.T}")
    print(f"   观测矩阵：\n{hmm.O}")
    
    # 生成观测序列
    length = 50
    hidden_states, observations = hmm.generate_sequence(length)
    
    print(f"\n生成的负载序列：")
    print(f"   真实电器状态：{hidden_states}")
    print(f"   总功率观测：{observations.flatten()}")
    
    # 使用维特比算法识别电器状态
    best_path = hmm.viterbi_algorithm(observations)
    print(f"\n识别的电器状态：{best_path}")
    print(f"   准确率：{np.mean(best_path == hidden_states):.2f}")
    
    # 绘制结果
    hmm.plot_sequence(hidden_states, observations)
    hmm.plot_posterior(observations, hidden_states)

# 主函数
def main():
    print("=== 隐马尔可夫模型（HMM）演示 ===")
    
    # 创建HMM模型
    n_states = 3
    observation_dim = 2
    
    hmm = HiddenMarkovModel(n_states, observation_dim)
    
    # 初始化参数
    hmm.initialize_parameters()
    
    # 生成序列
    length = 20
    hidden_states, observations = hmm.generate_sequence(length)
    
    print(f"\n生成的HMM序列：")
    print(f"   隐藏状态序列：{hidden_states}")
    print(f"   观测序列：{observations}")
    
    # 使用前向算法计算对数似然
    alpha, log_likelihood = hmm.forward_algorithm(observations)
    print(f"\n观测序列的对数似然：{log_likelihood:.4f}")
    
    # 使用维特比算法找到最可能的隐藏状态序列
    best_path = hmm.viterbi_algorithm(observations)
    print(f"\n维特比算法找到的最可能隐藏状态序列：{best_path}")
    print(f"   与真实序列的匹配度：{np.mean(best_path == hidden_states):.2f}")
    
    # 绘制结果
    hmm.plot_sequence(hidden_states, observations)
    hmm.plot_posterior(observations, hidden_states)
    
    # 非侵入式负载监控示例
    non_intrusive_load_monitoring_example()
    
    print("\n隐马尔可夫模型（HMM）演示完成！")
    print("\n要点总结:")
    print("- HMM是一种包含隐藏状态和观测序列的概率模型")
    print("- 隐藏状态构成马尔可夫链，观测是隐藏状态的噪声函数")
    print("- HMM的三个基本问题：评估（前向算法）、解码（维特比算法）、学习（Baum-Welch算法）")
    print("- 应用场景：语音识别、自然语言处理、生物信息学、非侵入式负载监控等")

if __name__ == "__main__":
    main()
