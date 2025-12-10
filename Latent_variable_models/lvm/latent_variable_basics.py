# 知识点1：潜在变量模型的基本概念
"""
- 图像等数据中存在许多变异因素（如性别、发色、姿态等），这些因素在未标注时是潜在的
- 潜在变量模型使用潜在变量 z 显式建模这些变异因素
- 只有阴影变量 x 是数据中可观测的（如图像像素值）
- 潜在变量 z 对应高级特征
  - 若 z 选择得当，p(x|z) 可能比 p(x) 简单得多
  - 训练后可通过 p(z|x) 识别特征（如 p(EyeColor=Blue|x)
  - 挑战：手动指定这些条件分布非常困难
"""

import numpy as np
import matplotlib.pyplot as plt

# 模拟简单的潜在变量模型示例
def simple_latent_variable_model():
    """模拟一个简单的潜在变量模型示例"""
    # 潜在变量z：假设为二元变量，0表示男性，1表示女性
    z = np.random.choice([0, 1], size=100, p=[0.5, 0.5])
    
    # 观测变量x：假设为身高，男性平均175cm，女性平均165cm，标准差均为5cm
    x = np.zeros_like(z, dtype=float)
    x[z == 0] = np.random.normal(175, 5, size=np.sum(z == 0))
    x[z == 1] = np.random.normal(165, 5, size=np.sum(z == 1))
    
    # 可视化结果
    plt.figure(figsize=(8, 6))
    plt.hist(x[z == 0], bins=20, alpha=0.5, label='男性')
    plt.hist(x[z == 1], bins=20, alpha=0.5, label='女性')
    plt.legend()
    plt.title('简单潜在变量模型示例：身高分布')
    plt.xlabel('身高 (cm)')
    plt.ylabel('频率')
    plt.grid(True)
    plt.savefig('../figure/latent_variable_example.png')
    plt.close()
    
    return z, x

if __name__ == "__main__":
    z, x = simple_latent_variable_model()
    print("潜在变量模型示例创建完成")
    print(f"生成了 {len(z)} 个样本")
    print(f"男性样本数：{np.sum(z == 0)}, 女性样本数：{np.sum(z == 1)}")
