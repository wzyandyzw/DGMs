# Normalizing Flows 知识点分类

## 知识点1. 基本概念与背景

### 从简单先验到复杂数据分布
- 任何模型分布 \( p_	heta(\mathbf{x}) \) 所需的理想性质：
  - 易于评估，具有闭合形式的密度函数（对训练有用）
  - 易于采样（对生成有用）
- 许多简单分布满足上述性质，例如高斯分布、均匀分布
- 遗憾的是，数据分布更为复杂（多模态）
- Normalizing Flows（NFs）的核心思想：通过可逆变换将简单分布（易于采样和评估密度）映射到复杂分布

### VAE与Normalizing Flows的对比
- NFs与VAEs相似：
  - 从简单分布 \( \mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I}_k) \) 开始
  - 通过 \( p_	heta(\mathbf{x}|\mathbf{z}) = \mathcal{N}(f_	heta(\mathbf{z}), \sigma^2_{	ext{dec}} \mathbf{I}_n) \) 进行变换
  - 尽管 \( p(\mathbf{z}) \) 简单，但边缘分布 \( p_	heta(\mathbf{x}) = \int p_	heta(\mathbf{x}, \mathbf{z})d\mathbf{z} \) 非常复杂/灵活
  - 如果我们可以轻松地"反转" \( p(\mathbf{x}|\mathbf{z}) \) 并设计计算 \( p(\mathbf{z}|\mathbf{x}) \) 呢？
- 使 \( \mathbf{x} = f_	heta(\mathbf{z}) \) 成为 \( \mathbf{z} \) 的确定性可逆函数，这样对于任何 \( \mathbf{x} \) 都有唯一对应的 \( \mathbf{z} \)

### 连续随机变量回顾
- 设 \( X \) 为连续随机变量
- \( X \) 的累积分布函数（CDF）为 \( F_X(x) = \mathbb{P}(X \leq x) \)
- \( X \) 的概率密度函数（pdf）为 \( p_X(x) = F'_X(x) = \frac{dF_X(x)}{dx} \)
- 通常考虑参数化密度：
  - 高斯分布：\( X \sim \mathcal{N}(\mu, \sigma^2) \)，其 \( p_X(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}
ight) \)
  - 均匀分布：\( X \sim \mathcal{U}(a, b) \)，其 \( p_X(x) = \frac{1}{b-a} \mathbb{I}_{\{a \leq x \leq b\}} \)
- 如果 \( \mathbf{X} \) 是连续随机向量，我们可以使用其联合pdf表示：
  - 高斯分布：\( \mathbf{X} \sim \mathcal{N}(oldsymbol{\mu}, oldsymbol{\Sigma}) \)，其 \( p_{\mathbf{X}}(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d \det(oldsymbol{\Sigma})}} \exp\left(-\frac{(\mathbf{x}-oldsymbol{\mu})^	op oldsymbol{\Sigma}^{-1}(\mathbf{x}-oldsymbol{\mu})}{2}
ight) \)

## 知识点2. 变量变换公式

### 1D变量变换公式
- 考虑一个均匀随机变量 \( Z \sim \mathcal{U}[0, 2] \)，其密度为 \( p_Z \)。那么 \( p_Z(1) \) 是多少？
- 令 \( X = 4Z \)，其密度为 \( p_X \)。那么 \( p_X(4) \) 是多少？
- 显然，\( X \) 在 \( [0, 8] \) 上均匀分布，因此 \( p_X(4) = 1/8 \)
- 要得到正确结果，我们需要使用变量变换公式

- 变量变换（1D情况）：如果 \( X = f(Z) \) 且 \( f(\cdot) \) 单调且有逆函数 \( Z = f^{-1}(X) = h(X) \)，则
  \( p_X(x) = p_Z(h(x)) |h'(x)| \)
- 前例：如果 \( X = f(Z) = 4Z \) 且 \( Z \sim \mathcal{U}[0, 2] \)，\( p_X(4) \) 是多少？
  - 注意 \( h(X) = X/4 \)
  - \( p_X(4) = p_Z(1) h'(4) = \frac{1}{2} 	imes \frac{1}{4} = \frac{1}{8} \)
- 更有趣的例子：如果 \( X = f(Z) = \exp(Z) \) 且 \( Z \sim \mathcal{U}[0, 2] \)，\( p_X(x) \) 是多少？
  - \( Z = h(X) = \log(X) \)，因此 \( p_X(x) = p_Z(h(x)) \cdot \left|\frac{1}{x}
ight| = \frac{1}{2x} \)，其中 \( x \in [1, \exp(2)] \)
- 注意 \( p_X(x) \) 的"形状"与先验 \( p_Z(z) \) 不同（更复杂）

### 几何视角：行列式与体积
- 设 \( \mathbf{Z} \) 是 \( [0,1]^n \) 上的均匀随机向量
- 设 \( \mathbf{X} = \mathbf{A}\mathbf{Z} \)，其中 \( \mathbf{A} \in \mathbb{R}^{n 	imes n} \) 是可逆矩阵，逆矩阵为 \( \mathbf{W} = \mathbf{A}^{-1} \)。\( \mathbf{X} \) 如何分布？
- 从几何角度看，矩阵 \( \mathbf{A} \) 将单位超立方体 \( [0,1]^n \) 映射到平行六面体
- 超立方体和平行六面体是正方形/立方体和平行四边形/平行六面体在更高维度的推广
![figure1](./figure/figure1.png)

- 平行六面体的体积等于矩阵 \( \mathbf{A} \) 行列式的绝对值：
  $$\det(\mathbf{A}) = \det\left( egin{bmatrix} a & c \ b & d \end{bmatrix} 
ight) = ad - bc$$
- 设 \( \mathbf{X} = \mathbf{A}\mathbf{Z} \)，其中 \( \mathbf{A} \in \mathbb{R}^{n 	imes n} \) 是可逆矩阵，逆矩阵为 \( \mathbf{W} = \mathbf{A}^{-1} \)。\( \mathbf{X} \) 在面积为 \( \det(\mathbf{A}) \) 的平行六面体上均匀分布。因此，我们有：
  $$p_{\mathbf{X}}(\mathbf{x}) = p_{\mathbf{Z}}(\mathbf{W}\mathbf{x}) / |\det(\mathbf{A})| = p_{\mathbf{Z}}(\mathbf{W}\mathbf{x}) \cdot |\det(\mathbf{W})|$$
- 注意这与1D情况的公式相似

### 广义变量变换公式
- 对于通过 \( \mathbf{A} \) 指定的线性变换，体积变化由 \( \mathbf{A} \) 的行列式给出
- 对于非线性变换 \( f(\cdot) \)，线性化的体积变化由 \( f(\cdot) \) 的雅可比行列式给出
- 变量变换（一般情况）：\( \mathbf{Z} \) 和 \( \mathbf{X} \) 之间的映射由 \( f: \mathbb{R}^n 	o \mathbb{R}^n \) 给出，该映射可逆，使得 \( \mathbf{X} = f(\mathbf{Z}) \) 和 \( \mathbf{Z} = f^{-1}(\mathbf{X}) \)，这导致：
  $$p_{\mathbf{X}}(\mathbf{x}) = p_{\mathbf{Z}}\left(f^{-1}(\mathbf{x})
ight) \left| \det\left( \frac{\partial f^{-1}(\mathbf{x})}{\partial \mathbf{x}} 
ight) 
ight|$$
  - 推广了1D情况 \( p_{\mathbf{X}}(\mathbf{x}) = p_{\mathbf{Z}}\left(f^{-1}(\mathbf{x})
ight) \left| (f^{-1})'(\mathbf{x}) 
ight| \)
  - 与VAEs不同，\( \mathbf{x} \) 和 \( \mathbf{z} \) 需要是连续的且维度相同
  - 对于任何可逆矩阵 \( \mathbf{A} \)，\( \det(\mathbf{A}^{-1}) = 1/\det(\mathbf{A}) \)
  $$p_{\mathbf{X}}(\mathbf{x}) = p_{\mathbf{Z}}(\mathbf{z}) \left| \det\left( \frac{\partial f(\mathbf{z})}{\partial \mathbf{z}} 
ight) 
ight|^{-1}$$

### 二维示例
- 设 \( Z_1 \) 和 \( Z_2 \) 是具有联合密度 \( p_{Z_1, Z_2} \) 的连续随机变量
- 设 \( u : \mathbb{R}^2 	o \mathbb{R}^2 \) 是可逆变换，\( v \) 是其逆变换
- 设 \( X_1 = u_1(Z_1, Z_2) \) 和 \( X_2 = u_2(Z_1, Z_2) \)。则 \( Z_1 = v_1(X_1, X_2) \) 和 \( Z_2 = v_2(X_1, X_2) \)
  $$egin{align*} p_{X_1, X_2}(x_1, x_2) &= p_{Z_1, Z_2}\left(v_1(x_1, x_2), v_2(x_1, x_2)
ight) \cdot \left| \det\left( egin{bmatrix} \frac{\partial v_1(x_1, x_2)}{\partial x_1} & \frac{\partial v_1(x_1, x_2)}{\partial x_2} \ \frac{\partial v_2(x_1, x_2)}{\partial x_1} & \frac{\partial v_2(x_1, x_2)}{\partial x_2} \end{bmatrix} 
ight) 
ight| \ &= p_{Z_1, Z_2}(z_1, z_2) \cdot \left| \det\left( egin{bmatrix} \frac{\partial u_1(z_1, z_2)}{\partial z_1} & \frac{\partial u_1(z_1, z_2)}{\partial z_2} \ \frac{\partial u_2(z_1, z_2)}{\partial z_1} & \frac{\partial u_2(z_1, z_2)}{\partial z_2} \end{bmatrix} 
ight) 
ight|^{-1} \end{align*}$$

## 知识点3. Normalizing Flows 基础

### Normalizing Flows 定义
- 考虑观察变量 \( \mathbf{X} \) 和潜在变量 \( \mathbf{Z} \) 上的有向潜在变量模型
- 在NF中，\( \mathbf{Z} \) 和 \( \mathbf{X} \) 之间的映射由 \( f_{	heta} : \mathbb{R}^n 	o \mathbb{R}^n \) 给出，该映射是确定性的且可逆的，使得 \( \mathbf{X} = f_{	heta}(\mathbf{Z}) \) 和 \( \mathbf{Z} = f_{	heta}^{-1}(\mathbf{X}) \)
- 使用变量变换，边际似然 \( p_{	heta}(\mathbf{x}) \) 由下式给出：
  $$p_{	heta}(\mathbf{x}) = p_{\mathbf{Z}}\left(f_{	heta}^{-1}(\mathbf{x})
ight) \left| \det\left( \frac{\partial f_{	heta}^{-1}(\mathbf{x})}{\partial \mathbf{x}} 
ight) 
ight|$$
- 注意 \( \mathbf{x}, \mathbf{z} \) 需要是连续的且维度相同

### 变换流
- Normalizing：变量变换在应用可逆变换后给出标准化密度
- Flow：可逆变换可以相互组合
  $$oldsymbol{z}_M = oldsymbol{f}^{(M)} \circ \dots \circ oldsymbol{f}^{(1)}(oldsymbol{z}_0) = oldsymbol{f}^{(M)}\left(\dots \left(oldsymbol{f}^{(1)}(oldsymbol{z}_0)
ight)
ight) := oldsymbol{f}_oldsymbol{	heta}(oldsymbol{z}_0)$$

> - 从 \( oldsymbol{z}_0 \) 的简单分布开始（例如高斯分布）
> - 应用 \( M \) 个可逆变换序列，从 \( oldsymbol{z} = oldsymbol{z}_0 \) 获得 \( oldsymbol{x} = oldsymbol{z}_M \)
> - 通过变量变换（和链式法则）：
>   $$p_{oldsymbol{X}}(oldsymbol{x}) = p_{oldsymbol{Z}}(oldsymbol{z}) \left| \det \left( \frac{\partial oldsymbol{f}_oldsymbol{	heta}(oldsymbol{z})}{\partial oldsymbol{z}} 
ight) 
ight|^{-1} = p_{oldsymbol{Z}}(oldsymbol{z}) \prod_{m=1}^M \left| \det \left( \frac{\partial oldsymbol{f}_oldsymbol{	heta}^{(m)}(oldsymbol{z}_{m-1})}{\partial oldsymbol{z}_{m-1}} 
ight) 
ight|^{-1}$$
> - 此外，注意 \( oldsymbol{z} = oldsymbol{f}_oldsymbol{	heta}^{-1}(oldsymbol{x}) \) 意味着 \( oldsymbol{z}_0 = \left( oldsymbol{f}_oldsymbol{	heta}^{(1)} 
ight)^{-1} \left( \dots \left( oldsymbol{f}_oldsymbol{	heta}^{(M)} 
ight)^{-1} (oldsymbol{z}_M) 
ight) \)，
>   $$p_{oldsymbol{X}}(oldsymbol{x}) = p_{oldsymbol{Z}} \left( oldsymbol{f}_oldsymbol{	heta}^{-1}(oldsymbol{x}) 
ight) \left| \det \left( \frac{\partial oldsymbol{f}_oldsymbol{	heta}^{-1}(oldsymbol{x})}{\partial oldsymbol{x}} 
ight) 
ight| = p_{oldsymbol{Z}} \left( oldsymbol{f}_oldsymbol{	heta}^{-1}(oldsymbol{x}) 
ight) \prod_{m=1}^M \left| \det \left( \frac{\partial \left( oldsymbol{f}_oldsymbol{	heta}^{(m)} 
ight)^{-1} (oldsymbol{z}_m)}{\partial oldsymbol{z}_m} 
ight) 
ight|$$

### 学习与推理
- 通过数据集 \( \mathcal{D} \) 上的最大似然估计进行学习，目标为：
  $$\log p_oldsymbol{	heta}(\mathcal{D}) = \sum_{oldsymbol{x} \in \mathcal{D}} \left( \log p_{oldsymbol{Z}} \left( oldsymbol{f}_oldsymbol{	heta}^{-1}(oldsymbol{x}) 
ight) + \log \left( \left| \det \left( \frac{\partial oldsymbol{f}_oldsymbol{	heta}^{-1}(oldsymbol{x})}{\partial oldsymbol{x}} 
ight) 
ight| 
ight) 
ight)$$
- 通过逆变换 \( oldsymbol{x} \leftrightarrow oldsymbol{z} \) 和变量变换公式进行精确似然评估
- 通过正向变换 \( oldsymbol{z} 	o oldsymbol{x} \) 进行采样：
  $$oldsymbol{z} \sim p_{oldsymbol{Z}}(oldsymbol{z}),\quad oldsymbol{x} = oldsymbol{f}_oldsymbol{	heta}(oldsymbol{z})$$
- 通过逆变换推断潜在表示（不需要推理网络！）
  $$oldsymbol{z} = oldsymbol{f}_oldsymbol{	heta}^{-1}(oldsymbol{x})$$

### Flow 模型的设计要点
- 简单先验 \( p_{oldsymbol{Z}}(oldsymbol{z}) \)，允许高效采样和易于处理的似然评估。例如，各向同性高斯分布
- 具有易处理评估的可逆变换：
  - 似然评估需要高效评估 \( oldsymbol{x} 	o oldsymbol{z} \) 映射
  - 采样需要高效评估 \( oldsymbol{z} 	o oldsymbol{x} \) 映射
- 计算似然还需要评估 \( n 	imes n \) 雅可比矩阵的行列式（\( n \) 是数据维度）
  - 计算 \( n 	imes n \) 矩阵的行列式是 \( O(n^3) \)：在学习循环中成本过高
  - 核心思想：选择变换使得生成的雅可比矩阵具有特殊结构。例如，三角矩阵的行列式是对角线元素的乘积，即 \( O(n) \) 操作

## 知识点4. 常用 Flow 架构

### Planar Flows (Rezende & Mohamed, 2016)
- 基础分布：高斯分布
![figure2](./figure/figure2.png)
- 基础分布：均匀分布
![figure3](./figure/figure3.png)
- 10个平面变换可以将简单分布转换为更复杂的分布

- Planar Flow定义：可逆变换
  $$\mathbf{x} = f_	heta(\mathbf{z}) = \mathbf{z} + \mathbf{u}h(\mathbf{w}^	op \mathbf{z} + b)$$
  由 \( 	heta = (\mathbf{w}, \mathbf{u}, b) \) 参数化，\( h(\cdot) \) 是非线性函数

- 雅可比行列式的绝对值由下式给出：
  $$\left| \det\left( \frac{\partial f_	heta(\mathbf{z})}{\partial \mathbf{z}} 
ight) 
ight| = \left| \mathbf{I}_n + h'(\mathbf{w}^	op \mathbf{z} + b)\mathbf{u}\mathbf{w}^	op 
ight| = \left| 1 + h'(\mathbf{w}^	op \mathbf{z} + b)\mathbf{w}^	op \mathbf{u} 
ight|$$

- 需要限制参数和非线性函数以保证映射可逆。例如，\( h(\cdot) = 	anh(\cdot) \) 和 \( h'(\mathbf{w}^	op \mathbf{z} + b)\mathbf{w}^	op \mathbf{u} > -1 \)

### 三角 Jacobian 矩阵
- 我们有：
  $$oldsymbol{x} = [x_1, \dots, x_n]^	op = oldsymbol{f}(oldsymbol{z}) = [f_1(oldsymbol{z}), \dots, f_n(oldsymbol{z})]^	op$$
  $$oldsymbol{J} = \frac{\partial oldsymbol{f}}{\partial oldsymbol{z}} = egin{bmatrix} \frac{\partial f_1}{\partial z_1} & \dots & \frac{\partial f_1}{\partial z_n} \ \dots & \dots & \dots \ \frac{\partial f_n}{\partial z_1} & \dots & \frac{\partial f_n}{\partial z_n} \end{bmatrix}$$
- 假设 \( x_i = f_i(oldsymbol{z}) \) 仅依赖于 \( oldsymbol{z}_{\leq i} \)。那么：
  $$oldsymbol{J} = \frac{\partial oldsymbol{f}}{\partial oldsymbol{z}} = egin{bmatrix} \frac{\partial f_1}{\partial z_1} & 0 & \dots & 0 \ \frac{\partial f_2}{\partial z_1} & \frac{\partial f_2}{\partial z_2} & \dots & 0 \ \dots & \dots & \dots & \dots \ \frac{\partial f_n}{\partial z_1} & \frac{\partial f_n}{\partial z_2} & \dots & \frac{\partial f_n}{\partial z_n} \end{bmatrix}$$
  具有下三角结构。行列式可以在线性时间内计算
- 类似地，如果 \( x_i \) 仅依赖于 \( oldsymbol{z}_{\geq i} \)，则 Jacobian 矩阵是上三角的

### NICE 模型：加性耦合层
- 将变量 \( \mathbf{z} \) 划分为两个不相交的子集，例如 \( \mathbf{z}_{1:d} \) 和 \( \mathbf{z}_{(d+1):n} \)（对于任何 \( 1 \leq d \leq n \)）

  - 正向映射 \( \mathbf{z} 
ightarrow \mathbf{x} \)：
    - \( \mathbf{x}_{1:d} = \mathbf{z}_{1:d} \)（恒等变换）
    - \( \mathbf{x}_{(d+1):n} = \mathbf{z}_{(d+1):n} + m_	heta(\mathbf{z}_{1:d}) \)（\( m_	heta : \mathbb{R}^d 
ightarrow \mathbb{R}^{n-d} \) 是由 \( 	heta \) 参数化的神经网络）

  - 逆映射 \( \mathbf{x} 
ightarrow \mathbf{z} \)：
    - \( \mathbf{z}_{1:d} = \mathbf{x}_{1:d} \)（恒等变换）
    - \( \mathbf{z}_{(d+1):n} = \mathbf{x}_{(d+1):n} - m_	heta(\mathbf{x}_{1:d}) \)

  - 正向映射的 Jacobian：
    $$\mathbf{J} = \frac{\partial \mathbf{x}}{\partial \mathbf{z}} = egin{bmatrix} \mathbf{I}_d & \mathbf{0}_{d 	imes (n-d)} \ \frac{\partial \mathbf{x}_{(d+1):n}}{\partial \mathbf{z}_{1:d}} & \mathbf{I}_{n-d} \end{bmatrix}$$

- 由于行列式为1，是体积保持变换

### NICE 模型：缩放层
- 加性耦合层被组合在一起（每层中变量的任意划分）
- NICE 的最后一层应用缩放变换
- 正向映射 \( \mathbf{z} 
ightarrow \mathbf{x} \)：
  $$x_i = s_i z_i$$
  其中 \( s_i > 0 \) 是第 \( i \) 维的缩放因子
- 逆映射 \( \mathbf{x} 
ightarrow \mathbf{z} \)：
  $$z_i = \frac{x_i}{s_i}$$
- 正向映射的 Jacobian：
  $$\mathbf{J} = 	ext{Diag}(\mathbf{s})$$
  $$\det(\mathbf{J}) = \prod_{i=1}^n s_i$$

### NICE 生成的样本
![figure4](./figure/figure4.png)
![figure5](./figure/figure5.png)

### Real NVP：NICE 的非体积保持扩展

- 正向映射 \( \mathbf{z} 
ightarrow \mathbf{x} \)：
  - \( \mathbf{x}_{1:d} = \mathbf{z}_{1:d} \)（恒等变换）
  - \( \mathbf{x}_{(d+1):n} = \mathbf{z}_{(d+1):n} \odot \exp(\alpha_	heta(\mathbf{z}_{1:d})) + \mu_	heta(\mathbf{z}_{1:d}) \)（\( \alpha_	heta \) 和 \( \mu_	heta \) 都是从 \( \mathbb{R}^d \) 到 \( \mathbb{R}^{n-d} \) 的神经网络，由 \( 	heta \) 参数化；\( \odot \) 表示元素级乘积）
- 逆映射 \( \mathbf{x} 
ightarrow \mathbf{z} \)：
  - \( \mathbf{z}_{1:d} = \mathbf{x}_{1:d} \)（恒等变换）
  - \( \mathbf{z}_{(d+1):n} = (\mathbf{x}_{(d+1):n} - \mu_	heta(\mathbf{z}_{1:d})) \odot \exp(-\alpha_	heta(\mathbf{z}_{1:d})) \)
- 正向映射的 Jacobian：
  \[ \mathbf{J} = \frac{\partial \mathbf{x}}{\partial \mathbf{z}} = egin{bmatrix} \mathbf{I}_d & \mathbf{0}_{d 	imes (n-d)} \ \frac{\partial \mathbf{x}_{(d+1):n}}{\partial \mathbf{z}_{1:d}} & 	ext{Diag}(\exp(\alpha_	heta(\mathbf{z}_{1:d}))) \end{bmatrix} \]
  \[ \det(\mathbf{J}) = \prod_{i=d+1}^n \exp\left( (\alpha_	heta(\mathbf{z}_{1:d}))_i 
ight) = \exp\left( \sum_{i=d+1}^n (\alpha_	heta(\mathbf{z}_{1:d}))_i 
ight) \]
- 通常是非体积保持变换，因为行列式可以小于或大于1

### Real NVP 生成的样本
![figure6](./figure/figure6.png)

### 连续自回归模型作为 NFs

- 考虑高斯自回归模型：
  \[ p(\mathbf{x}) = \prod_{i=1}^n p(x_i | \mathbf{x}_{<<i}) \]
  其中 \( p(x_i | \mathbf{x}_{<<i}) = \mathcal{N}(\mu_i(\mathbf{x}_{<<i}), \exp(\alpha_i(\mathbf{x}_{<<i}))^2) \)（对于 \( i > 1 \)，\( \mu_i \) 和 \( \alpha_i \) 是神经网络，对于 \( i = 1 \)，它们是常数）
- 从该模型采样：
  - 采样 \( z_i \sim \mathcal{N}(0, 1) \)（\( i = 1, \dots, n \)）
  - 令 \( x_1 = \mu_1 + \exp(\alpha_1)z_1 \)，并计算 \( \mu_2(x_1), \alpha_2(x_1) \)
  - 令 \( x_2 = \mu_2 + \exp(\alpha_2)z_2 \)，并计算 \( \mu_3(x_1, x_2), \alpha_3(x_1, x_2) \)
  - 令 \( x_3 = \mu_3 + \exp(\alpha_3)z_3 \)，并计算 \( \mu_4(x_1, x_2, x_3), \alpha_4(x_1, x_2, x_3), \dots \)
- Flow 解释：通过可逆变换（由 \( \mu_i(\cdot) \) 和 \( \alpha_i(\cdot) \) 参数化）将标准高斯样本 \( (z_1, z_2, \dots, z_n) \) 转换为模型生成的样本 \( (x_1, x_2, \dots, x_n) \)

### Masked Autoregressive Flows (MAFs)

\( x_i = z_i \cdot \exp(\alpha_i) + \mu_i \ \forall i \in \{1..n\} \)

![figure7](./figure/figure7.png)

- 从 \( \mathbf{z} 
ightarrow \mathbf{x} \) 的正向映射：
  - 令 \( x_1 = \exp(\alpha_1)z_1 + \mu_1 \)，并计算 \( \mu_2(x_1), \alpha_2(x_1) \)
  - 令 \( x_2 = \mu_2 + \exp(\alpha_2)z_2 \)，并计算 \( \mu_3(x_1, x_2), \alpha_3(x_1, x_2) \)
  - 令 \( x_3 = \mu_3 + \exp(\alpha_3)z_3 \)，并计算 \( \mu_4(x_1, x_2, x_3), \alpha_4(x_1, x_2, x_3), \dots \)
- 采样是顺序的且缓慢的（如自回归模型）：\( O(n) \) 计算时间

\( z_i = (x_i - \mu_i) \cdot \exp(-\alpha_i) \ \forall i \in \{1..n\} \)

![figure8](./figure/figure8.png)

- 从 \( \mathbf{x} 
ightarrow \mathbf{z} \) 的逆映射：
  - 计算所有 \( \mu_i, \alpha_i \)（可以使用例如 MADE 并行完成）
  - 令 \( z_1 = (x_1 - \mu_1) \exp(-\alpha_1) \)（缩放和移位）
  - 令 \( z_2 = (x_2 - \mu_2) \exp(-\alpha_2), \dots \)
- Jacobian 是下对角的，因此行列式计算高效
- 似然评估简单且可并行化（如 MADE）
- 反转 MAFs 的正向和反向映射会产生逆自回归流（IAFs），其正向映射快速，逆映射缓慢

## 知识点5. 总结

- 通过变量变换将简单分布转换为更复杂的分布
- 变换的 Jacobian 应具有易于处理的行列式，以实现高效学习和密度估计
- 在评估正向和反向变换时存在计算权衡