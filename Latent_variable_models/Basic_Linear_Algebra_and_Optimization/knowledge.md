# 知识点1：半正定矩阵
实对称矩阵$\mathbf{A} \in \mathbb{R}^{n \times n}$是半正定（PSD）的，当且仅当对所有$\mathbf{x} \in \mathbb{R}^n$，都有$\mathbf{x}^\top \mathbf{A}\mathbf{x} \geq 0$。等价地，所有特征值均非负的实对称矩阵是半正定矩阵。若$\mathbf{A}$是半正定的，我们记$\mathbf{A} \succeq 0$；若$\mathbf{A} - \mathbf{B}$是半正定的，我们记$\mathbf{A} \succeq \mathbf{B}$。


# 知识点2：矩阵的迹、范数和基本性质
对于任意矩阵$\mathbf{A}$，我们定义：
- **迹**：$\text{tr}(\mathbf{A}) := \sum_i a_{ii}$
- **弗罗贝尼乌斯范数**：$\|\mathbf{A}\|_F^2 := \text{tr}(\mathbf{A}^\top \mathbf{A}) = \sqrt{\sum_{i,j} a_{ij}^2}$
- **谱范数**：$\|\mathbf{A}\|_{2 \to 2} = \sup_{\mathbf{x}: \|\mathbf{x}\|_2=1} \|\mathbf{A}\mathbf{x}\|_2$

注意：
- $\mathbf{x}^\top \mathbf{A}\mathbf{x} = \text{tr}(\mathbf{A}\mathbf{x}\mathbf{x}^\top)$
- 迹的交换性：$\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})$


# 知识点3：Sherman-Morrison公式及其推广
**Sherman-Morrison公式**：对于可逆矩阵$\mathbf{A} \in \mathbb{R}^{n \times n}$和向量$\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$，$\mathbf{A} + \mathbf{u}\mathbf{v}^\top$可逆的充要条件是$1 + \mathbf{v}^\top \mathbf{A}^{-1}\mathbf{u} \neq 0$。此时有：
$$
(\mathbf{A} + \mathbf{u}\mathbf{v}^\top)^{-1} = \mathbf{A}^{-1} - \frac{\mathbf{A}^{-1}\mathbf{u}\mathbf{v}^\top \mathbf{A}^{-1}}{1 + \mathbf{v}^\top \mathbf{A}^{-1}\mathbf{u}}
$$

**推广形式（Woodbury矩阵恒等式）**：对于矩阵$\mathbf{U}, \mathbf{V} \in \mathbb{R}^{n \times k}$，有：
$$
(\mathbf{A} + \mathbf{U}\mathbf{V}^\top)^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{I}_k + \mathbf{V}^\top \mathbf{A}^{-1}\mathbf{U})^{-1}\mathbf{V}^\top \mathbf{A}^{-1}
$$


# 知识点4：奇异值分解（SVD）
1. **定义**：任意矩阵 $\boldsymbol{A} \in \mathbb{R}^{n \times d}$ 可分解为：
   $$\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^\top$$
   其中 $\boldsymbol{U} = [\boldsymbol{u}_1, \dots, \boldsymbol{u}_n] \in \mathbb{R}^{n \times n}$ 和 $\boldsymbol{V} = [\boldsymbol{v}_1, \dots, \boldsymbol{v}_d] \in \mathbb{R}^{d \times d}$ 是正交矩阵，$\boldsymbol{\Sigma} \in \mathbb{R}^{n \times d}$ 是**矩形对角矩阵**（其对角线元素为非负值）。

2. **奇异值**：对角线元素 $\sigma_i = \boldsymbol{\Sigma}_{ii}$ 被称为 $\boldsymbol{A}$ 的**奇异值**。非零奇异值的数量等于 $\boldsymbol{A}$ 的秩。若 $r = \text{rank}(\boldsymbol{A})$，则满足 $\sigma_1 \geq \dots \geq \sigma_r > 0$。

3. **另一种表示形式**：
   $$\boldsymbol{A} = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^\top$$
   其**紧凑形式**为 $\boldsymbol{A} = \boldsymbol{U}_r \boldsymbol{\Sigma}_r \boldsymbol{V}_r^\top$，其中 $\boldsymbol{U}_r = [\boldsymbol{u}_1, \dots, \boldsymbol{u}_r] \in \mathbb{R}^{n \times r}$，$\boldsymbol{V}_r \in \mathbb{R}^{d \times r}$，$\boldsymbol{\Sigma}_r = \text{Diag}(\sigma_1, \dots, \sigma_r) \in \mathbb{R}^{r \times r}$。

4. **应用**：
   - 计算**伪逆**
   - 矩阵近似
   - 确定矩阵的秩、值域和零空间
   - 信号处理、数据的最小二乘拟合以及过程控制等领域


# 知识点5：凸函数
如果对于所有 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ 和所有 $\alpha \in [0, 1]$，函数 $f : \mathbb{R}^n \rightarrow \mathbb{R}$ 满足
$$f(\alpha \mathbf{x} + (1 - \alpha) \mathbf{y}) \leq \alpha f(\mathbf{x}) + (1 - \alpha) f(\mathbf{y})$$
则称该函数为凸函数。

一个函数 $f$ 是凹的当且仅当 $-f$ 是凸的。


# 知识点6：Karush-Kuhn-Tucker定理
考虑以下优化问题：
$$\min_{\mathbf{x} \in \Omega} f(\mathbf{x}) \quad \text{s.t.} \quad h_i(\mathbf{x}) \leq 0 \text{ 和 } \ell_j(\mathbf{x}) = 0 \text{ 对所有 } i, j$$
其中 $\Omega$ 是某个开集，$f, h_i, \ell_j$ 是连续可微函数。如果 $\mathbf{x}_0 \in \Omega$ 是一个局部最小值点，则 $\mathbf{x}_0$ 满足 KKT 条件：

1. **驻点条件**：$\nabla f(\mathbf{x}_0) + \sum_i u_i \nabla h_i(\mathbf{x}_0) + \sum_j v_j \nabla \ell_j(\mathbf{x}_0) = \mathbf{0}$
2. **互补松弛性**：$u_i h_i(\mathbf{x}_0) = 0$ 对所有 $i$
3. **原始可行性**：$h_i(\mathbf{x}_0) \leq 0$ 和 $\ell_j(\mathbf{x}_0) = 0$ 对所有 $i, j$
4. **对偶可行性**：$u_i \geq 0$ 对所有 $i$


# 知识点7：示例1：优化问题求解
对于一个各分量为正的向量 $\boldsymbol{\lambda} = [\lambda_1, \dots, \lambda_n]$，求解以下优化问题：
$$\begin{aligned}
&\max_{\boldsymbol{\pi}} \sum_{i=1}^n \lambda_i \log(\pi_i) \
&\text{s.t.} \quad \pi_i > 0,\, i = 1, \dots, n \
&\quad\quad\ \pi_i < 1,\, i = 1, \dots, n \
&\quad\quad\ \sum_{i=1}^n \pi_i = 1
\end{aligned}$$


# 知识点8：示例2：霍尔德（Hölder）不等式证明
对任意向量 $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^n$，当 $1 \leq p \leq \infty$ 且 $1 \leq q \leq \infty$ 满足 $\frac{1}{p} + \frac{1}{q} = 1$ 时，有：
$$\sum_{i=1}^n |x_i y_i| \leq \|\boldsymbol{x}\|_p \cdot \|\boldsymbol{y}\|_q$$