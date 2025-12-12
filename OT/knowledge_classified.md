# 最优传输（Optimal Transport）知识分类

## 知识点1. 引言与背景

### Beauty lies in the eyes of the discriminator

![figure1](./figure/figure1.png)

Source: Robbie Barrat, Obvious

- GAN generated art auctioned at Christie's
  - Expected price: $7,000-$10,000
  - True price: $432,500

### Why we need optimal transport?

- Optimal transport (OT) finds the minimal cost to transform one data distribution into another
- It tells you how to get from one to the other, not just tell you if they are different
- OT respects geometry:
  - Bad measure: These two piles of dirt are different
  - Good measure (OT): To turn pile A into pile B, I need to move 10 kg of dirt an average of 5 meters
- When do we use it?
  - Computer vision (comparing images/colors)
  - Natural language processing (comparing document meanings)
  - Genomics (aligning cell data)
  - Any problem where the structure of the data matters

## 知识点2. 最优传输问题的数学定义

### The OT problem

- Let $(\mathcal{X}, p)$ and $(\mathcal{Y}, q)$ be finite probability spaces with $|\mathcal{X}| = m$ and $|\mathcal{Y}| = n$
- Let $\Gamma(p, q) \subseteq \Delta^{m \times n}$ be the collection of distributions on the product space $\mathcal{X} \times \mathcal{Y}$ with marginals $p$ on $\mathcal{X}$ and $q$ on $\mathcal{Y}$
- Consider a cost $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}_+$ and the OT problem:
  $$d_c(p, q) = \min_{\gamma \in \Gamma(p,q)} \langle c, \gamma \rangle = \min_{\gamma \in \Gamma(p,q)} \sum_{x,y} c(x,y)\gamma(x,y)$$
- For example, if $\mathcal{X}, \mathcal{Y} \subseteq \mathbb{R}^d$ and $c(\mathbf{x},\mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2$, then the OT between $p,q$ is the Wasserstein 1-distance (w.r.t. the $\ell_2$ Euclidean distance):
  $$d_c(p, q) = \min_{\gamma \in \Gamma(p,q)} \sum_{x,y} \|\mathbf{x} - \mathbf{y}\|_2 \cdot \gamma(x,y) = W_1(p, q)$$

### The OT problem (矩阵形式)

- Consider a cost $c : \mathcal{X} \times \mathcal{Y} \to \mathbb{R}_+$ and the OT problem:
  $$d_c(p, q) = \min_{\gamma \in \Gamma(p,q)} \langle c, \gamma \rangle = \min_{\gamma \in \Gamma(p,q)} \sum_{x,y} c(x,y)\gamma(x,y)$$
- Let $\Gamma \in \mathbb{R}^{m \times n}$ be the matrix corresponding to $\gamma$, with its $(i,j)$-th entry being $\gamma(x_i, y_j)$. The constraints on the marginals can be written as:
  $$p(x_i) = \sum_{j=1}^n \gamma(x_i, y_j) = (\Gamma \mathbf{1}_n)_i$$
  $$q(y_j) = \sum_{i=1}^m \gamma(x_i, y_j) = (\Gamma^\top \mathbf{1}_m)_j$$
- Thus we can write the OT problem as:
  $$d_c(p, q) = \min_{\substack{\Gamma \mathbf{1}_n = p, \Gamma^\top \mathbf{1}_m = q}} \sum_{x,y} c(x,y)\gamma(x,y)$$

## 知识点3. 熵正则化最优传输

### Entropy-regularized OT

- The OT problem is a linear program (LP), which can be solved via LP-solvers
- But we can simplify it (with approximation error) via the entropy-regularized OT problem:
  $$d_{c,\lambda}(p, q) := \min_{\gamma \in \Gamma(p,q)} \langle c, \gamma \rangle - \lambda \mathrm{H}(\gamma)$$
  where $\lambda > 0$ is a regularization parameter and $\mathrm{H}(\gamma) = -\sum_{x,y} \gamma(x,y)\log(\gamma(x,y))$
- Analogous to KL-divergence reframing of maximum likelihood: let $k(x,y) = \exp(-c(x,y)/\lambda)$ and $Z_\lambda = \sum_{x,y} k(x,y)$ (so $k(x,y)/Z_\lambda$ is a pdf $p_{c,\lambda}$), then:
  $$\mathrm{D}_{\mathrm{KL}}(\gamma \| p_{c,\lambda}) = \sum_{x,y} \gamma(x,y)\log(\gamma(x,y)) - \sum_{x,y} \gamma(x,y)\log(p_{c,\lambda})$$
  $$= -\mathrm{H}(\gamma) - \sum_{x,y} \gamma(x,y)\log(k(x,y)/Z_\lambda)$$
  $$= -\mathrm{H}(\gamma) + \frac{1}{\lambda}\langle c, \gamma \rangle + \log(Z_\lambda)$$
- It follows that:
  $$\min_{\gamma \in \Gamma(p,q)} \mathrm{D}_{\mathrm{KL}}(\gamma \| p_{c,\lambda}) = \min_{\gamma \in \Gamma(p,q)} \langle c, \gamma \rangle - \lambda \mathrm{H}(\gamma)$$

## 知识点4. 算法与求解方法

### A conceptual algorithm

- Our goal is to find
  $$\arg \min_{\gamma \in \Gamma(p,q)} \langle c, \gamma \rangle - \lambda \mathrm{H}(\gamma) = \arg \min_{\gamma \in \Gamma(p,q)} \mathrm{D}_{\mathrm{KL}}(\gamma \| p_{c,\lambda})$$
- Two sets of constraints: $\Gamma \mathbf{1}_n = \mathbf{p}$ and $\Gamma^\top \mathbf{1}_m = \mathbf{q}$
- How about alternating minimization? Initialize $\gamma^{(0)} = p_{c,\lambda}$ and iterate:
  $$\gamma^{(\ell+1)} = \begin{cases} \arg \min_{\Gamma \mathbf{1}_n = \mathbf{p}} \mathrm{D}_{\mathrm{KL}}(\gamma \| \gamma^{(\ell)}), & \ell \text{ even} \\ \arg \min_{\Gamma^\top \mathbf{1}_m = \mathbf{q}} \mathrm{D}_{\mathrm{KL}}(\gamma \| \gamma^{(\ell)}), & \ell \text{ odd} \end{cases}$$
- These sub-problems have a closed form!

### Solving the sub-problems

- Suppose that $\ell$ is even, we aim to find $\arg \min_{\Gamma \mathbf{1}_n = \mathbf{p}} \mathrm{D}_{\mathrm{KL}}(\gamma \| \gamma^{(\ell)})$
- Introducing Lagrange multipliers (recall the KKT condition):
  $$\mathcal{L}_{\mathbf{v}}(\gamma) := \mathrm{D}_{\mathrm{KL}}(\gamma \| \gamma^{(\ell)}) + \mathbf{v}^\top (\Gamma \mathbf{1}_n = \mathbf{p})$$
- First-order optimality:
  $$\frac{\partial \mathcal{L}_{\mathbf{v}}}{\partial \gamma_{ij}} = 1 + \log(\gamma_{ij}) - \log\left( \gamma_{ij}^{(\ell)} \right) + v_i = 0$$
- This gives
  $$\gamma_{ij} = \exp(-(1 + v_i)) \cdot \gamma_{ij}^{(\ell)}$$
- Since $\sum_j \gamma_{ij} = p_i$, we obtain
  $$\exp(-(1 + v_i)) \cdot \sum_j \gamma_{ij}^{(\ell)} = p_i, \quad \exp(-(1 + v_i)) = \frac{p_i}{\sum_j \hat{\gamma}_{ij}^{(\ell)}}$$
- Therefore,
  $$\Gamma = \mathrm{Diag}\left( \frac{\mathbf{p}}{\Gamma^{(\ell)} \mathbf{1}_n} \right) \Gamma^{(\ell)}$$
- Note that $\Gamma^{(\ell)} \mathbf{1}_n$ is not guaranteed to equal $\mathbf{p}$

### Solving the sub-problems (奇数迭代)

- Similarly, suppose that $\ell$ is odd, we aim to find $\arg \min_{\Gamma^\top \mathbf{1}_m = \mathbf{q}} \mathrm{D}_{\mathrm{KL}}(\gamma \| \gamma^{(\ell)})$
- The closed-form solution is
  $$\Gamma = \mathrm{Diag}\left( \frac{\mathbf{q}}{(\Gamma^{(\ell)})^\top \mathbf{1}_m} \right) \Gamma^{(\ell)}$$
- In each iteration, $\Gamma \in \mathbb{R}^{m \times n}$
- But there are only $m + n$ constraints
- We can optimize more efficiently in the dual space

### The dual perspective

- Let $\gamma_{c,\lambda} = \arg \min_{\gamma \in \Gamma(p,q)} \langle c, \gamma \rangle - \lambda \mathrm{H}(\gamma) = \arg \min_{\gamma \in \Gamma(p,q)} \mathrm{D}_{\mathrm{KL}}(\gamma \| p_{c,\lambda})$
- Proposition: Let $K \in \mathbb{R}^{m \times n}$ with $k_{ij} = \exp(-c(x_i, y_j)/\lambda)$. Then there exist $\mathbf{u} \in \mathbb{R}^m$ and $\mathbf{v} \in \mathbb{R}^n$ such that
  $$\Gamma_{c,\lambda} = \mathrm{Diag}(\mathbf{u}) K \mathrm{Diag}(\mathbf{v})$$
- Proof. Introducing dual variables $\mathbf{s} \in \mathbb{R}^n$ and $\mathbf{t} \in \mathbb{R}^m$ and consider the Lagrange multiplier
  $$\mathcal{L}_{\mathbf{s},\mathbf{t}}(\gamma) := \langle c, \gamma \rangle - \lambda \mathrm{H}(\gamma) - \mathbf{s}^\top (\Gamma \mathbf{1}_n - \mathbf{p}) - \mathbf{t}^\top (\Gamma^\top \mathbf{1}_m - \mathbf{q})$$
  First-order optimality occurs for $\gamma_{c,\lambda}$ satisfying
  $$\frac{\partial \mathcal{L}_{\mathbf{s},\mathbf{t}}}{\partial \gamma_{ij}} = 0$$
  which gives
  $$c(x_i, y_j) + \lambda(1 + \log((\Gamma_{c,\lambda})_{ij})) - s_i - t_j = 0$$
  Then,
  $$(\Gamma_{c,\lambda})_{ij} = \exp\left( \frac{-c(x_i, y_j) + s_i + t_j}{\lambda} - 1 \right) = \exp(s_i/\lambda - 1/2) k_{ij} \exp(t_j/\lambda - 1/2)$$
  This completes the proof

## 知识点5. 矩阵缩放与Sinkhorn-Knopp算法

### Matrix scaling

- Let $\gamma_{c,\lambda} = \arg \min_{\gamma \in \Gamma(p,q)} \langle c, \gamma \rangle - \lambda \mathrm{H}(\gamma) = \arg \min_{\gamma \in \Gamma(p,q)} \mathrm{D}_{\mathrm{KL}}(\gamma \| p_{c,\lambda})$
- The optimal solution satisfies
  $$\Gamma_{c,\lambda} = \mathrm{Diag}(\mathbf{u}) K \mathrm{Diag}(\mathbf{v})$$
- The corresponding $\mathbf{u}$ and $\mathbf{v}$ satisfy the constraints:
  $$\mathbf{p} = \Gamma_{c,\lambda} \mathbf{1}_n = \mathrm{Diag}(\mathbf{u}) K \mathbf{v}$$
  $$\mathbf{q} = \Gamma_{c,\lambda}^\top \mathbf{1}_m = \mathrm{Diag}(\mathbf{v}) K^\top \mathbf{u}$$
- This is an instance of the general matrix scaling problem, which finds a scaling of a matrix such that its columns and rows sum to two given vectors respectively

### Sinkhorn-Knopp algorithm

- Matrix scaling problem: Find $\mathbf{u}, \mathbf{v}$ such that
  $$\mathbf{p} = \Gamma_{c,\lambda} \mathbf{1}_n = \mathrm{Diag}(\mathbf{u}) K \mathbf{v}$$
  $$\mathbf{q} = \Gamma_{c,\lambda}^\top \mathbf{1}_m = \mathrm{Diag}(\mathbf{v}) K^\top \mathbf{u}$$
- The Sinkhorn-Knopp algorithm:
  - Initialize $\mathbf{u}^{(0)} = \mathbf{1}_m$ and $\mathbf{v}^{(0)} = \mathbf{1}_n$
  - Alternating updates: $\mathbf{u}^{(\ell+1)} = \frac{\mathbf{p}}{K \mathbf{v}^{(\ell)}}$, and $\mathbf{v}^{(\ell+1)} = \frac{\mathbf{q}}{K^\top \mathbf{u}^{(\ell+1)}}$

### From dual to primal iterates

- Focus on the even case. The primal update is
  $$\Gamma^{(2\ell+1)} = \mathrm{Diag}\left( \frac{\mathbf{p}}{\Gamma^{(2\ell)} \mathbf{1}_n} \right) \Gamma^{(2\ell)}$$
- The corresponding dual update is
  $$\begin{aligned}
  \Gamma^{(2\ell+1)} &= \mathrm{Diag}(\mathbf{u}^{(2\ell+1)}) K \mathrm{Diag}(\mathbf{v}^{(2\ell)}) \\ 
  &= \mathrm{Diag}\left( \frac{\mathbf{p}}{K \mathbf{v}^{(2\ell)}} \right) K \mathrm{Diag}(\mathbf{v}^{(2\ell)}) \\ 
  &= \mathrm{Diag}\left( \frac{\mathbf{p}}{K \mathbf{v}^{(2\ell)}} \right) \mathrm{Diag}(1/\mathbf{u}^{(2\ell)}) \Gamma^{(2\ell)} \\ 
  &= \mathrm{Diag}\left( \frac{\mathbf{p}}{\mathbf{u}^{(2\ell)} K \mathbf{v}^{(2\ell)}} \right) \Gamma^{(2\ell)} \\ 
  &= \mathrm{Diag}\left( \frac{\mathbf{p}}{\Gamma^{(2\ell)} \mathbf{1}_n} \right) \Gamma^{(2\ell)} 
  \end{aligned}$$

## 知识点

6. 生成Sinkhorn建模

### Generative Sinkhorn modeling

- Goal: Approximately minimize $W_1(p_{\mathrm{data}}, p_\theta)$
- Draw a batch of samples $\mathbf{x}_1, \dots, \mathbf{x}_B \sim p_{\mathrm{data}}$ and $\mathbf{y}_1, \dots, \mathbf{y}_B \sim p_\theta$
- Where (empirical distributions)
  $$\hat{p}_{\mathrm{data}}(\mathbf{x}) = \frac{1}{B} \sum_{i=1}^B \mathbb{I}_{\{x_i = x\}}, \quad \hat{p}_\theta(\mathbf{y}) = \frac{1}{B} \sum_{i=1}^B \mathbb{I}_{\{y_i = y\}}$$
- Estimate $W_1(\hat{p}_{\mathrm{data}}, \hat{p}_\theta)$ using the Sinkhorn-Knopp algorithm
- Compute gradient updates $\theta \leftarrow \theta - \eta \nabla_\theta W_1(\hat{p}_{\mathrm{data}}, \hat{p}_\theta)$