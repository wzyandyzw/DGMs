# Energy-Based Models
**Author**: Zhaoqiang Liu  
**Affiliation**: School of Computer Science & Mathematics (joint appointment), University of Electronic Science and Technology of China  
**Date**: 17 October, 2025

---

## Recap: Normalizing flows (NFs)
- Consider a directed, latent-variable model over observed variables $x$ and latent variables $Z$
- In a NF, the mapping between $Z$ and $X$ , given by $f_{\theta}: \mathbb{R}^{n} \to \mathbb{R}^{n}$ , is deterministic and invertible such that $X=f_{\theta}(Z)$ and $Z=f_{\theta}^{-1}(X)$
- Using change of variables, the marginal likelihood $p_{\theta}(x)$ is given by 
\[p_{\theta}(x)=p_{z}\left(f_{\theta}^{-1}(x)\right)\left|\det\left(\frac{\partial f_{\theta}^{-1}(x)}{\partial x}\right)\right|\]
- Note that $x$ , $Z$ need to be continuous and of the same dimension

---

## Recap: Learning and inference of NFs
- **Learning** via maximum likelihood estimation over the dataset $\mathcal{D}$ , with objective 
\[log p_{\theta}(\mathcal{D})=\sum_{x \in \mathcal{D}}\left(log p_{Z}\left(f_{\theta}^{-1}(x)\right)+log \left(\left|\det\left(\frac{\partial f_{\theta}^{-1}(x)}{\partial x}\right)\right|\right)\right)\]
  - Exact likelihood evaluation via inverse transformation $x \to z$ and change of variables formula
- **Sampling** via forward transformation $z \to x$ with 
\[z \sim p_z(z), x=f_{\theta}(z)\]
- **Latent representations inference** via inverse transformation (no inference network required!) 
\[z=f_{\theta}^{-1}(x)\]
- **Training**: Choose transformations so that the resulting Jacobian matrix has special structure. For example, the determinant of a triangular matrix is the product of the diagonal entries, i.e., an $O(n)$ operation

---

## Recap: Planar flows
- **Planar flow**: Invertible transformation 
\[x=f_{\theta}(z)=z+u h\left(w^{\top} z+b\right)\]
  - Parameterized by $\theta=(w, u, b)$ and $h(\cdot)$ is a non-linearity
- Absolute value of the determinant of the Jacobian is given by 
\[\left|\det\left( \frac {\partial f_{\theta }(z)}{\partial z}\right)\right| = \left|\det\left(I_{n}+h'\left(w^{\top } z+b\right) u w^{\top}\right)\right|=\left|1+h'\left(w^{\top} z+b\right) w^{\top} u\right|\]
- Need to restrict parameters and non-linearity for the mapping to be invertible. For example, $h(\cdot)=\tanh(\cdot)$ and $h'(w^{\top} z+b) w^{\top} u>-1$

---

## Recap: Nonlinear independent component estimation (NICE)
- Partition the variables $Z$ into two disjoint subsets, say $z_{1: d}$ and $z_{(d+1): n}$ for any $1 ≤d ≤n$
- **Forward mapping** $z \to x$ :
  - $x_{1: d}=z_{1: d}$ (identity transformation)
  - $x_{(d+1): n}=z_{(d+1): n}+m_{\theta}(z_{1: d})$ ($m_{\theta}: \mathbb{R}^{d} \to \mathbb{R}^{n-d}$ is a NN parameterized by $\theta$)
- **Inverse mapping** $x \to z$ :
  - $z_{1: d}=x_{1: d}$ (identity transformation) 
  \[z_{(d+1): n}=x_{(d+1): n}-m_{\theta}\left(x_{1: d}\right)\]
- **Jacobian of forward mapping**: 
\[J=\frac{\partial x}{\partial z}=\left[\begin{array}{cc} I_{d} & 0_{d \times(n-d)} \\ \frac{\partial x_{(d+1): n}}{\partial z_{1: d}} & I_{n-d}\end{array}\right]\]
- Volume preserving transformation since determinant is 1
- There is also a final rescaling layer

---

## EBMs
Given $x_{i} \sim P_{data }$ ($i=1,2, ..., n$), we aim to minimize $d(P_{data}, P_e)$ where $P_e$ is in model family $M$.

### Energy-based models (EBMs) characteristics
- Very flexible model architectures
- Stable training
- Relatively high sample quality
- Flexible composition

---

## Parameterizing probability distributions
Probability distributions $p(x)$ are core to generative modeling, satisfying two key properties:
1. **Non-negativity**: $p(x) ≥0$
2. **Sum-to-one property**: $\sum_{x} p(x)=1$ (discrete case) or $\int p(x) d x=1$ (continuous case)

- Coming up with a non-negative function $p_{\theta}(x)$ is not hard. Given any function $f_{\theta}(x)$ , we can construct non-negative functions such as:
  - $g_{\theta}(x)=f_{\theta}(x)^{2}$
  - $g_{\theta}(x)=\exp \left(f_{\theta}(x)\right)$
  - $g_{\theta}(x)=\left|f_{\theta}(x)\right|$
  - $g_{\theta}(x)=\log \left(f_{\theta}(x)\right)$ (with domain constraints)

---

## Parameterizing probability distributions (continuation)
- Problem: $g_{\theta}(x) ≥0$ is easy to satisfy, but $g_{\theta}(x)$ might not sum to one. In general, $\sum_{x} g_{\theta}(x)=Z(\theta) ≠1$ (discrete) or $\int g_{\theta}(x) d x ≠1$ (continuous), so $g_{\theta}(x)$ is not a valid probability mass/density function.
- **Solution**: Normalize $g_{\theta}(x)$ by its integral (volume)
\[p_{\theta}(x)=\frac{1}{Z(\theta)} g_{\theta}(x)=\frac{1}{\int g_{\theta}(x) d x} g_{\theta}(x)=\frac{1}{Volume\left(g_{\theta}\right)} g_{\theta}(x)\]
  - Then $\int p_{\theta }(x)dx=\int g_{\theta }(x)dx/Z(\theta )=Z(\theta )/Z(\theta )=1$

### Examples of $g_{\theta}(x)$ with analytical volume
1. **Gaussian distribution**: $g_{(\mu, \sigma^{2})}(x)=e^{-\frac{(x-\mu)^{2}}{2 \sigma^{2}}}$, volume $\int e^{-\frac{(x-\mu)^{2}}{2 \sigma^{2}}} d x=\sqrt{2 \pi} \sigma$
2. **Exponential distribution**: $g_{\lambda}(x)=e^{-\lambda x}$, volume $\int_{0}^{\infty} e^{-\lambda x} ~d x=\frac{1}{\lambda}$
3. **Exponential family**: $g_{\theta}(x)=h(x) \exp (\theta \cdot T(x))$, volume $\exp (A(\theta))$ where $A(\theta)=\log (\int h(x) \exp (\theta \cdot T(x)) d x)$ (covers Normal, Poisson, Bernoulli, Beta, Gamma, Dirichlet, Wishart distributions)

- Function forms $g_{\theta}(x)$ need analytical integrability (restrictive but useful as complex distribution building blocks)

---

## Likelihood based learning
- For complex models, we can combine normalized basic distributions:
  1. **Autoregressive models**: Products of normalized objects $p_{\theta}(x) p_{\theta'(x)}(y)$
     - $\int_{x} \int_{y} p_{\theta}(x) p_{\theta'(x)}(y) d x d y=\int_{x} p_{\theta}(x) \int_{y} p_{\theta'(x)}(y) d y d x=\int_{x} p_{\theta}(x) d x=1$
  2. **Latent variable models**: Mixture of normalized objects $\alpha p_{\theta}(x)+(1-\alpha) p_{\theta'}(x)$
     - $\int_{x} \alpha p_{\theta}(x)+(1-\alpha) p_{\theta'}(x) d x=\alpha+(1-\alpha)=1$

- Open question: How to handle models where $g_{\theta}(x)$’s volume/normalization constant is not analytically computable?

---

## Energy-based models (formal definition)
- For an unconstrained energy function $E_{\theta}: \mathbb{R}^{n} \to \mathbb{R}$, set $g_{\theta}(x)=e^{-E_{\theta}(x)}$, then the EBM probability distribution is:
\[p_{\theta}(x)=\frac{1}{\int g_{\theta}(x) d x} g_{\theta}(x)=\frac{1}{\int \exp \left(-E_{\theta}(x)\right) d x} \exp \left(-E_{\theta}(x)\right)=\frac{1}{Z(\theta)} \exp \left(-E_{\theta}(x)\right)\]
  - The normalization constant $Z(\theta)=\int \exp \left(-E_{\theta}(x)\right) d x$ is called the **partition function**

### Why use exponential form (not $E_{\theta}(x)^2$ etc.)
1. Captures large probability variations (log-probability is natural working scale; otherwise requires highly non-smooth $E_{\theta}$)
2. Aligns with exponential families (many common distributions fit this form)
3. Derived from statistical physics assumptions (maximum entropy, second law of thermodynamics)
4. Intuitive interpretation: Configurations $x$ with low energy are more likely

---

## Energy-based models (pros and cons)
### Pros
- Extreme flexibility (can use almost any $E_{\theta}(x)$ function)

### Cons
1. Sampling from $p_{\theta}(x)$ is difficult
2. Evaluating and optimizing likelihood $p_{\theta}(x)$ is hard (learning is challenging)
3. No inherent feature learning (can add latent variables to address this)
4. **Curse of dimensionality**: Computing $Z(\theta)$ numerically scales exponentially with $x$’s dimension (though some tasks don’t require $Z(\theta)$)

---

## Applications of energy-based models
- For two samples $x, x'$, the probability ratio avoids $Z(\theta)$:
\[\frac{p_{\theta}(x)}{p_{\theta}\left(x'\right)}=\exp \left(-E_{\theta}(x)+E_{\theta}\left(x'\right)\right)\]
  - This enables relative probability comparisons for applications like:
    - Anomaly detection
    - Denoising

### Typical EBM application scenarios
- Object recognition (energy function $E(Y,X)$ over label $Y$ and input $X$)
- Sequence labeling (energy function $E(Y,X)$ over label sequence $Y$ and input $X$)
- Image restoration (energy function over corrupted and restored image)

---

## Example: Ising models
- Task: Recover true image $y \in \{0,1\}^{3×3}$ from corrupted image $x \in \{0,1\}^{3×3}$ (modeled as Markov Random Field, $Y_i$ = true pixels, $X$ = noisy pixels)
- Joint probability distribution:
\[p(y, x)=\frac{1}{Z} \exp \left(\sum_{i} \psi_{i}\left(x_{i}, y_{i}\right)+\sum_{(i, j) \in E} \psi_{i j}\left(y_{i}, y_{j}\right)\right)\]
  - $\psi_{i}(x_{i}, y_{i})$: Corrupted pixel depends on original pixel
  - $\psi_{i j}(y_{i}, y_{j})$: Neighboring true pixels tend to have the same value
- Goal: Maximize $p(y, x)$ (or $p(y|x)$) to recover the original $y$

---

## Example: Product of experts
- Suppose we have trained multiple independent models $q_{\theta_1}(x), r_{\theta_2}(x), t_{\theta_3}(x)$ (each is an "expert" scoring $x$’s likelihood)
- Ensemble via product of expert scores (analogous to logical AND, unlike mixture models which are logical OR):
\[q_{\theta_{1}}(x) r_{\theta_{2}}(x) t_{\theta_{3}}(x)\]
- Normalize to get valid distribution:
\[p_{\theta_{1}, \theta_{2}, \theta_{3}}(x)=\frac{1}{Z\left(\theta_{1}, \theta_{2}, \theta_{3}\right)} q_{\theta_{1}}(x) r_{\theta_{2}}(x) t_{\theta_{3}}(x)\]

### Visual example of expert composition
1. Young (EBM)
2. Young AND Female (EBM)
3. Young AND Female AND Smiling (EBM)
4. Young AND Female AND Smiling AND Wavy Hair (EBM)

*(Image source: Du et al., 2020)*

---

## Example: Restricted Boltzmann machines (RBMs)
RBM is an EBM with latent variables, containing two types of variables:
- $x \in \{0,1\}^{n}$: Visible variables (e.g., pixel values)
- $z \in \{0,1\}^{m}$: Latent variables

### Joint distribution
\[P_{W, b, c}(x, z)=\frac{1}{Z} \exp \left(x^{\top} W z+b^{\top} x+c^{\top} z\right)=\frac{1}{Z} \exp \left(\sum_{i=1}^{n} \sum_{j=1}^{m} w_{i j} x_{i} z_{j}+b^{\top} x+c^{\top} z\right)\]

### Restriction
No visible-visible or hidden-hidden connections (no $x_i x_j$ or $z_i z_j$ terms in the objective)

---

## Example: Deep Boltzmann machines
- Stacked RBMs are early deep generative models
- **Structure**: Bottom layer $v$ (pixel values), upper layers $h^{(1)},h^{(2)},h^{(3)}$ (higher-level features like corners, edges) with weight matrices $W^{(1)},W^{(2)},W^{(3)}$
- **Historical note**: Early supervised deep neural networks relied on such pre-training to function effectively

### Deep Boltzmann Machines: Samples
*(Training samples vs. Generated samples; Image source: Salakhutdinov and Hinton, 2009)*

---

## Sampling from energy-based models
- No direct sampling (unlike autoregressive/flow models) due to difficulty in computing individual sample likelihoods, but relative sample comparison is easy
- **Iterative approach: Markov Chain Monte Carlo (MCMC)**
  1. Initialize $x^{(0)}$ randomly
  2. Generate $x'=x^{(t)}+\epsilon$
  3. Update rule: If $E_{\theta}(x')<E_{\theta}(x^{(t)})$, set $x^{(t+1)}=x'$; else set $x^{(t+1)}=x'$ with probability $\exp(-E_{\theta}(x')+E_{\theta}(x^{(t)}))$
  4. Repeat for many iterations (theoretically converges but can be slow)

### Langevin dynamics (continuous Markov process)
- SDE formulation:
\[d x_{t}=\nabla_{x} \log p_{\theta}\left(x_{t}\right) d t+\sqrt{2} d w_{t}\]
- **Langevin MCMC** update rule:
\[x^{(t+1)}=x^{(t)}+\eta \nabla_{x} \log p_{\theta}(x^{(t)})+\sqrt{2 \eta} z^{(t)}\]
  - $x^{(0)} \sim \pi(x)$ (prior), $z^{(t)} \sim N(0, I_{n})$, $\eta \to 0$ and $T \to \infty$ ensures $x^{(T)} \sim p_{\theta}(x)$
- For EBMs, score function is tractable:
\[\nabla_{x} \log p_{\theta}(x)=-\nabla_{x} E_{\theta}(x)-\nabla_{x} \log (Z(\theta))=-\nabla_{x} E_{\theta}(x)\]

---

## Modern EBMs (sampling results)
- **Langevin sampling face samples** (Image source: Nijkamp et al. 2019)
- **ImageNet samples**

---

## Training of EBMs
Core goal: Train $E_{\theta}$ such that $p_{\theta} ≈p_{data }$

### Key insight: Score function approximation
- Langevin dynamics only requires gradients of log-probability (score function), not the energy function directly:
\[\begin{aligned} x^{(t+1)} & =x^{(t)}+\eta \nabla_{x} \log p_{\theta}\left(x^{(t)}\right)+\sqrt{2 \eta} z^{(t)} \\ & =x^{(t)}-\eta \nabla_{x} E_{\theta}\left(x^{(t)}\right)+\sqrt{2 \eta} z^{(t)} \end{aligned}\]
- We can learn $s_{\theta}: \mathbb{R}^{n} \to \mathbb{R}^{n}$ to approximate the data score function $s(x)=\nabla_{x} \log p_{data }(x)$

---

## Score matching
- Objective: Learn $s_{\theta}$ to minimize the difference from $s=\nabla_{x} \log p_{data }(x)$, e.g., via MSE:
\[\mathbb{E}_{x \sim p_{data }}\left[\left\| s_{\theta}(x)-\nabla_{x} \log p_{data }(x)\right\| _{2}^{2}\right]\]
  - This corresponds to the **Fisher divergence** between $P_{data }$ and $p_{\theta}$:
\[D_{Fisher }(p \| q)=\frac{1}{2} \mathbb{E}_{x \sim p}\left[\left\| \nabla_{x} \log p(x)-\nabla \log q(x)\right\| _{2}^{2}\right]=\frac{1}{2} \mathbb{E}_{x \sim p}\left[\left\| \nabla_{x} \log \left(\frac{p(x)}{q(x)}\right)\right\| _{2}^{2}\right]\]
- Problem: $p_{data }$ is unknown (only dataset samples are available)

---

## Kernel based explicit score matching
- Approximate $p_{data }$ with **kernel density estimation (KDE)**:
\[\hat{p}(x)=\frac{1}{N} \sum_{i=1}^{N} K_{\sigma}\left(x-x_{i}\right)\]
  - Score of KDE approximation:
\[\nabla_{x} \log \hat{p}(x)=\frac{\nabla_{x} \hat{p}(x)}{\hat{p}(x)}=\frac{\frac{1}{N} \sum_{i=1}^{N} \nabla_{x} K_{\sigma}\left(x-x_{i}\right)}{\frac{1}{N} \sum_{i=1}^{N} K_{\sigma}\left(x-x_{i}\right)}\]

### Common kernel functions
1. **Gaussian kernel**: $K_{\sigma}(x)=\frac{1}{(2 \pi \sigma^{2})^{d / 2}} \exp (-\frac{\|x\|_{2}^{2}}{2 \sigma^{2}})$, $\nabla_{x} K_{\sigma}(x)=-\frac{x}{\sigma^{2}} K_{\sigma}(x)$
2. **Exponential kernel**: $K_{\sigma}(x)=\frac{1}{Z} \exp (-\frac{\|x\|_{2}}{\sigma})$, $\nabla_{x} K_{\sigma}(x)=-\frac{x}{\sigma\|x\|_{2}} K_{\sigma}(x)$
3. Other kernels: Matérn kernel, Epanechnikov kernel, etc.

### Approximated score matching objective
\[\begin{array} {rl}&{\mathbb {E}_{x\sim p_{data}}\left[ \| s_{\theta }(x)-\nabla _{x}log p_{data}(x)\| _{2}^{2}\right] \approx \mathbb {E}_{x\sim \hat {p}}\left[ \left\| s_{\theta }(x)-\nabla _{x}log \hat {p}(x)\right\| _{2}^{2}\right] }\\ &{=\int \hat {p}(x)\| s_{\theta }(x)-\nabla _{x}log \hat {p}(x)\| _{2}^{2}dx}\\ &{=\frac {1}{N}\sum _{i=1}^{N}\int K_{\sigma }\left( x-x_{i}\right) \| s_{\theta }(x)-\nabla _{x}log \hat {p}(x)\| _{2}^{2}dx}\end{array}\]

---

## Implicit score matching
### Proposition [Hyvärinen, 2005]
\[arg min _{\theta} \mathbb{E}_{x \sim p_{data }}\left[\left\| s_{\theta}(x)-\nabla_{x} \log p_{data }(x)\right\| _{2}^{2}\right]=arg min _{\theta} \mathbb{E}_{x \sim p_{data }}\left[2 tr\left(\nabla_{x} s_{\theta}(x)\right)+\left\| s_{\theta}(x)\right\| _{2}^{2}\right]\]

### Proof
Expand the original MSE objective:
\[\begin{aligned} & \mathbb{E}_{x \sim p_{data }}\left[\left\| s_{\theta}(x)-\nabla_{x} \log p_{data }(x)\right\| _{2}^{2}\right] \\ & =\mathbb{E}_{x \sim p_{data }}\left[\left\| s_{\theta}(x)\right\| _{2}^{2}+\left\| \nabla_{x} \log p_{data }(x)\right\| _{2}^{2}-2 s_{\theta}(x)^{\top} \nabla_{x} \log p_{data }(x)\right] \end{aligned}\]
It suffices to show:
\[-\mathbb{E}_{x \sim p_{data }}\left[s_{\theta}(x)^{\top} \nabla_{x} \log p_{data }(x)\right]=\mathbb{E}_{x \sim p_{data }}\left[tr\left(\nabla_{x} s_{\theta}(x)\right)\right]\]
Via integration by parts:
\[\begin{aligned} & -\mathbb{E}_{x \sim p_{data }}\left[s_{\theta}(x)^{\top} \nabla_{x} \log p_{data }(x)\right]=-\int p_{data }(x) s_{\theta}(x)^{\top} \nabla_{x} log p_{data }(x) d x \\ & =-\int s_{\theta}(x)^{\top} \nabla_{x} p_{data }(x) d x=-\sum_{i=1}^{n} \int\left(s_{\theta}(x)\right)_{i} \frac{\partial p_{data }(x)}{\partial x_{i}} d x \\ & =\sum_{i=1}^{n} \int \frac{\partial\left(s_{\theta}(x)\right)_{i}}{\partial x_{i}} p_{data }(x) d x=\mathbb{E}_{x \sim p_{data }}\left[tr\left(\nabla_{x} s_{\theta}(x)\right)\right] \end{aligned}\]

### EBM-specific optimization objective
Since $s_{\theta}(x)=\nabla_{x} \log p_{\theta}(x)=-\nabla_{x} E_{\theta}(x)$, the objective becomes:
\[\begin{aligned} & arg min _{\theta} \mathbb{E}_{x \sim p_{data }}\left[2 tr\left(\nabla_{x} s_{\theta}(x)\right)+\left\| s_{\theta}(x)\right\| _{2}^{2}\right] \\ & =arg min _{\theta} \mathbb{E}_{x \sim p_{data }}\left[\sum_{i=1}^{n}\left(-2 \frac{\partial^{2} E_{\theta}(x)}{\partial x_{i}^{2}}+\left(\frac{\partial E_{\theta}(x)}{\partial x_{i}}\right)^{2}\right)\right] \end{aligned}\]

### Limitation
Computing full second-order partial derivatives is $O(n^{2})$, and even Hessian trace computation is expensive with modern hardware/automatic differentiation. This formulation is only tractable for simple energy functions.

---

## Sliced score matching (SSM)
- Instead of minimizing Fisher divergence of vector-valued scores, project scores onto random vector $v$ and minimize:
\[\mathcal{L}(\theta):=\mathbb{E}_{x \sim p_{data }} \mathbb{E}_{v \sim p_{v}}\left[\left(v^{\top} s_{\theta}(x)-v^{\top} \nabla_{x} \log p_{data }(x)\right)^{2}\right]\]
  - Typical $v$ distribution: $\mathbb{E}[v]=0$, $\mathbb{E}[v v^{\top}]=I_{n}$ (e.g., $v \sim N(0, I_{n})$)

### SSM optimization objective
\[\begin{aligned} & arg min _{\theta} \mathcal{L}(\theta)=arg min _{\theta} \mathbb{E}_{x \sim p_{data }} \mathbb{E}_{v \sim p_{v}}\left[2 v^{\top} \nabla_{x} s_{\theta}(x) v+\left(v^{\top} s_{\theta}(x)\right)^{2}\right] \\ & =arg min _{\theta} \mathbb{E}_{x \sim p_{data }} \mathbb{E}_{v \sim p_{v}}\left[-2 \sum_{i=1}^{n} \sum_{j=1}^{n}\left(\frac{\partial^{2} E_{\theta}(x)}{\partial x_{i} \partial x_{j}} v_{i} v_{j}\right)+\sum_{i=1}^{n}\left(\frac{\partial E_{\theta}(x)}{\partial x_{i}} v_{i}\right)^{2}\right] \end{aligned}\]

### Computational advantage
Note that $\sum_{i=1}^{n} \sum_{j=1}^{n}\left(\frac{\partial^{2} E_{\theta}(x)}{\partial x_{i} \partial x_{j}} v_{i} v_{j}\right)=\sum_{i=1}^{n} \frac{\partial}{\partial x_{i}}\left(\sum_{j=1}^{n} \frac{\partial E_{\theta}(x)}{\partial x_{j}} v_{j}\right) v_{i}$. The term $\sum_{j=1}^{n} \frac{\partial E_{\theta}(x)}{\partial x_{j}} v_{j}$ is computed once ($O(n)$), plus $O(n)$ for the outer sum, reducing computational complexity.

---

## Why still train energy models $E_{\theta}$?
Scenarios where EBMs are preferable to direct score modeling:
1. **Scientific applications with physical meaning**: Energy functions can respect symmetries/conservation laws (e.g., physics simulations)
2. **Out-of-distribution/anomaly detection**: High energy indicates unlikely data (compare normal vs. anomalous sample energies)
3. **Structured/compositional knowledge**: EBMs are naturally compositional, suitable for multi-modal data and domain knowledge integration
4. **Stable training**: Modern techniques enable stable training without multiple noise levels (unlike diffusion models)
5. **Architecture flexibility**: Any scalar-output architecture works (easier to design than constrained vector-valued score functions)

---

