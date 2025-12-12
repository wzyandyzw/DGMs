# Score Based Diffusion Models
Zhaoqiang Liu
School of Computer Science & Mathematics (joint appointment)  
University of Electronic Science and Technology of China  
21 October, 2025

## Recap: Energy-based models (EBMs)
- For an unconstrained energy function \(E_{\theta}: \mathbb{R}^{n} \to \mathbb{R}\) , letting \(g_{\theta}(x)=e^{-E_{\theta}(x)}\) and 
\[p_{\theta}(x)=\frac{1}{\int g_{\theta}(x) d x} g_{\theta}(x)=\frac{1}{\int exp \left(-E_{\theta}(x)\right) d x} exp \left(-E_{\theta}(x)\right)=\frac{1}{Z(\theta)} exp \left(-E_{\theta}(x)\right)\]

### Goal
Training a good energy model \(E_{\theta}\) , and then using it for generation/sampling
- Sampling from Langevin Markov-Chain Monte Carlo (MCMC): \(x^{(0)} \sim \pi(x)\) (\(\pi(x)\) is a prior distribution)  
  Repeat \(x^{(t+1)}=x^{(t)}+\eta \nabla_{x} log p_{\theta}(x^{(t)})+\sqrt{2 \eta} z^{(t)}\) for \(t=0,1, ..., T-1\) , where \(z^{(t)} \sim N(0, I_{n})\) (suppose we can compute \(\nabla_{x} log p_{\theta}(x))\)
- If \(\eta \to 0\) and \(T \to \infty\) , we have \(x^{(T)} \sim p_{\theta}(x)\)
- Note that for energy-based models, the score function is tractable 
\[log p_{\theta}(x)=-\nabla_{x} E_{\theta}(x)-\nabla_{x} log (Z(\theta))=-\nabla_{x} E_{\theta}(x)\]

## Recap: Training of EBMs
- Motivated by the Langevin MCMC procedure, it is natural to learn \(s_{\theta}: \mathbb{R}^{n} \to \mathbb{R}^{n}\) that approximate gradients \(s(x)=\nabla_{x} log p_{data }(x)\) (the score function)
- **Explicit score matching** (infeasible as \(p_{data }\) is unknown!): 
\[\mathbb{E}_{x \sim p_{data }}\left[\left\| s_{\theta}(x)-\nabla_{x} log p_{data }(x)\right\| _{2}^{2}\right]\]

### Kernel based explicit score matching
Suppose we have training samples \(\{x_{1}, ..., x_{N}\}\) , We can use kernel density estimation (KDE) to approximate \(P_{data }\) 
\[\hat{p}(x)=\frac{1}{N} \sum_{i=1}^{N} K_{\sigma}\left(x-x_{i}\right), with \nabla_{x} log \hat{p}(x)=\frac{\nabla_{x} \hat{p}(x)}{\hat{p}(x)}=\frac{\frac{1}{N} \sum_{i=1}^{N} \nabla_{x} K_{\sigma}\left(x-x_{i}\right)}{\frac{1}{N} \sum_{i=1}^{N} K_{\sigma}\left(x-x_{i}\right)}\]

Then, we have 
\[\begin{array} {rl}&{\mathbb {E}_{x\sim p_{data }}\left[ \| s_{\theta }(x)-\nabla _{x}log p_{data }(x)\| _{2}^{2}\right] \approx \mathbb {E}_{x\sim \hat {p}}\left[ \left\| s_{\theta }(x)-\nabla _{x}log \hat {p}(x)\right\| _{2}^{2}\right] }\\ &{=\int \hat {p}(x)\left\| s_{\theta }(x)-\nabla _{x}log \hat {p}(x)\right\| _{2}^{2}dx}\\ &{=\frac {1}{N}\sum \sum _{i=1}^{N}\int K_{\sigma }\left(x-x_{i}\right) \| s_{\theta }(x)-\nabla _{x}log \hat {p}(x)\| _{2}^{2}dx}\end{array}\]

### Implicit score matching [Hyvärinen, 2005]
\[arg min _{\theta} \mathbb{E}_{x \sim p_{data }}\left[\left\| s_{\theta}(x)-\nabla_{x} log p_{data }(x)\right\| _{2}^{2}\right]=arg min _{\theta} \mathbb{E}_{x \sim p_{data }}\left[2 tr\left(\nabla_{x} s_{\theta}(x)\right)+\left\| s_{\theta}(x)\right\| _{2}^{2}\right]\]

Since \(s_{\theta}(x)=\nabla_{x} log p_{\theta}(x)=-\nabla_{x} E_{\theta}(x)\) , we obtain the following optimization 
\[\begin{aligned} & arg min _{\theta} \mathbb{E}_{x \sim p_{data }}\left[2 tr\left(\nabla_{x} s_{\theta}(x)\right)+\left\| s_{\theta}(x)\right\| _{2}^{2}\right] \\ & =arg min _{\theta} \mathbb{E}_{x \sim p_{data }}\left[\sum_{i=1}^{n}\left(-2 \frac{\partial^{2} E_{\theta}(x)}{\partial x_{i}^{2}}+\left(\frac{\partial E_{\theta}(x)}{\partial x_{i}}\right)^{2}\right)\right] \end{aligned}\]

> Note: Since computing second-order partial derivatives is generally expensive, implicit score matching has only been applied to relatively simple energy functions where computation of second-order partial derivatives is tractable

### Sliced score matching (SSM)
Instead of minimizing the Fisher divergence between two vector-valued scores, SSM randomly samples a projection vector v and minimizes 
\[\mathcal{L}(\theta):=\mathbb{E}_{x \sim p_{data }} \mathbb{E}_{v \sim p_{v}}\left[\left(v^{\top} s_{\theta}(x)-v^{\top} \nabla_{x} log p_{data }(x)\right)^{2}\right]\]

- The typical conditions are \(\mathbb{E}[v]=0\) and \(\mathbb{E}[v v^{\top}]=I_{n}\) . A simple choice is \(v \sim N(0, I_{n})\) 
\[\begin{aligned} & arg min _{\theta} \mathcal{L}(\theta)=arg min _{\theta} \mathbb{E}_{x \sim p_{data }} \mathbb{E}_{v \sim p_{v}}\left[2 v^{\top} \nabla_{x} s_{\theta}(x) v+\left(v^{\top} s_{\theta}(x)\right)^{2}\right] \\ & =arg min _{\theta} \mathbb{E}_{x \sim p_{data }} \mathbb{E}_{v \sim p_{v}}\left[-2 \sum_{i=1}^{n} \sum_{j=1}^{n}\left(\frac{\partial^{2} E_{\theta}(x)}{\partial x_{i} \partial x_{j}} v_{i} v_{j}\right)+\sum_{i=1}^{n}\left(\frac{\partial E_{\theta}(x)}{\partial x_{i}} v_{i}\right)^{2}\right] \end{aligned}\]

Note that 
\[\sum_{i=1}^{n} \sum_{j=1}^{n}\left(\frac{\partial^{2} E_{\theta}(x)}{\partial x_{i} \partial x_{j}} v_{i} v_{j}\right)=\sum_{i=1}^{n} \frac{\partial}{\partial x_{i}}\left(\sum_{j=1}^{n} \frac{\partial E_{\theta}(x)}{\partial x_{j}} v_{j}\right) v_{i}\]

The term \(\sum_{j=1}^{n} \frac{\partial E_{\theta}(x)}{\partial x_{j}} v_{j}\) is the same for different i , and we only need to compute it once with \(O(n)\) computation, plus another \(O(n)\) computation for the outer sum

## Why still train energy models \(E_{\theta}\) ?
When to consider energy models (instead of direct score modeling):
1. **Scientific applications with physical meaning**: E.g., in physics, the energy function might be designed to respect certain symmetries or conservation laws
2. **Out-of-distribution detection or anomaly detection**: High energy = unlikely data; compare energy of normal vs. anomalous samples
3. **Structured, compositional knowledge**: Energy functions are naturally compositional, useful for multi-modal data and incorporating domain knowledge
4. **Stable Training**: EBMs can be trained stably with modern techniques, without the need for multiple noise levels (as in diffusion models)
5. **Flexibility in architecture**: We are free to choose any architecture that outputs a scalar, which can be easier to design than a vector-valued score function that must satisfy certain constraints

## How to represent probability distributions?
Given \(x_{i} \sim P_{data }\) (\(i=1,2, ..., n\)) and model family with distance \(d(P_{data}, P_{\theta})\)

### Common representations
1. **Probability density function (p.d.f.) or probability mass function (p.m.f.)**: \(p(x)\)
2. **Various deep generative models**:
   - VAEs: \(p_{\theta}(x)=\int p_{\theta}(x | z) p_{z}(z) dz\)
   - ARMs: \(p_{\theta}(x)=\prod_{i=1}^{n} p_{\theta}\left(x_{i} | x_{<<i}\right)\)
   - GANs: \(p_{\theta}(x)=\left(G_{\theta} \# p_{z}\right)(x)\)
   - NFs: \(p_{\theta}(x)=p_{z}\left(f_{\theta}^{-1}(x)\right)\left|det\left(\frac{\partial f_{\theta}^{-1}(x)}{\partial x}\right)\right|\)
   - EBMs: \(p_{\theta}(x)=\frac{e^{-E_{\theta}(x)}}{Z_{\theta}}\)

When the pdf is differentiable, we can compute the gradient of a probability density: \(\nabla_{x} log p_{data }(x)\) (score function)  
However, \(P_{data }\) is unknown, hence \(\nabla_{x} log p_{data }(x)\) is also unknown

## Denoising score matching (Vincent, 2011)
- Consider the perturbed distribution for \(\tilde{x}=x+\sigma \epsilon\) , with \(\epsilon \sim N(0, I_{n})\) : 
\[q_{0 \sigma}(\tilde{x} | x)=\mathcal{N}\left(x, \sigma^{2} I_{n}\right), q_{\sigma}(\tilde{x})=\int q_{0 \sigma}(\tilde{x} | x) p_{data }(x) d x\]
- If the noise level σ is small, we have a good approximation with \(q_{\sigma}(\tilde{x}) \approx p_{data }(\tilde{x})\)
- Denoising score matching aims to match the score of a noise-perturbed distribution 
\[min _{\theta} \mathbb{E}_{\tilde{x} \sim q_{\sigma}}\left[\left\| s_{\theta}(\tilde{x})-\nabla_{\tilde{x}} log q_{\sigma}(\tilde{x})\right\| _{2}^{2}\right]\]
- Directly computing \(\nabla_{\tilde{x}} log q_{\sigma}(\tilde{x})\) is difficult, but note that 
\[\nabla_{\tilde{x}} log q_{0 \sigma}(\tilde{x} | x)=\frac{x-\tilde{x}}{\sigma^{2}}\]
- Let us consider replacing \(\nabla_{\tilde{x}} log q_{\sigma}(\tilde{x})\) with \(\nabla_{\tilde{x}} log q_{0 \sigma}(\tilde{x} | x)\)

### Proposition
We have 
\[arg min _{\theta} \mathbb{E}_{\tilde{x} \sim q_{\sigma}}\left[\left\| s_{\theta}(\tilde{x})-\nabla_{\tilde{x}} log q_{\sigma}(\tilde{x})\right\| _{2}^{2}\right]=arg min _{\theta} \mathbb{E}_{x \sim p_{data }} \mathbb{E}_{\tilde{x} | x \sim q_{0 \sigma}}\left[\left\| s_{\theta}(\tilde{x})-\frac{x-\tilde{x}}{\sigma^{2}}\right\| _{2}^{2}\right]\]

#### Proof
Similarly to implicit score matching, we only focus on the term 
\[\begin{aligned} & \mathbb{E}_{\tilde{x} \sim q_{\sigma}}\left[s_{\theta}(\tilde{x})^{\top} \nabla_{\tilde{x}} log q_{\sigma}(\tilde{x})\right]=\int s_{\theta}(\tilde{x})^{\top} \nabla_{\tilde{x}} q_{\sigma}(\tilde{x}) d \tilde{x} \\ & =\int s_{\theta}(\tilde{x})^{\top} \nabla_{\tilde{x}}\left(\int q_{0 \sigma}(\tilde{x} | x) p_{data }(x) d x\right) d \tilde{x} \\ & =\int s_{\theta}(\tilde{x})^{\top}\left(\int \nabla_{\tilde{x}} q_{0 \sigma}(\tilde{x} | x) p_{data }(x) d x\right) d \tilde{x} \\ & =\int p_{data }(x)\left(\int s_{\theta}(\tilde{x})^{\top} \nabla_{\tilde{x}} q_{0 \sigma}(\tilde{x} | x) d \tilde{x}\right) d x \\ & =\int p_{data }(x)\left(\int q_{0 \sigma}(\tilde{x} | x) s_{\theta}(\tilde{x})^{\top} \nabla_{\tilde{x}} log q_{0 \sigma}(\tilde{x} | x) d \tilde{x}\right) d x \\ & =\mathbb{E}_{x \sim p_{data }} \mathbb{E}_{\tilde{x} | x \sim q_{0 \sigma}}\left[s_{\theta}(\tilde{x})^{\top} \nabla_{\tilde{x}} log q_{0 \sigma}(\tilde{x} | x)\right] \end{aligned}\]

Recall that \(\nabla_{\tilde{x}} log q_{0 \sigma}(\tilde{x} | x)=\frac{x-\tilde{x}}{\sigma^{2}}\) , we obtain the desired result

## Tweedie’s formula
Tweedie’s formula: Optimal denoising strategy is to follow the gradient (score) with 
\[\mathbb{E}[X | \tilde{x}]=\tilde{x}+\sigma^{2} \nabla_{\tilde{x}} log q_{\sigma}(\tilde{x})\]

Proof: We have
\[\begin{aligned} \frac{\mathbb{E}[X | \tilde{x}]-\tilde{x}}{\sigma^{2}} & =\int p(x | \tilde{x}) \frac{x-\tilde{x}}{\sigma^{2}} d x \\ & =\int p(x | \tilde{x}) \nabla_{\tilde{x}} log q_{0 \sigma}(\tilde{x} | x) d x \\ & =\int \frac{q_{0 \sigma}(\tilde{x} | x) d p_{data }(x)}{q_{\sigma}(\tilde{x})} \nabla_{\tilde{x}} log q_{0 \sigma}(\tilde{x} | x) d x \\ & =\frac{1}{q_{\sigma}(\tilde{x})} \int p_{data }(x) \nabla_{\tilde{x}} q_{0 \sigma}(\tilde{x} | x) d x \\ & =\frac{1}{q_{\sigma}(\tilde{x})} \nabla_{\tilde{x}}\left(\int p_{data }(x) q_{0 \sigma}(\tilde{x} | x) d x\right) \\ & =\frac{1}{q_{\sigma}(\tilde{x})} \nabla_{\tilde{x}} q_{\sigma}(\tilde{x}) \\ & =\nabla_{\tilde{x}} log q_{\sigma}(\tilde{x}) \end{aligned}\]
and thus \(\mathbb{E}[X | \tilde{x}]=\int p(x | \tilde{x}) x d x\)

### Tweedie’s formula and denoising score matching
Score estimation for \(\nabla_{\tilde{x}} log q_{\sigma}(\tilde{x})\) is equivalent to denoising 
\[\mathbb{E}_{x \sim p_{data }} \mathbb{E}_{\tilde{x} | x \sim q_{0 \sigma}}\left[\left\| s_{\theta}(\tilde{x})-\frac{x-\tilde{x}}{\sigma^{2}}\right\| _{2}^{2}\right]\]

Therefore, denoising score matching is suitable for optimal denoising with 
\[\begin{aligned} \mathbb{E}[X | \tilde{x}] & =\tilde{x}+\sigma^{2} \nabla_{\tilde{x}} log q_{\sigma}(\tilde{x}) \\ & \approx \tilde{x}+\sigma^{2} s_{\theta}(\tilde{x}) \end{aligned}\]

In other words, denoising score matching reduces score estimation to a denoising task

## Alternative formula for denoising score matching
Recall the representation \(\tilde{x}=x+\sigma \epsilon\) , with \(\epsilon \sim N(0, I_{n})\) 
\[\left\| s_{\theta}(\tilde{x})-\frac{x-\tilde{x}}{\sigma^{2}}\right\| _{2}^{2}=\left\| s_{\theta}(x+\sigma \epsilon)+\frac{\epsilon}{\sigma}\right\| _{2}^{2}\]

Therefore, the optimization problem can be rewritten as 
\[arg min _{\theta} \mathbb{E}_{x \sim p_{data }} \mathbb{E}_{\tilde{x} | x \sim q_{0 \sigma}}\left[\left\| s_{\theta}(\tilde{x})-\frac{x-\tilde{x}}{\sigma^{2}}\right\| _{2}^{2}\right] =arg min _{\theta} \mathbb{E}_{x \sim p_{data }, \epsilon \sim \mathcal{N}\left(0, I_{n}\right)}\left[\left\| s_{\theta}(x+\sigma \epsilon)+\frac{\epsilon}{\sigma}\right\| _{2}^{2}\right]\]

> Note: We need to choose a very small σ , such that estimated score is close to the ground-truth score

## Pitfall of denoising score matching
- The loss variance will increase drastically as \(\sigma \to 0\) !
- Denoising score matching loss for Gaussian perturbations 
\[\begin{aligned} & \mathbb{E}_{x \sim p_{data }} \mathbb{E}_{\tilde{x} | x \sim q_{0 \sigma}}\left[\left\| s_{\theta}(\tilde{x})-\frac{x-\tilde{x}}{\sigma^{2}}\right\| _{2}^{2}\right] \\ & =\mathbb{E}_{x \sim p_{data }, \epsilon \sim \mathcal{N}\left(0, I_{n}\right)}\left[\left\| s_{\theta}(x+\sigma \epsilon)+\frac{\epsilon}{\sigma}\right\| _{2}^{2}\right] \\ & =\mathbb{E}_{x \sim p_{data }, \epsilon \sim \mathcal{N}\left(0, I_{n}\right)}\left[\left\| s_{\theta}(x+\sigma \epsilon)\right\| _{2}^{2}+2 s_{\theta}(x+\sigma \epsilon)^{\top} \frac{\epsilon}{\sigma}+\frac{\| \epsilon\| _{2}^{2}}{\sigma^{2}}\right] \end{aligned}\]
- If we choose very small \(\sigma \to 0\) 
  - \(Var\left(\frac{\epsilon}{\sigma}\right) \to \infty\)
  - \(Var\left(\frac{\| \epsilon\| _{2}^{2}}{\sigma^{2}}\right) \to \infty\)
- On the other hand, when σ is not small, estimates score of noise-perturbed data 
\[s_{\theta }(x)\approx \nabla _{x}log q_{\sigma }(x)\neq \nabla _{x}log p_{data }(x)\]

## Multi-scale noise perturbation
### Question: How much noise to add?
Answer: Multi-scale noise perturbations with \(\sigma_{1}>\sigma_{2}>...>\sigma_{L-1}>\sigma_{L}\)

## Annealed Langevin dynamics: Joint scores to samples
- Sampling using \(\sigma_{1}>\sigma_{2}>...>\sigma_{L-1}>\sigma_{L}\) sequentially with Langevin dynamics
- Anneal down the noise level
- Samples used as initialization for the next level

### Algorithm 1 Annealed Langevin dynamics
**Require**: \(\{...\}\) (original slide has incomplete params), \(\epsilon\), T
1. Initialize \(x_0\)
2. for \(i=1\) to L do 
3.    \(\alpha_i = \epsilon^2 \sigma_i^2\) (α is the step size)
4.    for \(t \leftarrow1\) to T do 
5.        Draw \(z_t \sim N(0, I)\)
6.        \(x_t = x_{t-1}+\alpha_i(-s_{\theta}(x_{t-1},\sigma_i))+\sqrt{2\alpha_i}z_t\)
7.    end for
8.    \(x_0 \leftarrow x_T\)
9. end for
return \(x_T\)

## Modified training objectives
- Denoising score matching is naturally suitable, since the goal is to estimate the score of perturbed data distributions
- Weighted combination of denoising score matching losses 
\[\frac{1}{L} \sum_{i=1}^{L} \lambda\left(\sigma_{i}\right) \mathbb{E}_{\tilde{x} \sim q_{\sigma_{i}}}\left[\left\| s_{\theta}\left(\tilde{x}, \sigma_{i}\right)-\nabla_{\tilde{x}} log q_{\sigma_{i}}(\tilde{x})\right\| _{2}^{2}\right]\]
\[=\frac{1}{L} \sum_{i=1}^{L} \lambda\left(\sigma_{i}\right) \mathbb{E}_{x \sim p_{data }, \epsilon \sim \mathcal{N}\left(0, I_{n}\right)}\left[\left\| s_{\theta}\left(x+\sigma_{i} \epsilon, \sigma_{i}\right)+\frac{\epsilon}{\sigma_{i}}\right\| _{2}^{2}\right]\]

## Choosing noise scales
### Maximum noise scale
\(\sigma_{1}\) = maximum pairwise distance between data points

### Minimum noise scale
\(\sigma_{L}\) should be small so that noise in final samples is negligible

### Main intuition
Adjacent noise scales should have sufficient overlap to facilitate transitioning across noise scales in annealed Langevin dynamics
- A geometric progression with sufficient length: 
  \[\sigma_{1}>\sigma_{2}>...>\sigma_{L-1}>\sigma_{L}\]
  \[\frac{\sigma_{1}}{\sigma_{2}}=\frac{\sigma_{2}}{\sigma_{3}}=...=\frac{\sigma_{L-1}}{\sigma_{L}}\]

## Choosing the weighting function
- Weighted combination of denoising score matching losses 
\[\frac{1}{L} \sum_{i=1}^{L} \lambda\left(\sigma_{i}\right) \mathbb{E}_{x \sim p_{data }, \epsilon \sim \mathcal{N}\left(0, I_{n}\right)}\left[\left\| s_{\theta}\left(x+\sigma_{i} \epsilon, \sigma_{i}\right)+\frac{\epsilon}{\sigma_{i}}\right\| _{2}^{2}\right]\]
- How to choose the weighting function \(\lambda: \mathbb{R}_{+} \to \mathbb{R}_{+}\) ?
- Goal: Balancing different score matching losses in the sum, which gives \(\lambda(\sigma)=\sigma^{2}\) 
\[\begin{aligned} & \frac{1}{L} \sum_{i=1}^{L} \sigma_{i}^{2} \mathbb{E}_{x \sim p_{data }, \epsilon \sim \mathcal{N}\left(0, I_{n}\right)}\left[\left\| s_{\theta}\left(x+\sigma_{i} \epsilon, \sigma_{i}\right)+\frac{\epsilon}{\sigma_{i}}\right\| _{2}^{2}\right] \\ & =\frac{1}{L} \sum_{i=1}^{L} \mathbb{E}_{x \sim p_{data }, \epsilon \sim \mathcal{N}\left(0, I_{n}\right)}\left[\left\| \sigma_{i} s_{\theta}\left(x+\sigma_{i} \epsilon, \sigma_{i}\right)+\epsilon\right\| _{2}^{2}\right] \\ & =\frac{1}{L} \sum_{i=1}^{L} \mathbb{E}_{x \sim p_{data }, \epsilon \sim \mathcal{N}\left(0, I_{n}\right)}\left[\left\| \epsilon_{\theta}\left(x+\sigma_{i} \epsilon, \sigma_{i}\right)+\epsilon\right\| _{2}^{2}\right] (where \epsilon_{\theta}\left(\cdot, \sigma_{i}\right)=\sigma_{i} s_{\theta}\left(\cdot, \sigma_{i}\right)) \end{aligned}\]

## High resolution image generation
(original slide has no specific content for this section, only title)

## Sampling as iterative denoising
- **Forward process**: Gradually add Gaussian noise
- **Reverse process**: Iteratively remove noise
- Process flow: \(X_0 \to ... \to X_t \to ... \to X_{T-1} \to X_T\) (forward, add noise); \(X_T \to ... \to X_0\) (reverse, remove noise via Langevin dynamics, start from pure noise)

## Denoising diffusion probabilistic model (DDPM)
- Two processes analogous to the encoder (forward process) and decoder (reverse process) in a VAE
- A sequence of variables \(x_{0}, x_{1}, ..., x_{T}\) whose joint distribution is \(q_{\phi}(x_{0: T})\) and \(p_{\theta}(x_{0: T})\) for the forward and reverse processes, respectively
- To make the processes tractable, a Markov chain structure is imposed where 
  - Forward from \(x_{0}\) to \(x_{T}\): \(q_{\phi}\left(x_{0: T}\right)=q\left(x_{0}\right) \prod_{i=1}^{T} q_{\phi}\left(x_{t} | x_{t-1}\right)\)
  - Reverse from \(x_{T}\) to \(x_{0}\): \(p_{\theta}\left(x_{0: T}\right)=p\left(x_{T}\right) \prod_{i=1}^{T} p_{\theta}\left(x_{t-1} | x_{t}\right)\)
- The transition distribution \(q_{\phi}(x_{t} | x_{t-1})\) is set to (where \(0<\alpha_{t}<1\))
\[q_{\phi}\left(x_{t} | x_{t-1}\right)=\mathcal{N}\left(x_{t} ; \sqrt{\alpha_{t}} x_{t-1},\left(1-\alpha_{t}\right) I_{n}\right)\]
- For \(\bar{\alpha}_{t}=\prod_{i=1}^{t} \alpha_{i}\) (and \(\bar{\alpha}_{0}=1\), \(\bar{\alpha}_{T} ≈0\) ), it is straightforward to derive 
\[q_{\phi}\left(x_{t} | x_{0}\right)=\mathcal{N}\left(x_{t} ; \sqrt{\overline{\alpha}_{t}} x_{0},\left(1-\overline{\alpha}_{t}\right) I_{n}\right)\]
- Such a setting ensures that the variance magnitude is preserved so that it will not explode and vanish after many iterations

## Variational inference of VAEs
Recall that for a VAE, we have an encoder \(q_{\theta}(z | x)\) and a decoder \(p_{\theta}(x | z)\)
- The corresponding variational inference proceeds as follows: 
\[\begin{aligned} & log p_{\theta}(x)=log \int p_{\theta}(x, z) d z \\ & =log \int q_{\phi}(z | x) \frac{p_{\theta}(x, z)}{q_{\phi}(z | x)} d z \\ & \geq \int q_{\phi}(z | x) log \frac{p_{\theta}(x, z)}{q_{\phi}(z | x)} d z \\ & =\int q_{\phi}(z | x) log \frac{p(z) p_{\theta}(x | z)}{q_{\phi}(z | x)} d z \\ & =\mathbb{E}_{q_{\phi}(z | x)}\left[log\ p_{\theta}(x | z)\right]-D_{KL}\left(q_{\phi}(z | x) \| p(z)\right) \end{aligned}\]

## Variational inference of diffusion models
- Similarly to that for VAEs, for diffusion models, we have 
\[\begin{aligned} & log p_{\theta}\left(x_{0}\right)=log \int p_{\theta}\left(x_{0: T}\right) d x_{1} ... d x_{T} \\ & =log \int q_{\phi}\left(x_{1: T} | x_{0}\right) \frac{p_{\theta}\left(x_{0: T}\right)}{q_{\phi}\left(x_{1: T} | x_{0}\right)} d x_{1} ... d x_{T} \\ & \geq \int q_{\phi}\left(x_{1: T} | x_{0}\right) log \frac{p_{\theta}\left(x_{0: T}\right)}{q_{\phi}\left(x_{1: T} | x_{0}\right)} d x_{1} ... d x_{T} \\ & =\int q_{\phi}\left(x_{1: T} | x_{0}\right) log \frac{p\left(x_{T}\right) \prod_{t=1}^{T} p_{\theta}\left(x_{t-1} | x_{t}\right)}{\prod_{t=1}^{T} q_{\phi}\left(x_{t} | x_{t-1}\right)} d x_{1} ... d x_{T} \end{aligned}\]
- We need to take care of the ratio \(\frac{\prod_{t=1}^{T} p_{\theta}(x_{t-1} | x_{t})}{\prod_{t=1}^{T} q_{\phi}(x_{t} | x_{t-1})}\) . It can be expressed as 
\[\frac{p_{\theta}\left(x_{0} | x_{1}\right) \prod_{t=1}^{T-1} p_{\theta}\left(x_{t} | x_{t+1}\right)}{q_{\phi}\left(x_{T} | x_{T-1}\right) \prod_{t=1}^{T-1} q_{\phi}\left(x_{t} | x_{t-1}\right)}\]
But we need to deal with \(x_{t-1}\) , \(x_{t}\) , \(x_{t+1}\) at the same time, which is cumbersome

### Remedy: Bayes’ rule
\[q_{\phi }\left(x_{t} | x_{t-1}\right) =q_{\phi }(x_{t}|x_{t-1},x_{0})=\frac {q_{\phi }(x_{t-1}|x_{t},x_{0})q_{\phi }(x_{t}|x_{0})}{q_{\phi }(x_{t-1}|x_{0})}\]

Then, 
\[\begin{array} {rl}&{\frac {\prod _{t=1}^{T}p_{\theta }(x_{t-1} | x_{t})}{\prod _{t=1}^{T}q_{\phi }(x_{t}|x_{t-1})}=\frac {\prod _{t=1}^{T}p_{\theta }(x_{t-1}|x_{t})}{q_{\phi }(x_{1}|x_{0})\prod _{t=2}^{T}\frac {q_{\phi }(x_{t-1}|x_{t},x_{0})q_{\phi }(x_{t}|x_{0})}{q_{\phi }(x_{t-1}|x_{0})}}}\\ &{=\frac {p_{\theta }(x_{0}|x_{1})\prod _{t=2}^{T}p_{\theta }(x_{t-1}|x_{t})}{q_{\phi }(x_{T}|x_{0})\prod _{t=2}^{T}q_{\phi }(x_{t-1}|x_{t},x_{0})}}\end{array}\]

Therefore, the evidence lower bound becomes 
\[\begin{aligned} & \int q_{\phi}\left(x_{1: T} | x_{0}\right) log \frac{p\left(x_{T}\right) \prod_{t=1}^{T} p_{\theta}\left(x_{t-1} | x_{t}\right)}{\prod_{t=1}^{T} q_{\phi}\left(x_{t} | x_{t-1}\right)} d x_{1} ... d x_{T} \\ & =\int q_{\phi}\left(x_{1: T} | x_{0}\right) log \frac{p\left(x_{T}\right) p_{\theta}\left(x_{0} | x_{1}\right)}{q_{\phi}\left(x_{T} | x_{0}\right)} d x_{1} ... d x_{T} \\ & -\int q_{\phi}\left(x_{1: T} | x_{0}\right) \sum_{t=2}^{T} log \frac{q_{\phi}\left(x_{t-1} | x_{t}, x_{0}\right)}{p_{\theta}\left(x_{t-1} | x_{t}\right)} d x_{1} ... d x_{T} \end{aligned}\]

### Further derivation
Note that 
\[\begin{aligned} & \int q_{\phi}\left(x_{1: T} | x_{0}\right) log \frac{p\left(x_{T}\right) p_{\theta}\left(x_{0} | x_{1}\right)}{q_{\phi}\left(x_{T} | x_{0}\right)} d x_{1} ... d x_{T} \\ & =\int q_{\phi}\left(x_{1: T} | x_{0}\right) log p_{\theta}\left(x_{0} | x_{1}\right) d x_{1} ... d x_{T} \\ & -\int q_{\phi}\left(x_{1: T} | x_{0}\right) log \frac{q_{\phi}\left(x_{T} | x_{0}\right)}{p\left(x_{T}\right)} d x_{1} ... d x_{T} \\ & =\int q_{\phi}\left(x_{1} | x_{0}\right) log p_{\theta}\left(x_{0} | x_{1}\right) d x_{1}-\int q_{\phi}\left(x_{T} | x_{0}\right) log \frac{q_{\phi}\left(x_{T} | x_{0}\right)}{p\left(x_{T}\right)} d x_{T} \\ & =\int q_{\phi}\left(x_{1} | x_{0}\right) log p_{\theta}\left(x_{0} | x_{1}\right) d x_{1}-D_{KL}\left(q_{\phi}\left(x_{T} | x_{0}\right) \| p\left(x_{T}\right)\right) \end{aligned}\]

Note that \(q_{\phi}(x_{T}|x_{0}) = \mathcal{N}(x_{T}; \sqrt{\bar{\alpha}_T}x_0, (1-\bar{\alpha}_{T}) I_{n})\) with \(\bar{\alpha}_{T} ≈0\) and \(p(x_{T})=\mathcal{N}(x_{T} ; 0, I_{n})\) , which gives \(D_{KL}(q_{\phi}(x_{T} | x_{0}) \| p(x_{T})) ≈0\)

Suppose \(p_{\theta}(x_{0} | x_{1})=\mathcal{N}(x_{0} ; x_{\theta}(x_{1}, 1), \sigma_{q}^{2}(1) I_{n})\) , we obtain (omitting constants) 
\[\begin{aligned} & \int q_{\phi}\left(x_{1} | x_{0}\right) log p_{\theta}\left(x_{0} | x_{1}\right) d x_{1}=-\mathbb{E}_{q_{\phi}\left(x_{1} | x_{0}\right)}\left[\frac{1}{2 \sigma_{q}^{2}(1)}\left\| x_{0}-x_{\theta}\left(x_{1}, 1\right)\right\| _{2}^{2}\right] \\ & =-\mathbb{E}_{q_{\phi}\left(x_{1} | x_{0}\right)}\left[\frac{1}{2 \sigma_{q}^{2}(1)} \frac{\left(1-\alpha_{1}\right)^{2} \overline{\alpha}_{0}}{\left(1-\overline{\alpha}_{1}\right)^{2}}\left\| x_{0}-x_{\theta}\left(x_{1}, 1\right)\right\| _{2}^{2}\right]\left( where \alpha_{0}=\overline{\alpha}_{0}=1\right) \end{aligned}\]

### For \(2 ≤t ≤T\)
\[\begin{aligned} & \int q_{\phi}\left(x_{1: T} | x_{0}\right) log \frac{q_{\phi}\left(x_{t-1} | x_{t}, x_{0}\right)}{p_{\theta}\left(x_{t-1} | x_{t}\right)} d x_{1} ... d x_{T}=\int q_{\phi}\left(x_{t-1}, x_{t} | x_{0}\right) log \frac{q_{\phi}\left(x_{t-1} | x_{t}, x_{0}\right)}{p_{\theta}\left(x_{t-1} | x_{t}\right)} d x_{t-1} d x_{t} \\ & =\int q_{\phi}\left(x_{t} | x_{0}\right) q_{\phi}\left(x_{t-1} | x_{t}, x_{0}\right) log \frac{q_{\phi}\left(x_{t-1} | x_{t}, x_{0}\right)}{p_{\theta}\left(x_{t-1} | x_{t}\right)} d x_{t-1} d x_{t} \\ & =\int q_{\phi}\left(x_{t} | x_{0}\right) D_{KL}\left(q_{\phi}\left(x_{t-1} | x_{t}, x_{0}\right) \| p_{\theta}\left(x_{t-1} | x_{t}\right)\right) d x_{t} \end{aligned}\]

Note that from Bayes’ rule, 
\[\begin{aligned} & q_{\phi}\left(x_{t-1} | x_{t}, x_{0}\right)=\frac{q_{\phi}\left(x_{t} | x_{t-1}\right) q_{\phi}\left(x_{t-1} | x_{0}\right)}{q_{\phi}\left(x_{t} | x_{0}\right)} \\ & =\frac{\mathcal{N}\left(x_{t} ; \sqrt{\alpha_{t}} x_{t-1}, 1-\alpha_{t} I_{n}\right) \mathcal{N}\left(x_{t-1} ; \sqrt{\overline{\alpha}_{t-1}} x_{0},\left(1-\overline{\alpha}_{t-1}\right) I_{n}\right)}{\mathcal{N}\left(x_{t} ; \sqrt{\overline{\alpha}_{t}} x_{0},\left(1-\overline{\alpha}_{t}\right) I_{n}\right)} \end{aligned}\]

Elementary calculations give that \(q_{\phi}(x_{t-1} | x_{t}, x_{0})=\mathcal{N}(x_{t-1} ; \mu_{q}(x_{t}, x_{0}), \sigma_{q}^{2}(t) I_{n})\) with 
\[\mu_{q}\left(x_{t}, x_{0}\right)=\frac{\left(1-\overline{\alpha}_{t-1}\right) \sqrt{\alpha_{t}}}{1-\overline{\alpha}_{t}} x_{t}+\frac{\left(1-\alpha_{t}\right) \sqrt{\overline{\alpha}_{t-1}}}{1-\overline{\alpha}_{t}} x_{0}, \sigma_{q}^{2}(t)=\frac{\left(1-\alpha_{t}\right)\left(1-\overline{\alpha}_{t-1}\right)}{1-\overline{\alpha}_{t}}\]

By setting \(p_{\theta}(x_{t-1} | x_{t})=\mathcal{N}(x_{t-1} ; \mu_{\theta}(x_{t}), \sigma_{q}^{2}(t) I_{n})\) , we obtain 
\[D_{KL}\left(q_{\phi}\left(x_{t-1} | x_{t}, x_{0}\right) \| p_{\theta}\left(x_{t-1} | x_{t}\right)\right)=\frac{1}{2 \sigma_{q}^{2}(t)}\left\| \mu_{\theta}\left(x_{t}\right)-\mu_{q}\left(x_{t}, x_{0}\right)\right\| _{2}^{2}\]

If further setting \(\mu_{\theta}(x_{t})=\frac{(1-\bar{\alpha}_{t-1}) \sqrt{\alpha_{t}}}{1-\bar{\alpha}_{t}} x_{t}+\frac{(1-\alpha_{t}) \sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t}} x_{\theta}(x_{t}, t)\) , we obtain 
\[D_{KL}\left(q_{\phi}\left(x_{t-1} | x_{t}, x_{0}\right) \| p_{\theta}\left(x_{t-1} | x_{t}\right)\right)=\frac{1}{2 \sigma_{q}^{2}(t)} \cdot \frac{\left(1-\alpha_{t}\right)^{2} \overline{\alpha}_{t-1}}{\left(1-\overline{\alpha}_{t}\right)^{2}}\left\| x_{\theta}\left(x_{t}, t\right)-x_{0}\right\| _{2}^{2}\]

Therefore, we eventually derive the evidence lower bound for a single \(x_{0}\) : 
\[-\sum_{t=1}^{T} \mathbb{E}_{q_{\phi}\left(x_{t} | x_{0}\right)}\left[\frac{1}{2 \sigma_{q}^{2}(t)} \cdot \frac{\left(1-\alpha_{t}\right)^{2} \overline{\alpha}_{t-1}}{\left(1-\overline{\alpha}_{t}\right)^{2}}\left\| x_{\theta}\left(x_{t}, t\right)-x_{0}\right\| _{2}^{2}\right]\]

Additionally, since \(p_{\theta}(x_{t-1} | x_{t})=\mathcal{N}(x_{t-1} ; \mu_{\theta}(x_{t}), \sigma_{q}^{2}(t) I_{n})\) , the sampling procedure for \(t=T, T-1, ..., 1\) is (with \(\epsilon_{t} \sim N(0, I_{n})\)) 
\[x_{t-1}=\mu_{\theta}\left(x_{t}\right)+\sigma_{q}(t) \epsilon_{t}=\frac{\left(1-\overline{\alpha}_{t-1}\right) \sqrt{\alpha_{t}}}{1-\overline{\alpha}_{t}} x_{t}+\frac{\left(1-\alpha_{t}\right) \sqrt{\overline{\alpha}_{t-1}}}{1-\overline{\alpha}_{t}} x_{\theta}\left(x_{t}, t\right)+\sigma_{q}(t) \epsilon_{t}\]

## Data prediction vs. noise prediction
For \(x_{t}=\sqrt{\bar{\alpha}_{t}} x_{0}+\sqrt{1-\bar{\alpha}_{t}} \epsilon_{t}\) with \(\epsilon_{t} \sim N(0, I_{n})\) , we use \(x_{\theta}(x_{t}, t)\) to predict \(x_{0}\), and \(x_{\theta}\) is called a **data prediction network**
- Alternatively, we may use a **noise prediction network** \(\epsilon_{\theta}\) such that \(\epsilon_{\theta}(x_{t}, t)\) predicts \(\epsilon_{t}\)

Since \(x_{0}=(x_{t}-\sqrt{1-\bar{\alpha}_{t}} \epsilon_{t}) / \sqrt{\bar{\alpha}_{t}}\) , we derive 
\[\begin{aligned} & \mu_{q}\left(x_{t}, x_{0}\right)=\frac{\left(1-\overline{\alpha}_{t-1}\right) \sqrt{\alpha_{t}}}{1-\overline{\alpha}_{t}} x_{t}+\frac{\left(1-\alpha_{t}\right) \sqrt{\overline{\alpha}_{t-1}}}{1-\overline{\alpha}_{t}} x_{0} \\ & =\frac{\left(1-\overline{\alpha}_{t-1}\right) \sqrt{\alpha_{t}}}{1-\overline{\alpha}_{t}} x_{t}+\frac{\left(1-\alpha_{t}\right) \sqrt{\overline{\alpha}_{t-1}}}{1-\overline{\alpha}_{t}} \frac{x_{t}-\sqrt{1-\overline{\alpha}_{t}} \epsilon_{t}}{\sqrt{\overline{\alpha}_{t}}}=\frac{1}{\sqrt{\alpha_{t}}} x_{t}-\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha}_{t}} \sqrt{\alpha_{t}}} \epsilon_{t} \end{aligned}\]

In order to match this form of \(\mu_{q}(x_{t}, x_{0})\) , we choose 
\[\mu_{\theta}\left(x_{t}\right)=\frac{1}{\sqrt{\alpha_{t}}} x_{t}-\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha}_{t}} \sqrt{\alpha_{t}}} \epsilon_{\theta}\left(x_{t}, t\right)\]

Then the evidence lower bound for a single \(x_{0}\) becomes: 
\[-\sum_{t=1}^{T} \mathbb{E}_{q_{\phi}\left(x_{t} | x_{0}\right)}\left[\frac{1}{2 \sigma_{q}^{2}(t)} \cdot \frac{\left(1-\alpha_{t}\right)^{2}}{\left(1-\overline{\alpha}_{t}\right) \alpha_{t}}\left\| \epsilon_{\theta}\left(x_{t}, t\right)-\epsilon_{t}\right\| _{2}^{2}\right]\]

Since \(p_{\theta}(x_{t-1} | x_{t})=\mathcal{N}(x_{t-1} ; \mu_{\theta}(x_{t}), \sigma_{q}^{2}(t) I_{n})\) , the sampling procedure becomes 
\[x_{t-1}=\mu_{\theta}\left(x_{t}\right)+\sigma_{q}(t) \epsilon_{t}=\frac{1}{\sqrt{\alpha_{t}}}\left(x_{t}-\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha}_{t}}} \epsilon_{\theta }\left(x_{t},t\right)\right)+\sigma_{q}(t) \epsilon_{t}\]

## Denoising diffusion implicit models (DDIM)
Since 
\[x_{t}=\sqrt{\overline{\alpha}_{t}} x_{0}+\sqrt{1-\overline{\alpha}_{t}} \epsilon\]
\[x_{t-1} =\sqrt{\overline{\alpha}_{t-1}} x_{0}+\sqrt{1-\overline{\alpha}_{t-1}} \epsilon' = \sqrt{\overline{\alpha}_{t-1}} x_{0}+\sqrt{1-\overline{\alpha}_{t-1}} \frac{x_{t}-\sqrt{\overline{\alpha}_{t}} x_{0}}{\sqrt{1-\overline{\alpha}_{t}}}\]

This motivates us to consider 
\[q\left(x_{t-1} | x_{t}, x_{0}\right)=\mathcal{N}\left(x_{t-1} ; \sqrt{\overline{\alpha}_{t-1}} x_{0}+\sqrt{1-\overline{\alpha}_{t-1}} \frac{x_{t}-\sqrt{\overline{\alpha}_{t}} x_{0}}{\sqrt{1-\overline{\alpha}_{t}}}, \sigma_{t}^{2} I_{n}\right)\]

we need 
\[q\left(x_{t-1} | x_{t}, x_{0}\right)=\mathcal{N}\left(x_{t-1} ; \sqrt{\overline{\alpha}_{t-1}} x_{0}+\sqrt{1-\overline{\alpha}_{t-1}-\sigma_{t}^{2}} \frac{x_{t}-\sqrt{\overline{\alpha}_{t}} x_{0}}{\sqrt{1-\overline{\alpha}_{t}}}, \sigma_{t}^{2} I_{n}\right)\]

and 
\[q\left(x_{t-1} | x_{0}\right)=\int q\left(x_{t-1} | x_{t}, x_{0}\right) q\left( x_{t} | x_{0}\right) dx_{t}\]