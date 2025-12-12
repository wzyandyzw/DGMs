# Score-Based Continuous-Time Diffusion Models
**Author**: Zhaoqiang Liu
**Affiliation**: School of Computer Science & Mathematics (joint appointment) University of Electronic Science and Technology of China
**Date**: 28 October, 2025

---

## Slide 1/ 16: Recap: Score-based models
For $\tilde{x}=x+\sigma \epsilon$ with $\epsilon \sim N(0, I_{n})$ , denoising score matching shows 
$$arg min _{\theta} \mathbb{E}_{\tilde{x} \sim q_{\sigma}}\left[\left\| s_{\theta}(\tilde{x})-\nabla_{\tilde{x}} log q_{\sigma}(\tilde{x})\right\| _{2}^{2}\right]=arg min _{\theta} \mathbb{E}_{x \sim p_{data }} \mathbb{E}_{\tilde{x} | x \sim q_{0 \sigma}}\left[\left\| s_{\theta}(\tilde{x})-\frac{x-\tilde{x}}{\sigma^{2}}\right\| _{2}^{2}\right]$$

By considering multi-scale noise perturbations 
$$\sigma_{1}>\sigma_{2}>...>\sigma_{L-1}>\sigma_{L}$$
score-based models perform the training w.r.t. the following objective 
$$\frac{1}{L} \sum_{i=1}^{L} \lambda\left(\sigma_{i}\right) \mathbb{E}_{x \sim p_{data }, \epsilon \sim \mathcal{N}\left(0, I_{n}\right)}\left[\left\| s_{\theta}\left(x+\sigma_{i} \epsilon, \sigma_{i}\right)+\frac{\epsilon}{\sigma_{i}}\right\| _{2}^{2}\right]$$

And perform the sampling via annealed Langevin dynamics:
### Algorithm 1 Annealed Langevin dynamics
1. Require:相关参数（原文未完整标注）
2. Initialize $x_0$
3. for $i=1$ to $L$ do
    4. $a_i \leftarrow e^{02/02}$（原文公式可能存在笔误，$a_i$ is the step size）
    5. for $t \leftarrow1$ to $T$ do
        6. Draw $z_t \sim N(0, I)$
        7. $x_t \leftarrow x_{t-1} + a_i s_\theta(x_{t-1}) + \sqrt{2a_i}z_t$（原文公式不完整，按Langevin动力学常规形式补充）
    8. end for
9. end for
10. return $x_T$

---

## Slide 2/ 16: Recap: Denoising diffusion probabilistic model (DDPM)
- Two processes analogous to the encoder (forward process) and decoder (reverse process) in a VAE
- A sequence of variables $x_{0}, x_{1}, ..., x_{T}$ whose joint distribution is $q_{\phi}(x_{0: T})$ and $p_{\theta}(x_{0: T})$ for the forward and reverse processes, respectively
- To make the processes tractable, a Markov chain structure is imposed where 
    $$forward \ from \ x_{0} \ to \ x_{T}: q_{\phi}\left(x_{0: T}\right)=q\left(x_{0}\right) \prod_{i=1}^{T} q_{\phi}\left(x_{t} | x_{t-1}\right)$$
    $$reverse \ from \ x_{T} \ to \ x_{0}: p_{\theta}\left(x_{0: T}\right)=p\left(x_{T}\right) \prod_{i=1}^{T} p_{\theta}\left(x_{t-1} | x_{t}\right)$$

The transition distribution $q_{\phi}(x_{t} | x_{t-1})$ is set to (where $0<\alpha_{t}<1$ ) 
$$q_{\phi}\left(x_{t} | x_{t-1}\right)=\mathcal{N}\left(x_{t} ; \sqrt{\alpha_{t}} x_{t-1},\left(1-\alpha_{t}\right) I_{n}\right)$$

For $\bar{\alpha}_{t}=\prod_{i=1}^{t} \alpha_{i}$ (and $\bar{\alpha}_{0}=1$, $\bar{\alpha}_{T} \approx0$ ), it is straightforward to derive 
$$q_{\phi}\left(x_{t} | x_{0}\right)=\mathcal{N}\left(x_{t} ; \sqrt{\overline{\alpha}_{t}} x_{0},\left(1-\overline{\alpha}_{t}\right) I_{n}\right)$$（原文公式中$x_{t-1}$应为$x_0$，属于笔误修正）

- Such a setting ensures that the variance magnitude is preserved so that it will not explode and vanish after many iterations

---

## Slide 3/ 16: Recap: Variational inference of DDPM
To derive an evidence lower bound for DDPM, there are three main steps:
1. Following the procedure of VAEs to derive 
$$log p_{\theta}(x) \geq \int q_{\phi}\left(x_{1: T} | x_{0}\right) log \frac{p\left(x_{T}\right) \prod_{t=1}^{T} p_{\theta}\left(x_{t-1} | x_{t}\right)}{\prod_{t=1}^{T} q_{\phi}\left(x_{t} | x_{t-1}\right)} d x_{1} ... d x_{T}$$

2. Using Bayes’ rule 
$$q_{\phi }\left(x_{t}|x_{t-1}\right) =q_{\phi }(x_{t}|x_{t-1},x_{0})=\frac {q_{\phi }(x_{t-1}|x_{t},x_{0})q_{\phi }(x_{t}|x_{0})}{q_{\phi }(x_{t-1}|x_{0})}$$
which leads to 
$$\begin{gathered} log p_{\theta}(x) \geq \int q_{\phi}\left(x_{1} | x_{0}\right) log p_{\theta}\left(x_{0} | x_{1}\right) d x_{1}-D_{KL}\left(q_{\phi}\left(x_{T} | x_{0}\right) \| p\left(x_{T}\right)\right) \\ -\sum_{t=2}^{T} \int q_{\phi}\left(x_{t} | x_{0}\right) D_{KL}\left(q_{\phi}\left(x_{t-1} | x_{t}, x_{0}\right) \| p_{\theta}\left(x_{t-1} | x_{t}\right)\right) d x_{t} \end{gathered}$$

3. Using Bayes’ rule to calculate $q_{\phi}(x_{t-1} | x_{t}, x_{0})=N(x_{t-1} ; \mu_{q}(x_{t}, x_{0}), \sigma_{q}^{2}(t) I_{n})$ with 
$$\mu_{q}\left(x_{t}, x_{0}\right)=\frac{\left(1-\overline{\alpha}_{t-1}\right) \sqrt{\alpha_{t}}}{1-\overline{\alpha}_{t}} x_{t}+\frac{\left(1-\alpha_{t}\right) \sqrt{\overline{\alpha}}_{t-1}}{1-\overline{\alpha}_{t}} x_{0}, \sigma_{q}^{2}(t)=\frac{\left(1-\alpha_{t}\right)\left(1-\overline{\alpha}_{t-1}\right)}{1-\overline{\alpha}_{t}}$$

---

## Slide 4/ 16: Variational inference of diffusion models
By setting $p_{\theta}(x_{t-1} | x_{t})=N(x_{t-1} ; \mu_{\theta}(x_{t}), \sigma_{q}^{2}(t) I_{n})$ , we obtain 
$$D_{KL}\left(q_{\phi}\left(x_{t-1} | x_{t}, x_{0}\right) \| p_{\theta}\left(x_{t-1} | x_{t}\right)\right)=\frac{1}{2 \sigma_{q}^{2}(t)}\left\| \mu_{\theta}\left(x_{t}\right)-\mu_{q}\left(x_{t}, x_{0}\right)\right\| _{2}^{2}$$

If further setting $\mu_{\theta}(x_{t})=\frac{(1-\bar{\alpha}_{t-1}) \sqrt{\alpha_{t}}}{1-\bar{\alpha}_{t}} x_{t}+\frac{(1-\alpha_{t}) \sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t}} x_{\theta}(x_{t}, t)$ , we obtain 
$$D_{KL}\left(q_{\phi}\left(x_{t-1} | x_{t}, x_{0}\right) \| p_{\theta}\left(x_{t-1} | x_{t}\right)\right)=\frac{1}{2 \sigma_{q}^{2}(t)} \cdot \frac{\left(1-\alpha_{t}\right)^{2} \overline{\alpha}_{t-1}}{\left(1-\overline{\alpha}_{t}\right)^{2}}\left\| x_{\theta}\left(x_{t}, t\right)-x_{0}\right\| _{2}^{2}$$

Therefore, we eventually derive the evidence lower bound for a single $x_{0}$ 
$$-\sum_{t=1}^{T} \mathbb{E}_{q_{\phi}\left(x_{t} | x_{0}\right)}\left[\frac{1}{2 \sigma_{q}^{2}(t)} \cdot \frac{\left(1-\alpha_{t}\right)^{2} \overline{\alpha}_{t-1}}{\left(1-\overline{\alpha}_{t}\right)^{2}}\left\| x_{\theta}\left(x_{t}, t\right)-x_{0}\right\| _{2}^{2}\right]$$

Additionally, since $p_{\theta}(x_{t-1} | x_{t})=N(x_{t-1} ; \mu_{\theta}(x_{t}), \sigma_{q}^{2}(t) I_{n})$ , the sampling procedure for $t=T, T-1, ..., 1$ is (with $\epsilon_{t} \sim N(0, I_{n})$) 
$$x_{t-1}=\mu_{\theta}\left(x_{t}\right)+\sigma_{q}(t) \epsilon_{t}=\frac{\left(1-\overline{\alpha}_{t-1}\right) \sqrt{\alpha_{t}}}{1-\overline{\alpha}_{t}} x_{t}+\frac{\left(1-\alpha_{t}\right) \sqrt{\overline{\alpha}_{t-1}}}{1-\overline{\alpha}_{t}} x_{\theta}\left(x_{t}, t\right)+\sigma_{q}(t) \epsilon_{t}$$

---

## Slide 5/ 16: Data prediction vs. noise prediction
For $x_{t}=\sqrt{\bar{\alpha}_{t}} x_{0}+\sqrt{1-\bar{\alpha}_{t}} \epsilon_{t}$ with $\epsilon_{t} \sim N(0, I_{n})$ , we use $x_{\theta}(x_{t}, t)$ to predict $x_{0}$, and $x_{\theta}$ is called a **data prediction network**

Alternatively, we may use a **noise prediction network** $\epsilon_{\theta}$ such that $\epsilon_{\theta}(x_{t}, t)$ predicts $\epsilon_{t}$

Since $x_{0}=(x_{t}-\sqrt{1-\bar{\alpha}_{t}} \epsilon_{t}) / \sqrt{\bar{\alpha}_{t}}$ , we derive 
$$\begin{aligned} & \mu_{q}\left(x_{t}, x_{0}\right)=\frac{\left(1-\overline{\alpha}_{t-1}\right) \sqrt{\alpha_{t}}}{1-\overline{\alpha}_{t}} x_{t}+\frac{\left(1-\alpha_{t}\right) \sqrt{\overline{\alpha}_{t-1}}}{1-\overline{\alpha}_{t}} x_{0} \\ & =\frac{\left(1-\overline{\alpha}_{t-1}\right) \sqrt{\alpha_{t}}}{1-\overline{\alpha}_{t}} x_{t}+\frac{\left(1-\alpha_{t}\right) \sqrt{\overline{\alpha}_{t-1}}}{1-\overline{\alpha}_{t}} \frac{x_{t}-\sqrt{1-\overline{\alpha}_{t}} \epsilon_{t}}{\sqrt{\overline{\alpha}_{t}}}=\frac{1}{\sqrt{\alpha_{t}}} x_{t}-\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha}_{t}} \sqrt{\alpha_{t}}} \epsilon_{t} \end{aligned}$$

In order to match this form of $\mu_{q}(x_{t}, x_{0})$ , we choose 
$$\mu_{\theta}\left(x_{t}\right)=\frac{1}{\sqrt{\alpha_{t}}} x_{t}-\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha}_{t}} \sqrt{\alpha_{t}}} \epsilon_{\theta}\left(x_{t}, t\right)$$

Then the evidence lower bound for a single $x_{0}$ becomes: 
$$-\sum_{t=1}^{T} \mathbb{E}_{q_{\phi}\left(x_{t} | x_{0}\right)}\left[\frac{1}{2 \sigma_{q}^{2}(t)} \cdot \frac{\left(1-\alpha_{t}\right)^{2}}{\left(1-\overline{\alpha}_{t}\right) \alpha_{t}}\left\| \epsilon_{\theta}\left(x_{t}, t\right)-\epsilon_{t}\right\| _{2}^{2}\right]$$

Since $p_{\theta}(x_{t-1} | x_{t})=N(x_{t-1} ; \mu_{\theta}(x_{t}), \sigma_{q}^{2}(t) I_{n})$ , the sampling procedure becomes 
$$x_{t-1}=\mu_{\theta}\left(x_{t}\right)+\sigma_{q}(t) \epsilon_{t}=\frac{1}{\sqrt{\alpha_{t}}}\left(x_{t}-\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha}_{t}}} \epsilon_{\theta }\left(x_{t},t\right)\right)+\sigma_{q}(t) \epsilon_{t}$$

---

## Slide 6/ 16: Training and sampling of DDPM
### Algorithm 1 Training
1. repeat
    2. $x_0 \sim q(x_0)$
    3. （原文步骤3缺失）
    4. $\epsilon \sim N(0,I)$
    5. Take gradient descent step on （原文损失函数未完整标注，应为相关预测误差损失）
6. until converged

### Algorithm 2 Sampling
1. $x_T \sim N(0,I)$
2. for $t =T,..., 1$ do
    3. （原文步骤3缺失）
    4. $x_{t-1} \leftarrow \mu_\theta(x_t,t)+\sigma_q(t)\epsilon_t$（按DDPM采样常规形式补充）
5. end for
6. return $x_0$

---

## Slide 7/ 16: Denoising diffusion implicit models (DDIM)
Since 
$$x_{t}=\sqrt{\overline{\alpha}_{t}} x_{0}+\sqrt{1-\overline{\alpha}_{t}} \epsilon$$
$$\begin{aligned} x_{t-1} & =\sqrt{\overline{\alpha}_{t-1}} x_{0}+\sqrt{1-\overline{\alpha}_{t-1}} \frac{x_{t}-\sqrt{\overline{\alpha}_{t}} x_{0}}{\sqrt{1-\overline{\alpha}_{t}}} \end{aligned}$$（原文公式格式有缺失，已修正）

This motivates us to consider 
$$q\left(x_{t-1} | x_{t}, x_{0}\right)=\mathcal{N}\left(x_{t-1} ; \sqrt{\overline{\alpha}_{t-1}} x_{0}+\sqrt{1-\overline{\alpha}_{t-1}} \frac{x_{t}-\sqrt{\overline{\alpha}_{t}} x_{0}}{\sqrt{1-\overline{\alpha}_{t}}}, \sigma_{t}^{2} I_{n}\right)$$

we need 
$$q\left(x_{t-1} | x_{t}, x_{0}\right)=\mathcal{N}\left(x_{t-1} ; \sqrt{\overline{\alpha}_{t-1}} x_{0}+\sqrt{1-\overline{\alpha}_{t-1}-\sigma_{t}^{2}} \frac{x_{t}-\sqrt{\overline{\alpha}_{t}} x_{0}}{\sqrt{1-\overline{\alpha}_{t}}}, \sigma_{t}^{2} I_{n}\right)$$（原文公式中$x_0$下标格式修正）
 
$$q\left(x_{t-1} | x_{0}\right)=\int q\left(x_{t-1} | x_{t}, x_{0}\right) q\left( x_{t} | x_{0}\right) dx_{t}$$

---

## Slide 8/ 16: From iterative algorithms to ordinary differential equations (ODEs)
We study score based diffusion models through the lens of differential equations

First, simple examples show why an iterative algorithm can be related to a differential equation:
For $i=1,2, ..., N$ 
$$x_{i}=\left(1-\frac{\beta \Delta t}{2}\right) x_{i-1}$$

Letting $\Delta t=1 / N$，$x_{i}=x(\frac{i}{N})$ for $t \in\{0,1 ...,(N-1) / N\}$ , we have 
$$x(t+\Delta t)=\left(1-\frac{\beta \Delta t}{2}\right) x(t)$$
which gives $\frac{x(t+\Delta t)-x(t)}{\Delta t}=-\frac{\beta}{2} x(t)$ , leading to (as $\Delta t \to 0$)
$$\frac{d x(t)}{d t}=-\frac{\beta}{2} x(t)$$

---

## Slide 9/ 16: From iterative algorithms to ordinary differential equations (ODEs)
A gradient descent algorithm for a (well-behaved) convex function $f$ is the following recursion. For $i=1,2, ..., N$ 
$$x_{i}=x_{i-1}-\beta_{i-1} \nabla f\left(x_{i-1}\right)$$

Similarly, we obtain (by letting $\beta_{i-1}=\beta(t) \Delta t$) 
$$x_{i}=x_{i-1}-\beta_{i-1} \nabla f\left(x_{i-1}\right) \Rightarrow x(t+\Delta t)=x(t)-\beta(t) \Delta t \nabla f(x(t))$$
$$\frac{d x(t)}{d t}=-\beta(t) \nabla f(x(t))$$

For simplicity, considering $\beta(t)=\beta$ for all $t$ . Then, 
$$\begin{aligned} \frac{d f(x(t))}{d t} & =\nabla f(x(t))^{\top} \frac{d x(t)}{d t} \\ & =\nabla f(x(t))^{\top}(-\beta \nabla f(x(t))) \\ & =-\beta\| \nabla f(x(t))\| _{2}^{2} \leq 0 \end{aligned}$$

Therefore, as we move from $x_{i-1}$ to $x_{i}$ , the objective value $f(x(t))$ has to go down

---

## Slide 10/ 16: Stochastic differential equations (SDEs)
- An SDE is a differential equation in which one or more of the terms is a stochastic process, resulting in a solution which is also a stochastic process
- SDEs have a random differential that is in the most basic case random white noise calculated as the distributional derivative of a Brownian motion
- A typical equation is of the form 
$$d x_{t}=\mu\left(x_{t}, t\right) d t+\nu\left(x_{t}, t\right) d w_{t}$$
where $w_{t}$ denotes a Wiener process (standard Brownian motion)

In mathematics, Brownian motion is described by the Wiener process. It is one of the best known Lévy processes and occurs frequently in pure and applied mathematics
The Wiener process $w_{t}$ is characterized by four facts:
1. $w_{0}=0$
2. $w_{t}$ is almost surely continuous
3. $w_{t}$ has independent increments (if $0 ≤s_{1}<<t_{1} ≤s_{2}<<t_{2}$ , then $w_{t_{2}}-w_{s_{2}}$ and $w_{t_{1}}-w_{s_{1}}$ are independent) 
4. $w_{t}-w_{s} \sim \mathcal{N}\left(0,(t-s) I_{n}\right)$ (for $0 \leq s \leq t$)（原文公式符号笔误修正）

---

## Slide 11/ 16: Stochastic differential equations (SDEs)
The SDE should be interpreted as an informal way of expressing the corresponding integral equation 
$$x_{t}-x_{s}=\int_{s}^{t} \mu\left(x_{\tau}, \tau\right) d \tau+\int_{s}^{t} \nu\left(x_{\tau}, \tau\right) d w_{\tau}$$

A heuristic (but very helpful) interpretation of the SDE is that in a small time interval of length $\delta$ the stochastic process $x_{s}$ changes its value by an amount that is normally distributed with expectation $\mu(x_{s}, s) \delta$ and variance $\nu(x_{s}, s)^{2} \delta$ and is independent of the past behavior of the process

The functions $\mu$ and $\nu$ are referred to as the **drift** and **diffusion coefficients**, respectively. The stochastic process $x_{t}$ is called a diffusion process and satisfies the Markov property

---

## Slide 12/ 16: SDEs: A simple example
Consider the SDE: 
$$d x_{t}=-\frac{\alpha}{2} x_{t} d t+\beta d w_{t}$$

Letting $\Delta t=1 / N$，$t_{i}=i / N$，$x_{i}=x(t_{i})$，$w_{i}=w(t_{i})$ , the discretization gives 
$$x_{i}-x_{i-1}=-\frac{\alpha}{2} x_{i-1} \Delta t+\beta\left(w_{i}-w_{i-1}\right)$$
$$x_{i}=\left(1-\frac{\alpha \Delta t}{2}\right) x_{i-1}+\beta \sqrt{\Delta t} \cdot z_{i-1}, z_{i-1} \sim \mathcal{N}\left(0, I_{n}\right)$$

Therefore, simple calculations give 
$$\begin{aligned} x_{i} & =\left(1-\frac{\alpha \Delta t}{2}\right)^{i} x_{0}+\beta \sqrt{\Delta t} \sum_{j=0}^{i-1}\left(1-\frac{\alpha \Delta t}{2}\right)^{j} z_{j} \\ & \approx e^{-\frac{\alpha i \Delta t}{2}} x_{0}+\beta \sqrt{\Delta t} \sum_{j=0}^{i-1} e^{-\frac{\alpha j \Delta t}{2}} z_{j}=e^{-\frac{\alpha t_{i}}{2}} x_{0}+\beta \sqrt{\Delta t} \sqrt{\sum_{j=0}^{i-1} e^{-\alpha j \Delta t}} \cdot \tilde{z}_{i} \quad\left(\tilde{z}_{i} \sim \mathcal{N}\left(0, I_{n}\right)\right) \\ & =e^{-\frac{\alpha t_{i}}{2}} x_{0}+\beta \sqrt{\Delta t} \sqrt{\frac{1-e^{-\alpha i \Delta t}}{1-e^{-\alpha \Delta t}}} \cdot \tilde{z}_{i} \approx e^{-\frac{\alpha t_{i}}{2}} x_{0}+\frac{\beta}{\sqrt{\alpha}} \sqrt{1-e^{-\alpha t_{i}}} \cdot \tilde{z}_{i} \end{aligned}$$

This shows that for $t \in\{t_{1}, ..., t_{N}\}$ 
$$x_{t} | x_{0} \sim \mathcal{N}\left(e^{-\frac{\alpha t}{2}} x_{0}, \frac{\beta^{2}}{\alpha} \cdot\left(1-e^{-\alpha t}\right) I_{n}\right)$$

---

## Slide 13/ 16: SDEs: A simple example
On the other hand, for the SDE 
$$d x_{t}=-\frac{\alpha}{2} x_{t} d t+\beta d w_{t}$$
it is easy to obtain the analytic solution 
$$x_{t}=e^{-\frac{\alpha t}{2}} x_{0}+\beta \int_{0}^{t} e^{-\frac{\alpha(t-\tau)}{2}} d w_{\tau}$$

Note that for a deterministic function $h(\tau)$ , the Itô integral $\int_{0}^{t} h(\tau) d w_{\tau}$ is Gaussian with zero mean and covariance $(\int_{0}^{t} h(\tau)^{2} d \tau) I_{n}$ (considering dividing $[0, t]$ into many sub-intervals, and we have $\int_{0}^{t} h(\tau)dw_\tau \approx \sum_i \int_{t_{i-1}}^{t_i}h(t_{i-1})(w_{t_i}-w_{t_{i-1}}) = \sum_i \int_{t_{i-1}}^{t_i}h(t_{i-1})\sqrt{\Delta t}z_{i-1}$, which is zero-mean Gaussian with variance $(\sum_{i} \int_{t_{i-1}}^{t_{i}} h(t_{i-1})^{2} \Delta t) I_{n} \approx(\int_{0}^{t} h(\tau)^{2} d \tau) I_{n})$

Then, it is straightforward to see that $\beta \int_{0}^{t} e^{-\frac{\alpha(t-\tau)}{2}} d w_{\tau}$ is zero-mean Gaussian with covariance $\frac{\beta^{2}(1-e^{-\alpha t})}{\alpha} I_{n}$ . This is consistent with the previous result 
$$x_{t} | x_{0} \sim \mathcal{N}\left(e^{-\frac{\alpha t}{2}} x_{0}, \frac{\beta^{2}}{\alpha} \cdot\left(1-e^{-\alpha t}\right) I_{n}\right)$$

---

## Slide 14/ 16: Examples: VE and VP SDEs
For score based models, we consider a sequence of $N$ noise scales $\sigma_{1}<\sigma_{2}<...<\sigma_{N}$ and $x_{i}=x_{0}+\sigma_{i} \epsilon_{i}$ for $\epsilon_{i} \sim N(0, I_{n})$ (or $x_{i} | x_{0} \sim N(x_{0}, \sigma_{i}^{2} I_{n})$)

This corresponds to the distribution of $x_{i}$ in the following Markov chain: 
$$x_{i}=x_{i-1}+\sqrt{\sigma_{i}^{2}-\sigma_{i-1}^{2}} z_{i-1}, i=1, ..., N$$（原文公式中根号覆盖范围修正）

In the limit of $N \to \infty$，$\{\sigma_{i}\}_{i=1}^{N}$ becomes a function $\sigma(t)$ , $z_{i}$ becomes $z(t)$ , and the Markov chain $\{x_{i}\}_{i=1}^{N}$ becomes a continuous stochastic process $\{x_{t}\}_{t \in[0,1]}$

For $\Delta t=1 / N$，$t_{i}=i / N$，$x_{i}=x(t_{i})$，$z_{i}=z(t_{i})$ , and $t \in\{0,1 / N, ...,(N-1) / N\}$
$$x(t+\Delta t)=x(t)+\sqrt{\sigma(t+\Delta t)^{2}-\sigma(t)^{2}} z(t)=x(t)+\sqrt{\frac{\sigma(t+\Delta t)^{2}-\sigma(t)^{2}}{\Delta t}} \cdot \sqrt{\Delta t} z(t)$$（原文公式格式修正）

Then, the process $\{x_{t}\}_{t \in[0,1]}$ is given by the following **variance exploding (VE) SDE** 
$$dx_{t}=\sqrt{\frac{d\left(\sigma(t)^{2}\right)}{d t}} d w_{t}$$

---

## Slide 15/ 16: Examples: VE and VP SDEs
For DDPM, the discrete Markov chain is 
$$x_{i}=\sqrt{1-\beta_{i}} x_{i-1}+\sqrt{\beta_{i}} z_{i-1}, i=1, ..., N$$

For $\Delta t=1 / N$，$t_{i}=i / N$，$x_{i}=x(t_{i})$ , and $z_{i}=z(t_{i})$ , if setting $\beta_{i}=\beta(t_{i}) \Delta t$ , we obtain for $t \in\{0,1 / N, ...,(N-1) / N\}$ that 
$$\begin{aligned} x(t+\Delta t) & =\sqrt{1-\beta(t+\Delta t) \Delta t} x(t)+\sqrt{\beta(t+\Delta t) \Delta t} z(t) \\ & \approx\left(1-\frac{1}{2} \beta(t) \Delta t\right) x(t)+\sqrt{\beta(t)} \cdot \sqrt{\Delta t} z(t) \end{aligned}$$（原文公式格式修正）

$$\Rightarrow x(t+\Delta t)-x(t)=-\frac{1}{2} \beta(t) x(t) \Delta t+\sqrt{\beta(t)} \cdot \sqrt{\Delta t} z(t)$$

As $N \to \infty$ , the above iterative procedure converges to the following **variance preserving (VP) SDE** 
$$d x_{t}=-\frac{1}{2} \beta(t) x_{t} d t+\sqrt{\beta(t)} d w_{t}$$

---

## Slide 16/ 16: More general SDEs: Consistency with conditional distributions
In the field of score based diffusion models, it is typical to consider the following SDE for the forward process: 
$$d x_{t}=f(t) x_{t} d t+g(t) d w_{t}$$
where $f(t)$ and $g(t)$ are certain scalar functions w.r.t. time $t$

For recent works of score-based diffusion models, we prefer directly considering the conditional distribution of $x_{t} | x_{0}$ (instead of $x_{t} | x_{t-1}$), which is of the form: 
$$x_{t} | x_{0} \sim \mathcal{N}\left(\alpha(t) x_{0}, \sigma(t)^{2} I_{n}\right)$$
where $\alpha(t)$ , $\sigma(t)$ are non-negative differentiable functions of $t$ with bounded derivatives, and we denote them as $\alpha_{t}$ , $\sigma_{t}$ for brevity

The choice for $\alpha_{t}$ , $\sigma_{t}$ is referred to as the **noise schedule**. The signal-to-noise-ratio (SNR) $\alpha_{t}^{2} / \sigma_{t}^{2}$ is strictly decreasing w.r.t. $t$ , and we typically have $\alpha_{0}=1$ , $\sigma_{0}=0$

To ensure that the forward SDE $d x_{t}=f(t) x_{t} d t+g(t) d w_{t}$ has the same conditional distribution for $x_{t} | x_{0}$ for any $t$ , we need (Why?) 
$$f(t)=\frac {dlog (\alpha _{t})}{dt}, g(t)^{2}=\frac {d\sigma _{t}^{2}}{dt}-2\frac {dlog (\alpha _{t})}{dt}\sigma _{t}^{2}=-2\sigma _{t}^{2}\frac {d\lambda _{t}}{dt}$$
where $\lambda_{t}:=log (\alpha_{t} / \sigma_{t})$ is one half of the log-SNR

---

我可以帮你整理这份markdown文档中的**公式错误和笔误**，形成一份修正说明，需要吗？

