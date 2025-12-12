# Efficient Sampling of Diffusion Models
**Author**: Zhaoqiang Liu
**Affiliation**: School of Computer Science & Mathematics (joint appointment) University of Electronic Science and Technology of China
**Date**: 31 October, 2025

---

## Recap: Stochastic differential equations (SDEs)
- A typical equation is of the form 
\[d x_{t}=\mu\left(x_{t}, t\right) d t+\nu\left(x_{t}, t\right) d w_{t}\]
where \(w_{t}\) denotes a **Wiener process** (standard Brownian motion)

- The Wiener process \(w_{t}\) is characterized by four facts:
  1. \(w_{0}=0\)
  2. \(w_{t}\) is almost surely continuous
  3. \(w_{t}\) has independent increments (if \(0 ≤s_{1}<<t_{1} ≤s_{2}<<t_{2}\), then \(w_{t_{2}}-w_{s_{2}}\) and \(w_{t_{1}}-w_{s_{1}}\) are independent)
  4. \(w_{t}-w_{s} \sim \mathcal{N}\left(0,(t-s) I_{n}\right)\) (for \(0 \leq s \leq t\))

- The SDE should be interpreted as an informal way of expressing the corresponding integral equation 
\[x_{t}-x_{s}=\int_{s}^{t} \mu\left(x_{\tau}, \tau\right) d \tau+\int_{s}^{t} \nu\left(x_{\tau}, \tau\right) d w_{\tau}\]

---

## Recap: A simple SDE
- Consider the SDE: 
\[d x_{t}=-\frac{\alpha}{2} x_{t} d t+\beta d w_{t}\]

- Letting \(\Delta t=1 / N\), \(t_{i}=i / N\), \(x_{i}=x(t_{i})\), \(w_{i}=w(t_{i})\), the discretization gives 
\[x_{i}-x_{i-1}=-\frac{\alpha}{2} x_{i-1} \Delta t+\beta\left(w_{i}-w_{i-1}\right)\]
\[x_{i}=\left(1-\frac{\alpha \Delta t}{2}\right) x_{i-1}+\beta \sqrt{\Delta t} \cdot z_{i-1}, z_{i-1} \sim \mathcal{N}\left(0, I_{n}\right)\]

- On the other hand, it is easy to obtain the analytic solution 
\[x_{t}=e^{-\frac{\alpha t}{2}} x_{0}+\beta \int_{0}^{t} e^{-\frac{\alpha(t-\tau)}{2}} d w_{\tau}\]

- The numerical and analytic solutions approximately give the same conditional distribution 
\[x_{t} | x_{0} \sim \mathcal{N}\left(e^{-\frac{\alpha t}{2}} x_{0}, \frac{\beta^{2}}{\alpha} \cdot\left(1-e^{-\alpha t}\right) I_{n}\right)\]

---

## Recap: VE and VP SDEs
- For score based models, we consider a sequence of N noise scales \(\sigma_{1}<\sigma_{2}<...<\sigma_{N}\) and \(x_{i}=x_{0}+\sigma_{i} \epsilon_{i}\) for \(\epsilon_{i} \sim N(0, I_{n})\) (or \(x_{i} | x_{0} \sim N(x_{0}, \sigma_{i}^{2} I_{n})\))

- This corresponds to the distribution of \(x_{i}\) in the following Markov chain: 
\[x_{i}=x_{i-1}+\sqrt{\sigma_{i}^{2}-\sigma_{i-1}^{2}} z_{i-1}, i=1, ..., N\]
As \(N \to \infty\), the iterative procedure converges to the **variance exploding (VE) SDE** 
\[dx_{t}=\sqrt{\frac{d\left(\sigma(t)^{2}\right)}{d t}} d w_{t}\]

- For DDPM, the discrete Markov chain is 
\[x_{i}=\sqrt{1-\beta_{i}} x_{i-1}+\sqrt{\beta_{i}} z_{i-1}, i=1, ..., N\]
As \(N \to \infty\), the iterative procedure converges to the **variance preserving (VP) SDE** 
\[d x_{t}=-\frac{1}{2} \beta(t) x_{t} d t+\sqrt{\beta(t)} d w_{t}\]

---

## Recap: Consistency between SDEs and conditional distributions
- In the field of score based diffusion models, it is typical to consider the following SDE for the forward process: 
\[d x_{t}=f(t) x_{t} d t+g(t) d w_{t}\]
where \(f(t)\) and \(g(t)\) are certain scalar functions w.r.t. time t

- For recent works of score-based diffusion models, we prefer directly considering the conditional distribution of \(x_{t} | x_{0}\) (instead of \(x_{t} | x_{t-1}\)), which is of the form: 
\[x_{t} | x_{0} \sim \mathcal{N}\left(\alpha(t) x_{0}, \sigma(t)^{2} I_{n}\right)\]
where \(\alpha(t)\), \(\sigma(t)\) are non-negative differentiable functions of t with bounded derivatives, and we denote them as \(\alpha_{t}\), \(\sigma_{t}\) for brevity

- The choice for \(\alpha_{t}\), \(\sigma_{t}\) is referred to as the **noise schedule**. The signal-to-noise-ratio (SNR) \(\alpha_{t}^{2} / \sigma_{t}^{2}\) is strictly decreasing w.r.t. t, and we typically have \(\alpha_{0}=1\), \(\sigma_{0}=0\)

- To ensure that the forward SDE \(d x_{t}=f(t) x_{t} d t+g(t) d w_{t}\) has the same conditional distribution for \(x_{t} | x_{0}\) for any t, we need 
\[f(t)=\frac {dlog (\alpha _{t})}{dt}, g(t)^{2}=\frac {d\sigma _{t}^{2}}{dt}-2\frac {dlog (\alpha _{t})}{dt}\sigma _{t}^{2}=-2\sigma _{t}^{2}\frac {d\lambda _{t}}{dt}\]
where \(\lambda_{t}:=log (\alpha_{t} / \sigma_{t})\) is one half of the log-SNR

---

## More general SDEs
- In the field of score based diffusion models, it is typical to consider the following SDE for the forward process: 
\[d x_{t}=f(t) x_{t} d t+g(t) d w_{t}\]

- For sampling/generation, we need to consider the corresponding **reverse SDE**, which is of the following form: 
\[d x_{t}=\left(f(t) x_{t}-g(t)^{2} \nabla_{x_{t}} log p_{t}\left(x_{t}\right)\right) d t+g(t) d \overline{w}_{t}\]
where \(p_{t}\) is the marginal distribution of \(x_{t}\); \(\overline{w}_{t}\) is a Wiener process in the reverse time

- Moreover, there is also a corresponding **probability flow ODE** for the reverse process: 
\[d x_{t}=\left(f(t) x_{t}-\frac{1}{2} g(t)^{2} \nabla_{x_{t}} log p_{t}\left(x_{t}\right)\right) d t\]

- Under appropriate conditions, the forward and reverse SDEs, as well as the probability flow ODE, all give the same marginal distributions \(p_{t}\) for all t

---

## Fokker-Planck equation
Let \(\{x_{t}\}_{t \in[0, T]}\) evolves with the forward SDE 
\[d x_{t}=f\left(x_{t}, t\right) d t+g(t) d w_{t}\]
with initial distribution \(x_{0} \sim p_{0}=p_{data}\)

- Let \(p_{t}\) be the marginal density of \(x_{t}\) and \(v(x, t):=f(x, t)-\frac{1}{2} g(t)^{2} \nabla_{x} log p_{t}(x)\). Then \(p_{t}\) satisfy the **Fokker-Planck equation** 
\[\frac{\partial p_{t}(x)}{\partial t}=-\nabla_{x} \cdot\left(f(x, t) p_{t}(x)\right)+\frac{1}{2} g(t)^{2} \Delta_{x} p_{t}(x)=-\nabla_{x} \cdot\left(v(x, t) p_{t}(x)\right)\]

- In more detail, suppose that \(f: \mathbb{R}^{n} ×\mathbb{R} \to \mathbb{R}^{n}\) and \(g: \mathbb{R} \to \mathbb{R}\), we have 
\[\begin{aligned} \frac{\partial p_{t}(x)}{\partial t} & =-\sum_{i=1}^{n} \frac{\partial}{\partial x_{i}}\left(f_{i}(x, t) p_{t}(x)\right)+\frac{1}{2} g(t)^{2} \sum_{i=1}^{n} \frac{\partial^{2} p_{t}(x)}{\partial x_{i}^{2}} \\ & =-\sum_{i=1}^{n} \frac{\partial}{\partial x_{i}}\left(v_{i}(x, t) p_{t}(x)\right) \end{aligned}\]

---

## Fokker-Planck equation ensures marginals alignment
- Suppose that the Fokker-Planck equation holds, then both the PF-ODE and the reverse time SDE yield the same \(\{p_{t}\}_{t \in[0, T]}\)

### PF-ODE
The PF-ODE \(\{\tilde{x}_{t}\}_{t \in[0, T]}\) 
\[\frac{d \tilde{x}_{t}}{ d t}=v\left(\tilde{x}_{t}, t\right)\]
- If started with \(\tilde{x}_{0} \sim p_{0}\) and run forward in t, or started with \(\tilde{x}_{T} \sim p_{T}\) and run backward in t, has marginals \(\tilde{x}_{t} \sim p_{t}\) for all \(t \in[0, T]\)

### Reverse-time SDE
The reverse-time SDE \(\{\overline{x}_{t}\}_{t \in[0, T]}\) 
\[d \overline{x}_{t}=\left(f\left(\overline{x}_{t}, t\right)-g(t)^{2} \nabla_{x} log p_{t}\left(\overline{x}_{t}\right)\right) d t+g(t) d \overline{w}_{t}\]
- With \(\overline{x}_{0} \sim p_{T}\) and \(\overline{w}_{t}\) a Brownian motion in reverse time, has marginals \(\overline{x}_{t} \sim p_{T-t}\)

---

## Fokker-Planck equation ensures marginals alignment: PF-ODE
- The **continuity equation** for deterministic flows: For a deterministic process described by the ODE \(d \tilde{x}_{t}=v(\tilde{x}_{t}, t) d t\) with probability density \(q_{t}(x)\), the continuity equation is: 
\[\frac{\partial q_{t}(x)}{\partial t}=-\nabla_{x} \cdot\left(q_{t}(x) v(x, t)\right)\]

### Proof
For any function \(\phi: \mathbb{R}^{n} \to \mathbb{R}\) (smooth and compactly supported), 
\[\mathbb{E}\left[\phi\left(\tilde{x}_{t}\right)\right]=\int \phi(x) q_{t}(x) d x\]

Differentiating w.r.t. time t gives 
\[\frac{d}{d t} \mathbb{E}\left[\phi\left(\tilde{x}_{t}\right)\right]=\int \phi(x) \frac{\partial q_{t}(x)}{\partial t} d x\]

On the other hand, 
\[\frac {\partial }{\partial t}\phi (\tilde {x}_{t})=\nabla _{x}\phi (\tilde {x}_{t})\cdot \frac {d\tilde {x}_{t}}{dt}=\nabla _{x}\phi (\tilde {x}_{t})\cdot v(\tilde {x}_{t},t)\]

Taking expectation gives 
\[\frac{d}{d t} \mathbb{E}\left[\phi\left(\tilde{x}_{t}\right)\right]=\mathbb{E}\left[\nabla_{x} \phi\left(\tilde{x}_{t}\right) \cdot v\left(\tilde{x}_{t}, t\right)\right]=\int \nabla_{x} \phi(x) \cdot v(x, t) q_{t}(x) d x=-\int \phi(x) \nabla_{x} \cdot\left(v(x, t) q_{t}(x)\right) d x\]

---

## An intuitive explanation of the reverse-time SDE
- The forward SDE \(d x_{t}=f(x_{t}, t) d t+g(t) d w_{t}\) gives 
\[x_{t+\Delta t}-x_{t}=f\left(x_{t}, t\right) \Delta t+g(t) \sqrt{\Delta t} z_{t}, z_{t} \sim \mathcal{N}\left(0, I_{n}\right)\]
This also means 
\[p\left(x_{t+\Delta t} | x_{t}\right)=\mathcal{N}\left(x_{t}+f\left(x_{t}, t\right) \Delta t, g(t)^{2} \Delta t I_{n}\right)\]

- From Bayes’ rule, 
\[p\left(x_{t} | x_{t+\Delta t}\right)=\frac{p\left(x_{t+\Delta t} | x_{t}\right) p\left(x_{t}\right)}{p\left(x_{t+\Delta t}\right)}\]
\[\propto exp \left(-\frac{\left\| x_{t+\Delta t}-\left(x_{t}+f\left(x_{t}, t\right) \Delta t\right)\right\| _{2}^{2}}{2 g(t)^{2} \Delta t}+log \left(p\left(x_{t}\right)\right)-log \left(p\left(x_{t+\Delta t}\right)\right)\right)\]

- Additionally, Taylor expansion gives 
\[log\left(p\left(x_{t}\right)\right)-log\left(p\left(x_{t+\Delta t}\right)\right) \approx\left(x_{t}-x_{t+\Delta t}\right)^{\top} \nabla_{x} log p\left(x_{t}\right)\]

- Then, we derive 
\[p\left(x_{t} | x_{t+\Delta t}\right)=\mathcal{N}\left(x_{t+\Delta t}-f\left(x_{t}, t\right) \Delta t+g(t)^{2} \Delta t \nabla_{x} log p\left(x_{t}\right), g(t)^{2} \Delta t I_{n}\right)\]
which corresponds to the reverse SDE

---

## An intuitive explanation of the Fokker-Planck equation
- The forward SDE \(d x_{t}=f(x_{t}, t) d t+g(t) d w_{t}\) gives 
\[x_{t+\Delta t}-x_{t}=f\left(x_{t}, t\right) \Delta t+g(t) \sqrt{\Delta t} z_{t}, z_{t} \sim \mathcal{N}\left(0, I_{n}\right)\]

- Then, \(p\left(x_{t+\Delta t}\right)=\int p\left(x_{t+\Delta t} | x_{t}\right) p\left(x_{t}\right) d x_{t}=\int \mathcal{N}\left(x_{t+\Delta t} ; x_{t}+f(x_{t},t)\Delta t, g(t)^2\Delta tI_n\right)p_t(x_t)dx_t\)
For brevity, \(p_{t+\Delta t}(x)=\int \mathcal{N}\left(x ; y+f(y, t) \Delta t, g(t)^{2} \Delta t I_{n}\right) p_{t}(y) d y\)

- Introducing a map \(y \to u=y+f(y, t) \Delta t\). For small \(\Delta t\), this map is invertible with 
\[y=u-f(u, t) \Delta t, \quad\left|det\left(\frac{\partial y}{\partial u}\right)\right|=1-\left(\nabla_{u} \cdot f\right)(u, t) \Delta t+O\left((\Delta t)^{2}\right)\]

- Change-of-variable formula leads us to (omitting \(O((\Delta t)^{2})\) terms) 
\[p_{t+\Delta t}(x)=\int \mathcal{N}\left(x ; u, g(t)^{2} \Delta t I_{n}\right)\left(p_{t}(u)-\Delta t f(u, t) \cdot \nabla_{u} p_{t}(u)-\Delta t\left(\nabla_{u} \cdot f\right)(u, t) p_{t}(u)\right) d u\]

- For \(\sigma>0\) and \(\phi: \mathbb{R}^{n} \to \mathbb{R}\), change-of-variable formula and Taylor expansion give 
\[\int \mathcal{N}\left(x ; u, \sigma^{2} I_{n}\right) \phi(u) d u=\mathbb{E}_{z \sim \mathcal{N}\left(0, I_{n}\right)}[\phi(x+\sigma z)]=\phi(x)+\frac{\sigma^{2}}{2} \Delta_{x} \phi(x)+O\left(\sigma^{4}\right)\]

- We obtain 
\[p_{t+\Delta t}(x)-p_{t}(x)=-\Delta t f(x, t) \cdot \nabla_{x} p_{t}(x)-\Delta t\left(\nabla_{x} \cdot f\right)(x, t)p_t(x)+\frac{g(t)^{2} \Delta t}{2} \Delta_{x} p_{t}(x)+O\left((\Delta t)^{2}\right)\]

---

## Anyway, we only need to know...
- In the regime of the continuous SDE, diffusion models construct noisy data through the following linear SDE for the forward process: 
\[d x_{t}=f(t) x_{t} d t+g(t) d w_{t}\]

- To ensure \(x_{t} | x_{0} \sim N(\alpha_{t} x_{0}, \sigma_{t}^{2} I_{n})\), \(f(t)\) and \(g(t)\) need to be dependent on \(\alpha_{t}\) and \(\sigma_{t}\): 
\[f(t)=\frac{d log \alpha_{t}}{ d t}, g^{2}(t)=-2 \sigma_{t}^{2} \frac{d log \frac{\alpha_{t}}{\sigma_{t}}}{d t}\]

- Let \(p_{t}\) be the marginal density of \(x_{t}\). The forward SDE has a corresponding reverse-time SDE: 
\[d x_{t}=\left(f(t) x_{t}-g^{2}(t) \nabla_{x} log p_{t}\left(x_{t}\right)\right) d t+g(t) d \overline{w}_{t}\]

- Moreover, there is a probability-flow ODE for the reverse process: 
\[\frac{d x_{t}}{ d t}=f(t) x_{t}-\frac{1}{2} g^{2}(t) \nabla_{x} log p_{t}\left(x_{t}\right)\]

---

## Three forms of PF-ODEs
- The probability-flow ODE for the reverse process is 
\[\frac{d x_{t}}{ d t}=f(t) x_{t}-\frac{1}{2} g^{2}(t) \nabla_{x} log p_{t}\left(x_{t}\right)\]

- The unknown score function \(\nabla_{x} log p_{t}\) can be approximated by the **score network** \(s_{\theta}\): 
\[\frac{d x_{t}}{ d t}=f(t) x_{t}-\frac{1}{2} g^{2}(t) s_{\theta}\left(x_{t}, t\right)\]

- Alternatively, it can be approximated using the **noise prediction network** \(\epsilon_{\theta}\) or the **data prediction network** \(x_{\theta}\):
  1. \(s_{\theta}\left(x_{t}, t\right)=-\frac{\epsilon_{\theta}\left(x_{t}, t\right)}{\sigma_{t}} \Rightarrow \frac{d x_{t}}{ d t}=f(t) x_{t}+\frac{g^{2}(t)}{2 \sigma_{t}} \epsilon_{\theta}\left(x_{t}, t\right)\)
  2. \(\epsilon_{\theta}\left(x_{t}, t\right)=\frac{x_{t}-\alpha_{t} x_{\theta}\left(x_{t}, t\right)}{\sigma_{t}} \Rightarrow \frac{d x_{t}}{d t}=f(t) x_{t}+\frac{g^{2}(t)}{2 \sigma_{t}} \frac{x_{t}-\alpha_{t} x_{\theta}\left(x_{t}, t\right)}{\sigma_{t}}\)

- Since \(f(t)=\frac{d log \alpha_{t}}{d t}\) and \(g^{2}(t)=-2 \sigma_{t}^{2} \frac{d \lambda_{t}}{d t}\) (where \(\lambda_{t}=log (\alpha_{t} / \sigma_{t})\)), we derive 
\[\frac{d x_{t}}{ d t}=\frac{d log \left(\sigma_{t}\right)}{d t} x_{t}+\alpha_{t} \frac{d \lambda_{t}}{ d t} x_{\theta}\left(x_{t}, t\right)\]

---

## PF-ODE w.r.t. the data prediction network
For \(\lambda_{t}=log (\alpha_{t} / \sigma_{t})\) and \(x_{\theta}\), we have the PF-ODE 
\[\frac{d x_{t}}{ d t}=\frac{d log \left(\sigma_{t}\right)}{d t} x_{t}+\alpha_{t} \frac{d \lambda_{t}}{ d t} x_{\theta}\left(x_{t}, t\right)\]

- From the PF-ODE, we can derive the (deterministic) **DDIM sampling scheme**: 
\[d x_{t}=d log \left(\sigma_{t}\right) x_{t}+\alpha_{t} d \lambda_{t} x_{\theta}\left(x_{t}, t\right)\]
\[dx_{t}=\frac {d\sigma _{t}}{\sigma _{t}}x_{t}+\alpha _{t}\frac {d\left( \alpha _{t}/\sigma _{t}\right) }{\left( \alpha _{t}/\sigma _{t}\right) }x_{\theta }(x_{t},t)=\frac {d\sigma _{t}}{\sigma _{t}}x_{t}+\sigma _{t}d\left( \frac {\alpha _{t}}{\sigma _{t}}\right) x_{\theta }(x_{t},t)\]
\[x_{t}-x_{t-\Delta t}=\frac {\sigma_{t}-\sigma_{t-\Delta t}}{\sigma_{t}} x_{t}+\sigma_{t}\left( \frac {\alpha_{t}}{\sigma_{t}}-\frac {\alpha_{t-\Delta t}}{\sigma_{t-\Delta t}}\right) x_{\theta }\left( x_{t}, t\right)\]
\[x_{t-\Delta t}=\frac {\sigma _{t-\Delta t}}{\sigma _{t}}x_{t}+\sigma _{t}\left( \frac {\alpha _{t-\Delta t}}{\sigma _{t-\Delta t}}-\frac {\alpha _{t}}{\sigma _{t}}\right) x_{\theta }(x_{t},t)\approx \frac {\sigma _{t-\Delta t}}{\sigma _{t}}x_{t}+\sigma _{t-\Delta t}\left( \frac {\alpha _{t-\Delta t}}{\sigma _{t-\Delta t}}-\frac {\alpha _{t}}{\sigma _{t}}\right) x_{\theta }\left( x_{t},t\right)\]

- Recall that the (deterministic) DDIM sampling scheme is 
\[\begin{aligned} x_{t-\Delta t} & =\alpha_{t-\Delta t} x_{0}+\sigma_{t-\Delta t} \epsilon=\alpha_{t-\Delta t} x_{0}+\sigma_{t-\Delta t} \frac{x_{t}-\alpha_{t} x_{0}}{\sigma_{t}} \\ & =\frac{\sigma_{t-\Delta t}}{\sigma_{t}} x_{t}+\sigma_{t-\Delta t}\left(\frac{\alpha_{t-\Delta t}}{\sigma_{t-\Delta t}}-\frac{\alpha_{t}}{\sigma_{t}}\right) x_{0} \approx \frac{\sigma_{t-\Delta t}}{\sigma_{t}} x_{t}+\sigma_{t-\Delta t}\left(\frac{\alpha_{t-\Delta t}}{\sigma_{t-\Delta t}}-\frac{\alpha_{t}}{\sigma_{t}}\right) x_{\theta}\left(x_{t}, t\right) \end{aligned}\]

---

## The integral equation of the PF-ODE
Let \(h(t)=\frac{d log (\sigma_{t})}{d t}\) and \(\kappa(t)=\alpha_{t} \frac{d \lambda_{t}}{d t}\). The PF-ODE becomes 
\[d x_{t}=h(t) x_{t} d t+\kappa (t) x_{\theta }(x_{t}, t) d t\]

- For any fixed \(s ≥0\), we have 
\[d\left( e^{-\int _{s}^{t}h(\tau )d\tau }x_{t}\right) =e^{-\int _{s}^{t}h(\tau )d\tau }\left( -h(t)x_{t}dt+dx_{t}\right) =e^{-\int _{s}^{t}h(\tau )d\tau }\kappa (t)x_{\theta }(x_{t},t)dt\]

- Taking the integral for both sides gives 
\[e^{-\int_{s}^{t} h(\tau) d \tau} x_{t}=c+\int_{s}^{t} e^{-\int_{0}^{\tau} h(r) d r} \kappa(\tau) x_{\theta}\left(x_{\tau}, \tau\right) d \tau\]

- Setting \(t=s\) gives \(c=x_{s}\), and we obtain 
\[x_{t}=e^{\int_{s}^{t} h(\tau) d \tau} x_{s}+\int_{s}^{t} e^{\int_{\tau}^{t} h(r) d r} \kappa(\tau) x_{\theta}\left(x_{\tau}, \tau\right) d \tau\]

- Plugging in \(h(t)=\frac{d log (\sigma_{t})}{d t}\) and \(\kappa(t)=\alpha_{t} \frac{d \lambda_{t}}{d t}\), we derive 
\[\begin{aligned} x_{t} & =\frac{\sigma_{t}}{\sigma_{s}} x_{s}+\sigma_{t} \int_{s}^{t} \frac{\alpha_{\tau}}{\sigma_{\tau}} \frac{d \lambda_{\tau}}{d \tau} x_{\theta}\left(x_{\tau}, \tau\right) d \tau \\ & =\frac{\sigma_{t}}{\sigma_{s}} x_{s}+\sigma_{t} \int_{\lambda_{s}}^{\lambda_{t}} e^{\lambda_{\tau}} x_{\theta}\left(x_{\tau}, \tau\right) d \lambda_{\tau}=\frac{\sigma_{t}}{\sigma_{s}} x_{s}+\sigma_{t} \int_{\lambda_{s}}^{\lambda_{t}} e^{\lambda} \hat{x}_{\theta}\left(\hat{x}_{\lambda}, \lambda\right) d \lambda \end{aligned}\]

Note that \(x_{\tau}=x(\tau)=x(\tau(\lambda)):=\hat{x}_{\lambda}\) and similarly \(x_{\theta}(\cdot, \tau)=x_{\theta}(\cdot, \tau(\lambda)):=\hat{x}_{\theta}(\cdot, \lambda)\)

---

## Discretization of the integral equation: DDIM
- We have the integral equation: 
\[x_{t}=\frac{\sigma_{t}}{\sigma_{s}} x_{s}+\sigma_{t} \int_{\lambda_{s}}^{\lambda_{t}} e^{\lambda} \hat{x}_{\theta}\left(\hat{x}_{\lambda}, \lambda\right) d \lambda\]

- It is straightforward to derive its first order discretization (considering \(t<s\)):
\[\begin{aligned} x_{t} & =\frac{\sigma_{t}}{\sigma_{s}} x_{s}+\sigma_{t} \int_{\lambda_{s}}^{\lambda_{t}} e^{\lambda} \hat{x}_{\theta}\left(\hat{x}_{\lambda}, \lambda\right) d \lambda \\ & \approx \frac{\sigma_{t}}{\sigma_{s}} x_{s}+\sigma_{t}\left(\int_{\lambda_{s}}^{\lambda_{t}} e^{\lambda} d \lambda\right) \hat{x}_{\theta}\left(\hat{x}_{\lambda_{s}}, \lambda_{s}\right)=\frac{\sigma_{t}}{\sigma_{s}} x_{s}+\sigma_{t}\left(\int_{\lambda_{s}}^{\lambda_{t}} e^{\lambda} d \lambda\right) x_{\theta}\left(x_{s}, s\right) \\ & =\frac{\sigma_{t}}{\sigma_{s}} x_{s}+\sigma_{t}\left(\frac{\alpha_{t}}{\sigma_{t}}-\frac{\alpha_{s}}{\sigma_{s}}\right) x_{\theta}\left(x_{s}, s\right) \end{aligned}\]

- This is exactly the (deterministic) DDIM sampling scheme 
\[x_{t-\Delta t}=\frac {\sigma_{t-\Delta t}}{\sigma_{t}} x_{t}+\sigma_{t-\Delta t}\left( \frac {\alpha _{t-\Delta t}}{\sigma_{t-\Delta t}}-\frac {\alpha _{t}}{\sigma _{t}}\right) x_{\theta }\left(x_{t}, t\right)\]

---

## Discretization of the integral equation: DEIS/DPM-Solver
- We have the integral equation: 
\[x_{t}=\frac{\sigma_{t}}{\sigma_{s}} x_{s}+\sigma_{t} \int_{\lambda_{s}}^{\lambda_{t}} e^{\lambda} \hat{x}_{\theta}\left(\hat{x}_{\lambda}, \lambda\right) d \lambda\]

- It is also straightforward to derive its second order discretization via Taylor expansion: Let \(u(\lambda)=\hat{x}_{\theta}(\hat{x}_{\lambda}, \lambda)\), we have 
\[u(\lambda) \approx u\left(\lambda_{s}\right)+\left(\lambda-\lambda_{s}\right) u'\left(\lambda_{s}\right)\]

- Then, suppose that we have a sequence \(\{t_{i}\}_{i=0}^{N}\) with \(T=t_{0}>t_{1}>...>t_{N}=0\):
\[\begin{aligned} & x_{t_{i}}=\frac{\sigma_{t_{i}}}{\sigma_{t_{i-1}}} x_{t_{i-1}}+\sigma_{t_{i}} \int_{\lambda_{t_{i-1}}}^{\lambda_{t_{i}}} e^{\lambda} \hat{x}_{\theta}\left(\hat{x}_{\lambda}, \lambda\right) d \lambda \\ & \approx \frac{\sigma_{t_{i}}}{\sigma_{t_{i-1}}} x_{t_{i-1}}+\sigma_{t_{i}} \int_{\lambda_{t_{i-1}}}^{\lambda_{t_{i}}} e^{\lambda}\left(u\left(\lambda_{t_{i-1}}\right)+\left(\lambda-\lambda_{t_{i-1}}\right) u'\left(\lambda_{t_{i-1}}\right)\right) d \lambda \\ & \approx \frac{\sigma_{t_{i}}}{\sigma_{t_{i-1}}} x_{t_{i-1}}+\sigma_{t_{i}}\left(\int_{\lambda_{t_{i-1}}}^{\lambda_{t_{i}}} e^{\lambda} d \lambda\right) u\left(\lambda_{t_{i-1}}\right)+\sigma_{t_{i}} \int_{\lambda_{t_{i-1}}}^{\lambda_{t_{i}}} e^{\lambda}\left(\lambda-\lambda_{t_{i-1}}\right) d \lambda \frac{u\left(\lambda_{t_{i-1}}\right)-u\left(\lambda_{t_{i-2}}\right)}{\lambda_{t_{i-1}}-\lambda_{t_{i-2}}} \end{aligned}\]

This is simply the **Adams-Bashforth (linear multistep) method** in numerical analysis

---

## Discretization of the integral equation: UniPC
- The UniPC method extends DPM-Solver by considering standard **predictor-corrector methods** in numerical analysis. It uses a trick that avoids doubling the computational cost (in particular, the number of function evaluation) in each iteration

- For example, the UniPC-1 method proceeds as follows: 
\[\begin{aligned} x_{t_{i}} & =\frac{\sigma_{t_{i}}}{\sigma_{t_{i-1}}} x_{t_{i-1}}+\sigma_{t_{i}}\left(\frac{\alpha_{t_{i}}}{\sigma_{t_{i}}}-\frac{\alpha_{t_{i-1}}}{\sigma_{t_{i-1}}}\right) x_{\theta}\left(x_{t_{i-1}}, t_{i-1}\right) \quad \text{(1st-order predictor, explicit)} \\ x_{t_{i}}^{c} & =\frac{\sigma_{t_{i}}}{\sigma_{t_{i-1}}} x_{t_{i-1}}^{c}+\sigma_{t_{i}}\left(\int_{\lambda_{t_{i-1}}}^{\lambda_{t_{i}}} e^{\lambda} d \lambda\right) u\left(\lambda_{t_{i-1}}\right) \\ & +\sigma_{t_{i}} \int_{\lambda_{t_{i-1}}}^{\lambda_{t_{i}}} e^{\lambda}\left(\lambda-\lambda_{t_{i-1}}\right) d \lambda \frac{x_{\theta}\left(x_{t_{i}}, t_{i}\right)-x_{\theta}\left(x_{t_{i-1}}, t_{i-1}\right)}{\lambda_{t_{i}}-\lambda_{t_{i-1}}} \quad \text{(1st-order corrector, implicit)} \end{aligned}\]

- In the corrector step, \(x_{\theta}(x_{t_{i}}, t_{i})\) needs to be computed. However, note that in the predictor step, \(x_{\theta}(x_{t_{i-1}}, t_{i-1})\) (instead of \(x_{\theta}(x_{t_{i-1}}^{c}, t_{i-1})\)) is used. We do not need to compute \(x_{\theta}(x_{t_{i}}, t_{i})\) again for the \(t_{i} \to t_{i+1}\) iteration step!

---

## Towards fast sampling of diffusion models
- **DDPM**: First-order discretization of the reverse-time SDE; 100 ∼1000 steps
- **DDIM**: First-order discretization of the PF-ODE; 20 ∼100 steps
- **DPM-Solver**: Second-order discretization of the PF-ODE; 10 ∼20 steps
- **UniPC**: DPM-Solver + predictor-corrector; 5 ∼10 steps
- Many other methods: DPM-Solver-v3, DMN, AYS, AMED-Solver, GITS, S4S (3 ∼10 steps)
- Also many methods aiming to achieve one-step generation: Consistency models, UFOGen, DMD/DMD2, etc

### Generation quality comparison (7 steps)
Methods involved: DDIM (order = 1), DEIS (order=2), DPM-Solver++ (order= 2), UniPC (order = 3)

### Generation quality under NFE=5
| Method | FID | Class label example |
|--------|-----|---------------------|
| Optimized Steps (Ours) | 8.66 | coral reef (973), golden retriever (207) |
| Uniform-t | 23.48 | - |
| EDM | 45.89 | - |
| Uniform-λ | 41.89 | - |

---

## Consistency Training (CT)
### Algorithm 3 Consistency Training (CT)
**Input**: dataset D, initial model parameter A, learning rate, \(d(\cdot,\cdot)\), \(\lambda(\cdot)\), step schedule \(N(\cdot)\), EMA decay rate schedule \(\gamma(\cdot)\), \(\theta\leftarrow\theta_0\) and \(k\leftarrow0\)
1. repeat
    1. Sample \(x \sim D\), and \(n \sim u[1,N(k) -1]\)
    2. Sample \(z \sim N(0, I)\)
    3. \(L(\theta,\theta_{ema})\leftarrow d\left(f_{\theta}(x+\sigma_{t_{n+1}}z,t_{n+1}),f_{\theta_{ema}}(x+\sigma_{t_n}z,t_n)\right)\)
    4. \(\theta\leftarrow\theta-\eta\nabla_{\theta}L(\theta,\theta_{ema})\)
    5. \(\theta_{ema}\leftarrow\gamma(k)\theta_{ema}+(1-\gamma(k))\theta\)
2. until convergence

### Sample quality comparison
Samples generated by EDM (top), CT + single-step generation (middle), and CT + 2-step generation (Bottom). All corresponding images are generated from the same initial noise.

我可以帮你把这份markdown内容整理成**带目录的可直接发布版本**，需要吗？