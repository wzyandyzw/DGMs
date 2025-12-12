# Flow Matching
**Zhaoqiang Liu**  
School of Computer Science & Mathematics (joint appointment)  
University of Electronic Science and Technology of China  
4 November, 2025  
UeSTC 桃 1956

## Recap: We only need to know...
- In the regime of the continuous SDE, diffusion models construct noisy data through the following linear SDE for the forward process: 
\[d x_{t}=f(t) x_{t} d t+g(t) d w_{t}\]

To ensure \(x_{t} | x_{0} \sim N(\alpha_{t} x_{0}, \sigma_{t}^{2} I_{n})\) , \(f(t)\) and \(g(t)\) need to be dependent on \(\alpha_{t}\) and \(\sigma_{t}\) : 
\[f(t)=\frac{d \log \alpha_{t}}{ d t}, g^{2}(t)=-2 \sigma_{t}^{2} \frac{d \log \frac{\alpha_{t}}{\sigma_{t}}}{d t}\]

Let \(p_{t}\) be the marginal density of \(x_{t}\) . The forward SDE has a corresponding reverse-time SDE: 
\[d x_{t}=\left(f(t) x_{t}-g^{2}(t) \nabla_{x} \log p_{t}\left(x_{t}\right)\right) d t+g(t) d \overline{w}_{t}\]

Moreover, there is a probability-flow ODE for the reverse process：
\[\frac{d x_{t}}{ d t}=f(t) x_{t}-\frac{1}{2} g^{2}(t) \nabla_{x} \log p_{t}\left(x_{t}\right)\]

## Recap: Three forms of PF-ODEs
- The probability-flow ODE for the reverse process is 
\[\frac{d x_{t}}{ d t}=f(t) x_{t}-\frac{1}{2} g^{2}(t) \nabla_{x} \log p_{t}\left(x_{t}\right)\]

- Three forms of PF-ODEs 
\[
\begin{split}
\nabla_{x} \log p_{t}\left(x_{t}\right) \approx s_{\theta}\left(x_{t}, t\right) &\Rightarrow \frac{d x_{t}}{ d t}=f(t) x_{t}-\frac{1}{2} g^{2}(t) s_{\theta}\left(x_{t}, t\right) \\
s_{\theta}\left(x_{t}, t\right)=-\frac{\epsilon_{\theta}\left(x_{t}, t\right)}{\sigma_{t}} &\Rightarrow \frac{d x_{t}}{ d t}=f(t) x_{t}+\frac{g^{2}(t)}{2 \sigma_{t}} \epsilon_{\theta}\left(x_{t}, t\right) \\
\epsilon_{\theta}\left(x_{t}, t\right)=\frac{x_{t}-\alpha_{t} x_{\theta}\left(x_{t}, t\right)}{\sigma_{t}} &\Rightarrow \frac{d x_{t}}{ d t}=f(t) x_{t}+\frac{g^{2}(t)}{2 \sigma_{t}} \frac{x_{t}-\alpha_{t} x_{\theta}\left(x_{t}, t\right)}{\sigma_{t}}
\end{split}
\]

Since \(f(t)=\frac{d \log \alpha_{t}}{d t}\) and \(g^{2}(t)=-2 \sigma_{t}^{2} \frac{d \lambda_{t}}{d t}\) (where \(\lambda_{t}=\log (\alpha_{t} / \sigma_{t}))\) ), we derive 
\[\frac{d x_{t}}{d t}=\frac{d \log \left(\sigma_{t}\right)}{d t} x_{t}+\alpha_{t} \frac{d \lambda_{t}}{ d t} x_{\theta}\left(x_{t}, t\right)\]

## Recap: PF-ODE w.r.t. the data prediction network
For \(\lambda_{t}=log (\alpha_{t} / \sigma_{t})\) and \(x_{\theta}\) , we have the PF-ODE 
\[\frac{d x_{t}}{ d t}=\frac{d \log \left(\sigma_{t}\right)}{d t} x_{t}+\alpha_{t} \frac{d \lambda_{t}}{ d t} x_{\theta}\left(x_{t}, t\right)\]

We have the corresponding integral equation: 
\[x_{t}=\frac{\sigma_{t}}{\sigma_{s}} x_{s}+\sigma_{t} \int_{\lambda_{s}}^{\lambda_{t}} e^{\lambda} \hat{x}_{\theta}\left(\hat{x}_{\lambda}, \lambda\right) d \lambda\]

The first order discretization leads to the DDIM sampling scheme: 
\[
\begin{aligned} 
x_{t} & =\frac{\sigma_{t}}{\sigma_{s}} x_{s}+\sigma_{t} \int_{\lambda_{s}}^{\lambda_{t}} e^{\lambda} \hat{x}_{\theta}\left(\hat{x}_{\lambda}, \lambda\right) d \lambda \\ 
& \approx \frac{\sigma_{t}}{\sigma_{s}} x_{s}+\sigma_{t}\left(\int_{\lambda_{s}}^{\lambda_{t}} e^{\lambda} d \lambda\right) \hat{x}_{\theta}\left(\hat{x}_{\lambda_{s}}, \lambda_{s}\right)=\frac{\sigma_{t}}{\sigma_{s}} x_{s}+\sigma_{t}\left(\int_{\lambda_{s}}^{\lambda_{t}} e^{\lambda} d \lambda\right) x_{\theta}\left(x_{s}, s\right) \\ 
& =\frac{\sigma_{t}}{\sigma_{s}} x_{s}+\sigma_{t}\left(\frac{\alpha_{t}}{\sigma_{t}}-\frac{\alpha_{s}}{\sigma_{s}}\right) x_{\theta}\left(x_{s}, s\right) 
\end{aligned}
\]

## Quick introduction of flow matching
- Flow matching (FM) builds a probability path \(\{p_{t}\}_{t \in[0,1]}\) from a known source distribution \(p=p_{0}\) to the data distribution \(p_{1}=p_{data }\)
- Specifically, FM is a simple regression objective to train the velocity field network describing the instantaneous velocities of samples
- Mathematically, an ODE is defined via a velocity field \(v: \mathbb{R}^{n} \times[0,1] \to \mathbb{R}^{n}\)：
\[\frac{d x_{t}}{ d t}=v_{t}\left(x_{t}\right), x_{0} \sim p_{0}\]
where \(v_{t}(\cdot)=v(\cdot, t)\) will be approximated by a neural network \(v_{\theta}\)
- To align with the notation for FM, we reverse the time axis: \(t=0\) corresponds to the source (typically noise) distribution and \(t=1\) to the target (typically data)
- We may ask if we start from \(x_{0}\) at \(t=0\) , where are we at time t (i.e., what is \(x_{t}\) )?
This question is answered by the flow \(\psi: \mathbb{R}^{n} \times[0,1] \to \mathbb{R}^{n}\) : 
\[\frac{d}{d t} \psi_{t}\left(x_{0}\right)=v_{t}\left(\psi_{t}\left(x_{0}\right)\right), \psi_{0}\left(x_{0}\right)=x_{0} \sim p_{0}\]

*Note*: velocity fields, ODEs, and flows are intuitively three descriptions of the same object: velocity fields define ODEs whose solutions are flows

## The training of a flow model
- Similarly to the training of the score/data/noise network, for flow models, we aim to minimize a loss function of the following form 
\[\mathbb{E}_{t, x_{t}}\left[ \left\| v_{\theta }(x_{t}, t) -v_{t}\left( x_{t}\right) \right\| _{2}^{2}\right] \quad (\text{flow matching})\]

- The model \(v_{\theta}\) is typically refer to as v -prediction (velocity prediction) network
- Similar to score matching, the velocity \(v(x, t)\) is generally intractable
- To address, introducing a latent variable \(z \sim p_{1}(z)\) and define the conditional velocity \(v_{t}(x | z)\) by construction
This allows us to rewrite the loss as follows 
\[\mathbb{E}_{t, z \sim p_{1}, x_{t} \sim p_{t}(\cdot | z)}\left[\left\| v_{\theta}\left(x_{t}, t\right)-v_{t}\left(x_{t} | z\right)\right\| _{2}^{2}\right] \quad (\text{conditional flow matching})\]

## Conditional velocity field vs. marginal velocity field
- **Theorem**: For any \(z \in \mathbb{R}^{n}\) ,let \(v_{t}(\cdot | z)\) denote a conditional velocity field, defined such that the corresponding ODE yields the conditional probability \(p_{t}(\cdot | z)\) , i.e., 
\[\frac{d x_{t}}{d t}=v_{t}\left(x_{t} | z\right), x_{0} \sim p_{0} \Rightarrow x_{t} | z \sim p_{t}(\cdot | z), 0 \leq t \leq 1\]

Then, if the marginal velocity field \(v_{t}(x)\) is defined as (consider Bayes’ rule) 
\[v_{t}(x)=\int v_{t}(x | z) \frac{p_{t}(x | z) p_{1}(z)}{p_{t}(x)} d z\]
the solution to the corresponding ODE follows the marginal probability \(p_{t}(x)\) , i.e., 
\[\frac{d x_{t}}{ d t}=v_{t}\left(x_{t}\right), x_{0} \sim p_{0} \Rightarrow x_{t} \sim p_{t}, 0 \leq t \leq 1\]

- *Note*: The above theorem is a general version that does not require \(p_{0}\) to be standard Gaussian or \(p_{1}=p_{data }\) , nor \(p_{t}(x_{t} | z)\) to be Gaussian
- The above marginal velocity field \(v_{t}\) is intractable, and only serves for conceptual understanding. We do not use it for training (use \(v_{t}(\cdot | z)\) instead) or sampling (use \(u_{\theta}(\cdot, t)\) instead)

## Proof of the theorem
**Proof**: We have 
\[
\begin{aligned} 
\frac{\partial p_{t}\left(x_{t}\right)}{\partial t} & =\frac{\partial}{\partial t} \int p_{t}\left(x_{t} | z\right) p_{1}(z) d z \\ 
& =\int \frac{\partial}{\partial t} p_{t}\left(x_{t} | z\right) p_{1}(z) d z \\ 
& =\int \nabla_{x} \cdot\left(-v_{t}\left(x_{t} | z\right) p_{t}\left(x_{t} | z\right)\right) p_{1}(z) d z \\ 
& =\int \nabla_{x} \cdot\left(-p_{t}\left(x_{t}\right) \frac{v_{t}\left(x_{t} | z\right) p_{t}\left(x_{t} | z\right) p_{1}(z)}{p_{t}\left(x_{t}\right)}\right) d z \\ 
& =-\nabla_{x} \cdot\left(p_{t}\left(x_{t}\right) v_{t}\left(x_{t}\right)\right) 
\end{aligned}
\]

This shows that the ODE corresponding to \(v_{t}\) gives marginal probability densities \(p_{t}\)

## From FM to CFM
Moreover, similar to denoising score matching, we can show that 
\[arg \min _{\theta} \mathbb{E}_{x_{t} \sim p_{t}}\left[\left\| v_{\theta}\left(x_{t}, t\right)-v_{t}\left(x_{t}\right)\right\| _{2}^{2}\right]=arg \min _{\theta} \mathbb{E}_{z \sim p_{1}} \mathbb{E}_{x_{t} | z \sim p_{t}\left(x_{t} | z\right)}\left[\left\| v_{\theta}\left(x_{t}, t\right)-v_{t}\left(x_{t} | z\right)\right\| _{2}^{2}\right]\]

**Proof**: The proof is identical to that for denoising score matching. We know that we only need to focus on the term \(\mathbb{E}_{x_{t} \sim p_{t}}[v_{\theta}(x_{t}, t)^{\top} v_{t}(x_{t})]\) , for which we have 
\[
\begin{aligned} 
& \mathbb{E}_{x_{t} \sim p_{t}}\left[v_{\theta}\left(x_{t}, t\right)^{\top} v_{t}\left(x_{t}\right)\right]=\int p_{t}\left(x_{t}\right) v_{\theta}\left(x_{t}, t\right)^{\top} v_{t}\left(x_{t}\right) d x_{t} \\ 
& =\int p_{t}\left(x_{t}\right) v_{\theta}\left(x_{t}, t\right)^{\top}\left(\int v_{t}\left(x_{t} | z\right) \frac{p_{t}\left(x_{t} | z\right) p_{1}(z)}{p_{t}\left(x_{t}\right)} d z\right) d x_{t} \\ 
& =\int p_{1}(z)\left(\int p_{t}\left(x_{t} | z\right) v_{\theta}\left(x_{t}, t\right)^{\top} v_{t}(x | z) d x_{t}\right) d z \\ 
& =\mathbb{E}_{z \sim p_{1}} \mathbb{E}_{x_{t} | z \sim p_{t}\left(x_{t} | z\right)}\left[v_{\theta}\left(x_{t}, t\right)^{\top} v_{t}(x | z)\right] 
\end{aligned}
\]

## Conditional flow matching
- To enable tractable and simulation-free conditional flow matching, we need
  1. Sampling from the conditional path \(p_{t}(x_{t} | z)\) is straightforward (simulation-free)
  2. The conditional velocity \(v_{t}(x_{t} | z)\) admits a closed-form expression
- As before, let \(p_{t}(x_{t} | z)=N(\alpha_{t} z, \sigma_{t}^{2} I_{n})\) for noise schedules \(\alpha_{t}\) and \(\sigma_{t}\)
- Next, we consider what should \(v_{t}(x_{t} | z)\) be such that the following ODE leads to the above conditional distribution (for any fixed Z ) 
\[\frac{d x_{t}}{ d t}=v_{t}\left(x_{t} | z\right)\]

This is similar to what we have learned for the forward SDE \(d x_{t}=f(t) x_{t} d t+g(t) d w_{t}\)

## Conditional velocity field
- Recall that for \(\lambda_{t}:=\log (\alpha_{t} / \sigma_{t})\) , we need 
\[f(t)=\frac{d \log \left(\alpha_{t}\right)}{d t}, g(t)^{2}=-2 \sigma_{t}^{2} \frac{d \lambda_{t}}{ d t}\]

- Additionally, the forward SDE has a corresponding probability flow ODE: 
\[\frac{d x_{t}}{ d t}=f(t) x_{t}-\frac{1}{2} g(t)^{2} \nabla_{x} \log p_{t}\left(x_{t}\right)\]

- Then, we may consider that for the conditional case, we should set 
\[
\begin{aligned} 
v_{t}\left(x_{t} | z\right) & =f(t) x_{t}-\frac{1}{2} g(t)^{2} \nabla_{x} \log p_{t}\left(x_{t} | z\right) \\ 
& =\frac{d \log \left(\alpha_{t}\right)}{d t} x_{t}+\sigma_{t}^{2} \frac{d \lambda_{t}}{ d t} \cdot \frac{\alpha_{t} z-x_{t}}{\sigma_{t}^{2}} \\ 
& =\frac{d \log \left(\alpha_{t}\right)}{d t} x_{t}+\frac{d \lambda_{t}}{ d t} \cdot\left(\alpha_{t} z-x_{t}\right) \\ 
& =\frac{d \log \left(\sigma_{t}\right)}{d t} x_{t}+\alpha_{t} \frac{d \lambda_{t}}{ d t} z \\ 
& =\frac{\sigma_{t}'}{\sigma_{t}} x_{t}+\left(\alpha_{t}'-\alpha_{t} \frac{\sigma_{t}'}{\sigma_{t}}\right) z 
\end{aligned}
\]

Additionally, we can show that if \(x_{0} \sim p_{0}=N(0, I_{n})\) , the above construction of \(v_{t}(x_{t} | z)\) indeed leads to the desired conditional distributions \(p_{t}(x_{t} | z)=N(\alpha_{t} z, \sigma_{t}^{2} I_{n})\)

## FM for Gaussian conditional distributions
- For the Gaussian conditional distributions \(x_{t} | z \sim p_{t}(x_{t} | z)=N(\alpha_{t} z, \sigma_{t}^{2} I_{n})\) , we have 
\[x_{t}=\alpha_{t} z+\sigma_{t} \epsilon, \epsilon \sim \mathcal{N}\left(0, I_{n}\right)\]

Then, we obtain 
\[
\begin{aligned} 
v_{t}\left(x_{t} | z\right) & =\frac{\sigma_{t}'}{\sigma_{t}} x_{t}+\left(\alpha_{t}'-\alpha_{t} \frac{\sigma_{t}'}{\sigma_{t}}\right) z \\ 
& =\frac{\sigma_{t}'}{\sigma_{t}}\left(\alpha_{t} z+\sigma_{t} \epsilon\right)+\left(\alpha_{t}'-\alpha_{t} \frac{\sigma_{t}'}{\sigma_{t}}\right) z \\ 
& =\sigma_{t}' \epsilon+\alpha_{t}' z 
\end{aligned}
\]

Therefore, the CFM loss can be rewritten as (also setting \(p_{1}=p_{data }\) as usual) 
\[
\begin{aligned} 
& \mathbb{E}_{z \sim p_{data }} \mathbb{E}_{x_{t} | z \sim p_{t}\left(x_{t} | z\right)}\left[\left\| v_{\theta}\left(x_{t}, t\right)-v_{t}\left(x_{t} | z\right)\right\| _{2}^{2}\right] \\ 
& =\mathbb{E}_{z \sim p_{data }, \epsilon \sim \mathcal{N}\left(0, I_{n}\right)}\left[\left\| v_{\theta}\left(\alpha_{t} z+\sigma_{t} \epsilon, t\right)-\left(\sigma_{t}' \epsilon+\alpha_{t}' z\right)\right\| _{2}^{2}\right] 
\end{aligned}
\]

- For flow models, a typical choice is \(\alpha_{t}=t\) and \(\sigma_{t}=1-t\) (thus \(p_{0}\) is standard Gaussian and \(p_{1}=p_{data }\) ), and the corresponding training objective is 
\[\mathbb{E}_{z \sim p_{data }, \epsilon \sim \mathcal{N}\left(0, I_{n}\right)}\left[\left\| v_{\theta}(t z+(1-t) \epsilon, t)-(z-\epsilon)\right\| _{2}^{2}\right]\]

## The training and sampling of flow models
### Algorithm 3 Flow Matching Training Procedure (here for Gaussian CondOT path \(p(x|z) = N(tz, (1 -t)^2)\))
**Require**: A dataset of samples \(z \sim P_{data}\), neural network \(v_\theta\)
1. for each mini-batch of data do 
2.  Sample a data example \(z\) from the dataset.
3.  Sample a random time \(t \sim \text{Unif}[0,1]\). 
4.  Sample noise \(\epsilon \sim \mathcal{N}\left(0, I_n\right)\)
5.  Set \(x=tz+(1-t)\epsilon\) (General case: \(x \sim p(\cdot|z)\))
6.  Compute loss \(L(\theta)=\left\|v_\theta(x,t)-(z-\epsilon)\right\|_2^2\) (General case: \(L(\theta)=\left\|v_\theta(x,t)-v_{t}(x|z)\right\|_2^2\))
7.  Update the model parameters \(\theta\) via gradient descent on \(L(\theta)\) 
8. end for

### Algorithm 1 Sampling from a Flow Model with Euler method
**Require**: Neural network vector field \(v_\theta\), number of steps \(n\)
1. Set \(t=0\) 
2. Set step size \(h = \frac{1}{n}\)
3. Draw a sample \(x_0 \sim P_{init}\)
4. for \(i= 1,...,n -1\) do 
5.  \(x_{t+h}= x_t+hv_\theta(x_t,t)\) 
6.  Update \(t \leftarrow t+h\)
7. end for
8. return \(x_1\)

## Comparison of diffusion models and general flow models

| Aspect | Diffusion Model | General FM |
| --- | --- | --- |
| Source dist. \(p_{src }\) | Gaussian prior | Any |
| Target dist. \(p_{tgt }\) | Data distribution | Any |
| Latent dist. \(\pi(z)\) | \(p_{data }\) | See Section 5.3.2 |
| Cond. dist. \(p_{t}(x_{t} | z)\) | \(N(x_{t} ; \alpha_{t} x_{0}, \sigma_{t}^{2} I)\) | See Section 5.3.2 |
| Marginal dist. \(p_{t}(x_{t})\) | \(\int p_{t}(x_{t} | x_{0}) p_{data }(x_{0}) d x_{0}\) | \(\int p_{t}(x_{t} | z) \pi(z) d z\) |
| Cond. velocity \(v_{t}(x | z)\) | \(f(t) x-\frac{1}{2} g^{2}(t) \nabla \log p_{t}(x | x_{0})\) | See Section 5.3.2 |
| Marginal velocity \(v_{t}(x)\) | \(f(t) x-\frac{1}{2} g^{2}(t) \nabla \log p_{t}(x)\) | See Equation (5.2.10) |
| Learning objective | \(L_{SM}=L_{DSM}+C\) | \(L_{FM}=L_{CFM}+C\) |
| Underlying Rule | Fokker-Planck / Continuity Equation | - |


