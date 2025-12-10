# 距离度量知识点

## 知识点1：Total variation distance
- Let \( P \sim p \) and \( Q \sim q \) be distributions on \( \mathcal{X} = \mathbb{R}^d \). The total variation distance between \( P \) and \( Q \) is:
  \[
  \text{TV}(P, Q) = \sup_{A \subset \mathcal{X}} |P(A) - Q(A)|
  \]

- **Properties of total variation distance**:
  - \( 0 \leq \text{TV}(P, Q) \leq 1 \)
  - \( \text{TV}(P, Q) = 0 \iff P = Q \)
  - TV is a metric
  - \( \text{TV}(P, Q) = 1 \iff \exists A \subset \mathcal{X} \text{ with } P(A) = 1 \text{ and } Q(A) = 0 \)

- **Alternative forms (Scheffe's Theorem)**:
  - \( \text{TV}(P, Q) = \frac{1}{2} \int |p(x) - q(x)| \text{d}x \)
  - \( \text{TV}(P, Q) = 1 - \int \min\{p(x), q(x)\} \text{d}x \)
  - \( \text{TV}(P, Q) = P(B) - Q(B) \) for \( B := \{x : p(x) \geq q(x)\} \)


## 知识点2：Hellinger distance
- Let \( P \sim p \) and \( Q \sim q \) be distributions on \( \mathcal{X} = \mathbb{R}^d \). The Hellinger distance between \( P \) and \( Q \) is:
  \[
  \text{H}(P, Q) = \left( \int \left( \sqrt{p(x)} - \sqrt{q(x)} \right)^2 \text{d}x \right)^{1/2}
  \]

- **Properties of Hellinger distance**:
  - \( \text{H}(P, Q) \) is the \( L_2 \) distance between \( \sqrt{p} \) and \( \sqrt{q} \)
  - \( 0 \leq \text{H}(P, Q)^2 \leq 2 \)
  - \( \text{H}(P, Q) = 0 \iff P = Q \)
  - H is a metric
  - \( \text{H}(P, Q)^2 = 2 \iff \exists A \subset \mathcal{X} \text{ with } P(A) = 1 \text{ and } Q(A) = 0 \)


## 知识点3：Kullback-Liebler (KL) divergence
- **Definition**: The KL-divergence between \( P \sim p \) and \( Q \sim q \) is:
  \[
  \text{KL}(P\|Q) = \text{KL}(p\|q) = \int p(x) \log \left( \frac{p(x)}{q(x)} \right) \text{d}x
  \]
  (Analogous for discrete distributions.)

- KL-divergence is **not symmetric** and **not a metric**. Note that:
  \[
  \text{KL}(P\|Q) = \mathbb{E}_p \left[ \log \left( \frac{p(X)}{q(X)} \right) \right]
  \]

- \( \text{KL}(P\|Q) \geq 0 \), with equality iff \( P = Q \)


## 知识点4：KL divergence examples
- **Example**: Let \( p, q \) be pmfs on \( \{0,1\} \) with:
  \[
  p(0) = p(1) = \frac{1}{2}, \quad q(0) = \frac{1-\epsilon}{2}, \, q(1) = \frac{1+\epsilon}{2}
  \]
  Exact expressions:
  - \( \text{KL}(p\|q) = -\frac{1}{2} \log(1 - \epsilon^2) \)
  - \( \text{KL}(q\|p) = \frac{1}{2} \log(1 - \epsilon^2) + \frac{\epsilon}{2} \log\left( \frac{1+\epsilon}{1-\epsilon} \right) \)

- **Example (Exercise)**: For \( P \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0) \), \( Q \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) \) (positive definite \( \boldsymbol{\Sigma}_0, \boldsymbol{\Sigma}_1 \)):
  \[
  2\text{KL}(P\|Q) = \text{tr}\left( \boldsymbol{\Sigma}_1^{-1}\boldsymbol{\Sigma}_0 \right) + (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^T \boldsymbol{\Sigma}_1^{-1} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0) + \ln\left( \frac{\det(\boldsymbol{\Sigma}_1)}{\det(\boldsymbol{\Sigma}_0)} \right) - d
  \]


## 知识点5：Jensen-Shannon (JS) divergence
- **Definition**: The JS-divergence between \( P \sim p \) and \( Q \sim q \) is:
  \[
  \text{JS}(P\|Q) = \text{JS}(p\|q) = \frac{1}{2}\text{KL}\left( p \bigg\| \frac{p+q}{2} \right) + \frac{1}{2}\text{KL}\left( q \bigg\| \frac{p+q}{2} \right)
  \]

- **Properties**:
  - JS-divergence is symmetric
  - JS-divergence is bounded: \( [0, \log 2] \)


## 知识点6：f-divergence
- A general divergence (Ali-Silvey divergence): Let \( P \sim p, Q \sim q \) (on \( \mathcal{X} = \mathbb{R}^d \)), and \( f : \mathbb{R}_+ \to \mathbb{R} \) (convex, \( f(1) = 0 \)). The f-divergence is:
  \[
  \text{D}_f(P\|Q) = \text{D}_f(p\|q) := \int q(x)f\left( \frac{p(x)}{q(x)} \right) \text{d}x
  \]

- **Examples**:
  - KL-divergence: \( f(t) = t\log t \) (convex, \( f(1)=0 \))
  - TV-distance: \( f(t) = \frac{1}{2}|t - 1| \)
  - Hellinger distance: \( f(t) = (\sqrt{t} - 1)^2 = t + 1 - 2\sqrt{t} \)
  - \( \chi^2 \)-divergence: \( f(t) = \frac{1}{2}(t - 1)^2 \), so:
    \[
    \chi^2(P\|Q) = \chi^2(p\|q) := \frac{1}{2} \int q(x) \left( \frac{p(x)}{q(x)} - 1 \right)^2 \text{d}x
    \]


## 知识点7：Hellinger distance vs. total variation distance
- Inequalities for f-divergences (useful for product distributions):
  - For densities \( p, q \):
    \[
    \int \min\{p(x), q(x)\}\text{d}x \geq \frac{1}{2} \left( \int \sqrt{p(x)q(x)}\text{d}x \right)^2 = \frac{1}{2} \left( 1 - \frac{1}{2}\text{H}(p, q)^2 \right)^2
    \]

- For distributions \( P, Q \):
  \[
  \frac{1}{2}\text{H}(P, Q)^2 \leq \text{TV}(P, Q) \leq \text{H}(P, Q)\sqrt{1 - \frac{\text{H}(P, Q)^2}{4}}
  \]
  - \( \text{H}(P, Q)^2 = 0 \iff \text{TV}(P, Q) = 0 \); \( \text{H}(P, Q)^2 = 2 \iff \text{TV}(P, Q) = 1 \)
  - \( \text{H}(P_n, Q_n) \to 0 \iff \text{TV}(P_n, Q_n) \to 0 \)


## 知识点8：KL divergence vs. total variation and Hellinger distances
- For any \( P, Q \):
  - \( \text{H}(P, Q)^2 \leq \text{KL}(P\|Q) \)
  - (Exercise) \( \text{TV}(P, Q)^2 \leq \frac{\text{KL}(P\|Q)}{2} \) (Pinsker's inequality)

- **Product distribution**: For \( P = \otimes_{i=1}^n P_i \) (densities \( p(x_1,\dots,x_n)=p_1(x_1)\dots p_n(x_n) \)):
  - **Tensorization**: For \( P_1,\dots,P_n \) and \( Q_1,\dots,Q_n \):
    - \( \text{TV}(\otimes_{i=1}^n P_i, \otimes_{i=1}^n Q_i) \leq \sum_{i=1}^n \text{TV}(P_i, Q_i) \)
    - \( \text{H}(\otimes_{i=1}^n P_i, \otimes_{i=1}^n Q_i)^2 \leq \sum_{i=1}^n \text{H}(P_i, Q_i)^2 \)
    - \( \text{KL}(\otimes_{i=1}^n P_i, \otimes_{i=1}^n Q_i) = \sum_{i=1}^n \text{KL}(P_i, Q_i) \)


## 知识点9：Wasserstein distance
- **Coupling**: A pair \( (X, Y) \) (jointly distributed) where \( X \sim P \), \( Y \sim Q \).

- **Wasserstein \( \rho \)-distance**: For \( P \sim p, Q \sim q \):
  \[
  W_\rho(P, Q) = \left( \inf_{\gamma \sim \Gamma(P, Q)} \mathbb{E}_{(X,Y) \sim \gamma} \left[ |X - Y|^\rho \right] \right)^{1/\rho}
  \]
  (\( \Gamma(P, Q) \): set of all couplings of \( P, Q \))

- **1-d continuous case**:
  \[
  W_1(p, q) = \int |\text{cdf}_p(x) - \text{cdf}_q(x)| \text{d}x
  \]