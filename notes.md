# State Estimation for Robotics

## 概率密度函数

## 高斯概率密度函数

### 高斯概率密度函数

#### 定义

一维高斯（正态）分布的概率密度函数如下：
$$
p(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{1}{2} \frac{(x - \mu)^2}{\sigma^2} \right)
$$
其中$\mu$ 称为均值（mean），$\sigma^2$ 为方差（variance），$\sigma$ 称为标准差（standard deviation）

![Alt text](images/%E4%B8%80%E7%BB%B4%E9%AB%98%E6%96%AF%E5%AF%86%E5%BA%A6%E5%87%BD%E6%95%B0.jpg)

##### 中心极限

假设我们有一个样本 $\{X_1, X_2, \ldots, X_n\}$，它们是从某个总体分布中独立抽取的，且总体分布具有均值 $\mu$ 和方差 $\sigma^2$。那么，当样本容量 $n$ 足够大时，样本均值 $\bar{X}$ 的分布将近似于正态分布，其均值为 $\mu$，方差为 $\frac{\sigma^2}{n}$。即：
\[
\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i
\]

中心极限定理正式表述如下：

\[
\frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)
\]

其中，$\xrightarrow{d}$ 表示收敛于分布，$\mathcal{N}(0, 1)$ 表示标准正态分布（均值为0，方差为1）。

### 多维高斯分布

多维高斯分布是一种描述多变量联合概率分布的分布形式。它是高斯分布在高维空间的推广。给定一个 $N$ 维随机向量 $\mathbf{x} \in \mathbb{R}^N$，其服从多维高斯分布的概率密度函数（PDF）表示为：

\[
p(\mathbf{x}|\mu, \Sigma) = \frac{1}{\sqrt{(2\pi)^N \det \Sigma}} \exp\left( -\frac{1}{2} (\mathbf{x} - \mu)^\mathrm{T} \Sigma^{-1} (\mathbf{x} - \mu) \right)
\]

其中：

- $\mathbf{x}$ 是 $N$ 维随机向量。
- $\mu$ 是 $N$ 维均值向量。
- $\Sigma$ 是 $N \times N$ 的协方差矩阵。

参数解释：

均值向量 $\mu$
  \[
  \mu = \mathbb{E}[\mathbf{x}] = \int_{-\infty}^{\infty} \mathbf{x} p(\mathbf{x}) \, d\mathbf{x}
  \]

协方差矩阵 $\Sigma$
  \[
  \Sigma = \mathbb{E} \left[ (\mathbf{x} - \mu)(\mathbf{x} - \mu)^\mathrm{T} \right] = \int_{-\infty}^{\infty} (\mathbf{x} - \mu)(\mathbf{x} - \mu)^\mathrm{T} p(\mathbf{x}) \, d\mathbf{x}
  \]

基本就是一维高斯分布的拓展，当$N = 1$时就是一维。

### Isserlis 定理

Isserlis 定理，也被称为 Wick 定理，是一种在计算高斯分布下的高阶矩（moments）时非常有用的工具。它的主要目的是将高斯随机变量的高阶矩表示为二阶矩（即协方差）的组合。

Isserlis 定理适用于均值为零的高斯随机变量。假设我们有 $2n$ 个均值为零的高斯随机变量 $X_1, X_2, \ldots, X_{2n}$，那么它们的高阶矩可以表示为所有可能的配对（pairings）之积的和：

\[
\mathbb{E}[X_1 X_2 \cdots X_{2n}] = \sum \prod \mathbb{E}[X_i X_j]
\]

其中，和的每一项表示将 $2n$ 个随机变量进行不同的配对，并且对每个配对计算二阶矩 $\mathbb{E}[X_i X_j]$ 的乘积。

对于四阶矩（$n=2$）的情况，我们有四个随机变量 $X_1, X_2, X_3, X_4$。根据 Isserlis 定理，四阶矩可以表示为：

\[
\mathbb{E}[X_1 X_2 X_3 X_4] = \mathbb{E}[X_1 X_2] \mathbb{E}[X_3 X_4] + \mathbb{E}[X_1 X_3] \mathbb{E}[X_2 X_4] + \mathbb{E}[X_1 X_4] \mathbb{E}[X_2 X_3]
\]

\subsection*{一般形式}

对于任意 $2n$ 个高斯随机变量 $X_1, X_2, \ldots, X_{2n}$，Isserlis 定理表示为：

\[
\mathbb{E}[X_1 X_2 \cdots X_{2n}] = \sum_{\text{pairings}} \prod_{(i,j)} \mathbb{E}[X_i X_j]
\]

\subsection*{例子}

假设我们有六个高斯随机变量 $X_1, X_2, X_3, X_4, X_5, X_6$，那么它们的六阶矩可以表示为所有可能配对的组合：

\[
\mathbb{E}[X_1 X_2 X_3 X_4 X_5 X_6] = \sum \mathbb{E}[X_{i_1} X_{i_2}] \mathbb{E}[X_{i_3} X_{i_4}] \mathbb{E}[X_{i_5} X_{i_6}]
\]

\subsection*{应用}

Isserlis 定理在许多领域有应用，包括：

\begin{itemize}
    \item \textbf{量子场论}：用于计算费曼图中的期望值。
    \item \textbf{信号处理}：用于计算高阶累积量和高阶谱。
    \item \textbf{统计学}：用于计算高阶协方差矩阵。
\end{itemize}

\subsection*{总结}

Isserlis 定理（Wick 定理）是一个强有力的工具，用于将高斯随机变量的高阶矩表示为二阶矩的组合。这一特性大大简化了高阶矩的计算，特别是在处理复杂的高斯随机变量系统时。

\end{document}

#### 协方差矩阵$\Sigma$

对于一个随机向量 $\mathbf{x} \in \mathbb{R}^N$，协方差矩阵 $\Sigma$ 的定义为：

\[
\Sigma = \mathbb{E} \left[ (\mathbf{x} - \mathbb{E}[\mathbf{x}]) (\mathbf{x} - \mathbb{E}[\mathbf{x}])^\mathrm{T} \right]
\]

在均值为零的情况下，$\mathbb{E}[\mathbf{x}] = 0$，因此上式简化为：

\[
\Sigma = \mathbb{E} \left[ \mathbf{x} \mathbf{x}^\mathrm{T} \right]
\]

这意味着，对于均值为零的多维高斯随机变量 $\mathbf{x}$，其协方差矩阵 $\Sigma$ 就是 $\mathbf{x} \mathbf{x}^\mathrm{T}$ 的期望值。

\section*{数学推导}

假设 $\mathbf{x}$ 是一个 $N$ 维零均值高斯随机向量，其协方差矩阵为 $\Sigma$，我们有：

\[
\mathbf{x} \sim \mathcal{N}(0, \Sigma)
\]

根据协方差矩阵的定义：

\[
\Sigma = \mathbb{E} \left[ (\mathbf{x} - \mathbb{E}[\mathbf{x}]) (\mathbf{x} - \mathbb{E}[\mathbf{x}])^\mathrm{T} \right]
\]

由于 $\mathbb{E}[\mathbf{x}] = 0$：

\[
\Sigma = \mathbb{E} \left[ \mathbf{x} \mathbf{x}^\mathrm{T} \right]
\]

这说明，均值为零的高斯随机向量 $\mathbf{x}$ 的协方差矩阵 $\Sigma$ 可以直接通过计算 $\mathbf{x} \mathbf{x}^\mathrm{T}$ 的期望值来得到。

\section*{结论}

因此，我们有：

\[
\mathbb{E}[\mathbf{x} \mathbf{x}^\mathrm{T}] = \Sigma
\]

这是均值为零的多维高斯分布的一个基本性质，用于在许多概率和统计计算中，包括我们之前讨论的高阶矩的计算。

\end{document}

## 线性高斯系统的状态估计

# 运动和观测模型

定义运动和观测模型如下：

运动方程:  
$$
\mathbf{x}_k = \mathbf{A}_{k-1}\mathbf{x}_{k-1} + \mathbf{v}_k + \mathbf{w}_k, \quad k = 1, \ldots, K
$$

观测方程:  
$$
\mathbf{y}_k = \mathbf{C}_k \mathbf{x}_k + \mathbf{n}_k, \quad k = 0, \ldots, K
$$

其中 \( k \) 为时间下标，最大值为 \( K \)。
各变量的含义如下：

- **系统状态:**  $\mathbf{x}_k \in \mathbb{R}^N$
- **初始状态:**  $\mathbf{x}_0 \in \mathbb{R}^N \sim \mathcal{N}(\check{{\mathbf{x}}}_0, \check{\mathbf{P}}_0)$
- **输入:**  $\mathbf{v}_k \in \mathbb{R}^N$
- **过程噪声:**  $\mathbf{w}_k \in \mathbb{R}^N \sim \mathcal{N}(0, \mathbf{Q}_k)$
- **测量:**  $\mathbf{y}_k \in \mathbb{R}^M$
- **测量噪声:**  $\mathbf{n}_k \in \mathbb{R}^M \sim \mathcal{N}(0, \mathbf{R}_k)$

用 $\hat{}$（上帽子）来表示后验估计（包含了观测量的），而用 $\check{}$（下帽子）表示先验估计（不含观测量的）。


这些变量中，除了 \(\mathbf{v}_k\) 为确定性变量外，其他的都是随机变量。噪声和初始状态一般假设为互不相关的，并且在各个时刻与自己也互不相关。矩阵 \(\mathbf{A}_k \in \mathbb{R}^{N \times N}\) 称为**转移矩阵**（transition matrix），在状态空间模型中，转移矩阵用于描述系统状态从一个时间点到下一个时间点的变化。\(\mathbf{C}_k \in \mathbb{R}^{M \times N}\) 称为**观测矩阵**（observation matrix）。

我们定义**状态估计问题**为：

**状态估计问题是指，在 \(k\) 个（一个或多个）时间点上，基于初始的状态信息 \(\tilde{\mathbf{x}}_0\)、一系列观测数据 \(\mathbf{y}_{0:K,\text{meas}}\)、一系列输入 \(\mathbf{u}_{1:K}\)，以及系统的运动和观测模型，来计算系统的真实状态的估计值 \(\hat{\mathbf{x}}_k\)。**

### 最大后验估计（Maximum A Posteriori, MAP）

**MAP** 的⽬标：
$$
\hat{\mathbf{x}} = \arg \max_{\mathbf{x}} p(\mathbf{x}|\mathbf{v}, \mathbf{y}) 
$$
$$
x = x_{0:K} = (x_0, \ldots, x_K), \quad \mathbf{v} = (\tilde{\mathbf{x}}_0, \mathbf{v}_{1:K}), \quad \mathbf{y} = \mathbf{y}_{0:K} = (y_0, \ldots, y_K)
$$
首先，用贝叶斯公式重写 MAP 估计：

$$
\hat{\mathbf{x}} = \arg \max_{\mathbf{x}} p(\mathbf{x}|\mathbf{v}, \mathbf{y}) = \arg \max_{\mathbf{x}} \frac{p(\mathbf{y}|\mathbf{x}, \mathbf{v}) p(\mathbf{x}|\mathbf{v})}{p(\mathbf{y}|\mathbf{v})} = \arg \max_{\mathbf{x}} p(\mathbf{y}|\mathbf{x}) p(\mathbf{x}|\mathbf{v})
$$

在线性系统中，高斯密度函数可展开为：

1. 初始状态的概率密度函数：
   \[
   p(\mathbf{x}_0|\tilde{\mathbf{x}}_0) = \frac{1}{\sqrt{(2\pi)^N \det \mathbf{P}_0}} \exp \left( -\frac{1}{2} (\mathbf{x}_0 - \tilde{\mathbf{x}}_0)^\mathrm{T} \mathbf{P}_0^{-1} (\mathbf{x}_0 - \tilde{\mathbf{x}}_0) \right)
   \]

2. 状态转移概率密度函数：
   \[
   p(\mathbf{x}_k|\mathbf{x}_{k-1}, \mathbf{v}_k) = \frac{1}{\sqrt{(2\pi)^N \det \mathbf{Q}_k}} \exp \left( -\frac{1}{2} (\mathbf{x}_k - \mathbf{A}_{k-1}\mathbf{x}_{k-1} - \mathbf{v}_k)^\mathrm{T} \mathbf{Q}_k^{-1} (\mathbf{x}_k - \mathbf{A}_{k-1}\mathbf{x}_{k-1} - \mathbf{v}_k) \right)
   \]

3. 观测概率密度函数：
   \[
   p(\mathbf{y}_k|\mathbf{x}_k) = \frac{1}{\sqrt{(2\pi)^M \det \mathbf{R}_k}} \exp \left( -\frac{1}{2} (\mathbf{y}_k - \mathbf{C}_k\mathbf{x}_k)^\mathrm{T} \mathbf{R}_k^{-1} (\mathbf{y}_k - \mathbf{C}_k\mathbf{x}_k) \right)
   \]


**定义目标函数**

**初始状态项 $J_{x,0}(\mathbf{x})$:**
$$
J_{x,0}(\mathbf{x}) = \frac{1}{2} (\mathbf{x}_0 - \tilde{\mathbf{x}}_0)^\mathrm{T} \mathbf{P}_0^{-1} (\mathbf{x}_0 - \tilde{\mathbf{x}}_0), \quad k = 0 
$$
这项度量了初始状态$x$与初始估计$x_0$的差异，反映了初始估计的准确性。

**状态转移项$J_{x,k}(\mathbf{x})$:**
$$
J_{x,k}(\mathbf{x}) = \frac{1}{2} (\mathbf{x}_k - \mathbf{A}_{k-1}\mathbf{x}_{k-1} - \mathbf{v}_k)^\mathrm{T} \mathbf{Q}_k^{-1} (\mathbf{x}_k - \mathbf{A}_{k-1}\mathbf{x}_{k-1} - \mathbf{v}_k), \quad k = 1, \ldots, K
$$
这项度量了每个时间步的状态$x_k$与基于前一状态$x_{k-1}$和控制输入$v_k$预测的状态之间的差异，反映了状态转移模型的准确性。
**观测项$J_{y,k}(\mathbf{x})：$**
$$
J_{y,k}(\mathbf{x}) = \frac{1}{2} (\mathbf{y}_k - \mathbf{C}_k\mathbf{x}_k)^\mathrm{T} \mathbf{R}_k^{-1} (\mathbf{y}_k - \mathbf{C}_k\mathbf{x}_k), \quad k = 0, \ldots, K
$$
这项度量了观测值$\mathbf{y}_k$与基于当前状态$\mathbf{x}_k$的预测观测值之间的差异，反映了观测模型的准确性。

综合一下：
$$
J_{x,k}(\mathbf{x}) = 
\begin{cases} 
\frac{1}{2} (\mathbf{x}_0 - \tilde{\mathbf{x}}_0)^\mathrm{T} \mathbf{P}_0^{-1} (\mathbf{x}_0 - \tilde{\mathbf{x}}_0), & k = 0 \\
\frac{1}{2} (\mathbf{x}_k - \mathbf{A}_{k-1}\mathbf{x}_{k-1} - \mathbf{v}_k)^\mathrm{T} \mathbf{Q}_k^{-1} (\mathbf{x}_k - \mathbf{A}_{k-1}\mathbf{x}_{k-1} - \mathbf{v}_k), & k = 1, \ldots, K 
\end{cases}
$$

$$
J_{y,k}(\mathbf{x}) = \frac{1}{2} (\mathbf{y}_k - \mathbf{C}_k\mathbf{x}_k)^\mathrm{T} \mathbf{R}_k^{-1} (\mathbf{y}_k - \mathbf{C}_k\mathbf{x}_k), \quad k = 0, \ldots, K
$$

这些量表示马氏距离（Mahalanobis Distance），用于衡量实际状态与估计状态的差异。

综合所有项，可以定义**整体的目标函数** \(J(\mathbf{x})\) 为：

$$
J(\mathbf{x}) = \sum_{k=0}^{K} \left( J_{x,k}(\mathbf{x}) + J_{y,k}(\mathbf{x}) \right) \quad \text{(3.10)}
$$


## 非线性非高斯系统的状态估计


