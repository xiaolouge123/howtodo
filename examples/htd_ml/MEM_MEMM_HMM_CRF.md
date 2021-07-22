### 最大熵原理

概率模型学习的一个准则，在学习概率模型时，**在所有可能的概率模型（分布）中，熵最大的模型是最好的模型。**

通常用约束条件来限制模型集合，即在满足限制条件的模型中，选熵最大的模型。

假设随机变量$X$的概率分布$P(X)$，熵是$H(P)=-\sum\limits_xP(x)logP(x)$

熵满足这个不等式$0\le H(P)\le log|X|$

直观的说，最大熵原理认为选择的概率模型首先满足已有的事实，即约束条件，对于不确定的部分，就直接假设是等可能的，即用熵作为衡量**等可能**的量化指标。

### 最大熵模型

应用到分类模型。假设分类模型是一个条件概率分布$P(Y|X), X\in \chi \subseteq \mathbb{R}^n, Y\in \gamma$, $X,Y$表示输入和输出变量，$\chi,\gamma$，表示输入和输出变量的取值集合，这个模型表示为，给定输入$X$以条件概率$P(Y|X)$输出$Y$。

选定一个训练集$T=\{(x_1,y_1), (x_2,y_2),...,(x_n,y_n)\}$，通过训练集的数据我们可以确定两个经验分布，$\widetilde{P}(X,Y), \widetilde{P}(X)$, 就是联合分布$P(X,Y)$和边缘分布$P(X)$的经验分布。
$$
\widetilde{P}(X=x,Y=y) = \frac{count(X=x,Y=y)}{n}\\
\widetilde{P}(X=x) = \frac{count(X=x)}{n}
$$
定义特征函数$f(x,y)$描述输入x和输出y之间的某一个事实，
$$
f(x,y)=
\begin{cases}
1, x与y满足某一事实\\
0, others
\end{cases}
$$

1. 特征函数f(x,y)关于经验分布$\widetilde{P}(X,Y)$的期望值用$E_{\tilde{p}}(f)$表示。

$$
E_{\tilde{P}}(f) = \sum_{x,y} \widetilde{P}(x,y)f(x,y)
$$

2. 特征函数f(x,y)关于模型$P(Y｜X)$和经验分布$\widetilde{P}(X)$的期望值用$E_P(f)$表示。
   $$
   E_P(f) = \sum_{x,y}\widetilde{P}(x)P(y|x)f(x,y)
   $$

如果模型能够学习到训练数据中的信息，那么就可以假设这两个期望值相等，即
$$
E_{\tilde{P}}(f) =E_P(f)
\\
\sum_{x,y} \widetilde{P}(x,y)f(x,y) =\sum_{x,y}\widetilde{P}(x)P(y|x)f(x,y)
$$
以上面这个等式作为模型学习的约束条件，如果有m个特征方程$f_i(x,y)$，就有m个约束条件。

**定义**

假设满足所有约束条件的模型集合为：
$$
\C \equiv\{P\in \rho | E_{\tilde{P}}(f_i) =E_P(f_i), i=1,2,3,..., m\}
$$
定义在条件概率分布$P(Y|X)$上的条件熵为：
$$
H(P) = -\sum_{x,y}\widetilde{P}(x)P(y|x)logP(y|x)
$$
那么在模型候选集$\C$中能让条件熵$H(P)$最大的模型称为**最大熵模型**。



### MEMM 

### HMM

### CRF

