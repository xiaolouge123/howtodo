信息：指音讯、消息、通讯系统传输和处理的对象，泛指人类社会传播的一切内容。

**信息熵**：用来度量信息。

信息量的大小和信息的**不确定性**有直接的关系。如，要搞清楚一个非常不确定的事，需要大量的信息。反之，比较明确的事，只需要不多的信息就能搞清楚。

**信息量的度量就可以等于不确定性的多少。**

那怎么去按照特性去构造这个**不确定性**与**信息量**之间的函数呢？

**特性1**: 不确定性在描述信息量时应该是单调的。(不确定性)$\uarr \Rightarrow$ (信息量)$\uparrow$

**特性2**:如果有两个不相关的是件x和y，那么同时观察事件x和y需要的信息量应该等于分别观察x，观察y所需的信息量。

对于随机变量$x$， 概率$p(x)$就可以描述随机变量的不确定性，又为了满足上述的特性，可以发现log函数可以满足需求，可以得到：
$$
信息量 \quad I(x)=-logp(x)
$$
对于概率分布$p(x)$的信息量的期望：
$$
E[I(X)]=H(X)=-\sum\limits_xp(x)logp(x)
\\
当随机变量X是均匀分布时，H(X)最大，且\;0 \le H(X) \le logn \quad n时X的取值数
$$
**联合熵**多随机变量的联合分布$p(X,Y)$的信息量是：$H(X,Y)=-\sum\limits_{x,y}p(x,y)logp(x,y)=-\sum\limits_{i=1}^{n}\sum\limits_{j=1}^{m}p(x_i,y_j)logp(x_i,y_j)$

**条件熵**
$$
H(Y|X) = \sum\limits_xp(x)H(Y|X=x)
\\= -\sum\limits_xp(x)\sum\limits_yp(y|x)logp(y|x)
\\= -\sum\limits_x\sum\limits_yp(x)p(y|x)logp(y|x)
\\=-\sum\limits_x\sum\limits_yp(x,y)logp(y|x)
\\=-\sum\limits_{x,y}p(x,y)logp(y|x)
$$
联合熵，条件熵，熵的计算
$$
H(X,Y)=-\sum\limits_{x,y}p(x,y)logp(x,y) \\
= -\sum\limits_{x,y}p(x,y)log\big(p(y|x)p(x)\big) \\
= -\sum\limits_{x,y}p(x,y)logp(y|x)  -\sum\limits_{x,y}p(x,y)logp(x) \\
= H(Y|X) -\sum\limits_{x,y}p(x,y)logp(x) \\
= H(Y|X) -\sum\limits_x\sum\limits_yp(x,y)logp(x) \\
= H(Y|X) - \sum\limits_xlogp(x)\sum\limits_yp(x,y) \\
= H(Y|X) - \sum\limits_x\big(logp(x)\big)p(x) \\
= H(Y|X) - \sum\limits_xp(x)logp(x) \\
= H(Y|X) + H(X)
$$
描述随机事件X和Y所需要的信息量是，描述X需要的信息量和在给定X条件下描述Y所需的信息量的加和。

**交叉熵**

对于随机变量X假设可以用两种概率分布$p(x),q(x)$来描述。$p(x)$是X的真实分布，$q(x)$是从假设空间得到的一个估计的概率分布。
$$
那么, 当用真实分布度量随机变量X的信息量的期望：\\
H(p)=-\sum\limits_x p(x)logp(x)
\\ 如果用假设的分布q来描述信息量的期望：\\
H(p,q) = -\sum\limits_xp(x)logq(x) 
$$
说明：对于假设的概率分布q(x),我们得到的是随机变量的取值x对应的概率值，用假设的分布估计信息量。而p(x)是其实使用样本分布来代替总体分布,对于实际样本其分布是确定的，我们只用替换q(x)的来表达引入假设导致信息量期望的变化。在分类问题中p(x)实际上变成了一种one-hot的分布,形如[0，0，1，0，0].

用于Jensen不等式，$-\sum\limits_{i=1}^{n}p_ilogp_i \le -\sum\limits_{i=1}^{n}p_ilogq_i, 仅当\forall i,p_i=q_i,时取等号。$所以只用最小化交叉熵，才能让假设分布q(x)趋近于真实分布p(x)。只有我们完全猜中了真实分布，交叉熵才能为0。

**相对熵，KL散度**

随机变量X假设可以用两种概率分布$p(x),q(x)$来描述。则p对q的相对熵是：
$$
D_{KL}(p||q)=\sum\limits_xp(x)log(\frac{p(x)}{q(x)})=E_{p(x)}[log(\frac{p(x)}{q(x)})]
\\ 如果p(x)和q(x)分布相同，D_{KL}(p||q)=0
\\ D_{KL}(p||q) \ne D_{KL}(q||p),不具有对称性
\\ D_{KL}(p||q) \ge 0
$$
用来描述两个概率分布之间的差异，p与q之间的对数差在p上的期望。



**转换**
$$
D_{KL}(p||q)=\sum\limits_xp(x)log(\frac{p(x)}{q(x)})=\sum\limits_xp(x)logp(x)-\sum\limits_xp(x)logq(x)=-H(p)+H(p,q)=H(p,q)-H(p)
\\
D_{KL}(p||q)=H(p,q)-H(p)
$$
即当用假设分布q(x)得到的平均信息量期望减去真实分布p(x)得到的平均信息量期望的差（多出来的信息量）就是相对熵。如果$H(p)$为常量（训练集分布是固定不变的），则最小化KL散度等价与最小化交叉熵。

令P(real), P(train), P(model)分别代表真实数据分布，从真实分布中独立同分布采样的样本数据分布和模型假设的分布。我们的目标其实是期望 P(model)～P(real)，但真实分布不可知（我们只有观测值并不掌握生成的机制），所以退而求其次期望P(model)～P(train)。这里其实对P(train)也有要求，如果训练集采样有偏（数据集质量不行）也会导致模型的表现较差junk in junk out. 又由于当训练集确定了，P(train)也就确定了。所以求 $D_{KL}(p_{train}||p_{model})$等价于求$H(p_{train},p_{model})$,所以可以用交叉熵用来计算学习得到的模型分布和训练分布的差异，作为损失函数。





**MLE**

在对模型做参数估计的时候，可以利用最大化似然函数的思想
$$
\hat \theta = argmax_{\theta}\prod\limits_{i=1}^{N}q(y_i|x_i;\theta)
$$
等价于最小化负对数似然(NLL, nagetive-log-likelihood
$$
\hat \theta = argmin_{\theta}-\sum\limits_{i=1}^{N}log(q(y_i|x_i,\theta))
$$
