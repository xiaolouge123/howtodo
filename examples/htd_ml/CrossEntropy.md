对于分类任务**CE**这个损失函数的细节

ref: https://www.programmersought.com/article/94902040914/

https://peterroelants.github.io/posts/cross-entropy-softmax/

https://peterroelants.github.io/posts/cross-entropy-logistic/

概率类型的loss

计算图
$$
\begin{matrix} 
\\ s & \rarr    & p & \rarr    & o
\\   &          &   & \nearrow &
\\   &          & t &          &
\end{matrix}
\tag{figure 1}
$$


sigmoid & softmax activation function
$$
Sigmoid\quad f(s_i)=\frac{1}{1+exp(-s_i)}
\\
Softmax\quad f(s_i)=\frac{exp(s_i)}{\sum\limits_{j=1}^{K}exp(s_j)}, i=1...C
$$
model -> logits -> act_fn -> score $s_i$

CrossEntropy的定义
$$
CE=-\sum\limits_{i}^{C}t_ilog(s_i)
$$
这里的$t_i$就是真实的标签，$s_i$是对应类$i\in C$的得分。

**BinaryCrossEntropy**在二分类的任务中，loss可以写成$CE=-\sum\limits_{i=1}^{C^{'}=2}t_ilog(s_i)$，两种情况带入就有$CE=-t_1log(s_1)-t_2log(s_2)=-t_1log(s_1)-(1-t1)log(1-s_1),t_1=1,s_1=\frac{1}{1+exp(-x)},s_2=\frac{exp(-x)}{1+exp(-x)}$

**CategoricalCrossEntropy**在多分类任务中，模型的输出是对标签集的概率分布，即对某个标签$i, score=f(s)_i=\frac{exp(x_i)}{\sum\limits_{j}^{C}exp(x_j)}$，这样模型输出的logit $ x$对应各个类别的score即是 $f(s)_i\in\mathbb{R}, i\in \{1...C\}$，同时对与样本$i$，它的标签$t_i=[0,0,0...t_p=1...,0,0,0],p\in\{1...C\}$是one-hot的形式，所以$t_ilog(s_i)$是向量结果，然后C个


$$
稳定的softmax与溢出
\\ p_i=\frac{exp(s_i)}{\sum\limits_{j=1}^{K}exp(s_j)}, i=1...K
\\但是考虑到数据大小的限制，比如python中float64最高支持到e^{308}，否则就会越界inf
\\因此解决方案是用max(s_i)作为base进行s_i的平移这样s_i'=s_i-max(s_i)\le 0
\\s_i'=s_i-max(s_i) \quad p_i=\frac{exp(s_i-s_i')}{\sum\limits_{j=1}^{K}exp(s_j-s_i')}, i=1...K
$$

```python
def softmax(x):
  exps = np.exp(x)
  return exps/np.sum(exps)

def stable_softmax(x):
  exps = np.exp(x-np.max(x))
  return exps/np.sum(exps)
```

探索一下softmax的求导过程
$$
\frac{\partial p}{\partial s}|^{softmax}
\\这是向量对向量的求导有
\\ \frac{\partial p_i}{\partial s_j}|^{softmax}_{s_1...s_C}=\frac{\partial \Bigg(\frac{e^{s_i}}{\sum\limits^{C}_{j=1}e^{s_j}}\Bigg)}{\partial s_j}=\frac{\partial \Bigg(\frac{g(s_i)}{h(s_1...s_C)}\Bigg)}{\partial s_j}
\\ =\frac{\frac{\partial g(s_i)}{\partial s_j}h(s_1...s_C)-g(s_i)\frac{\partial h(s_1...s_C)}{\partial s_j}}{h(s_1...s_C)^2}
\\ =
\begin{cases}
\frac{e^{s_i}\sum\limits^{C}_{j=1}e^{s_j}-e^{s_i}e^{s_j}}{(\sum\limits^{C}_{j=1}e^{s_j})^2}=
\frac{e^{s_i}(\sum\limits^{C}_{j=1}e^{s_j}-e^{s_j})}{(\sum\limits^{C}_{j=1}e^{s_j})^2}=
\frac{e^{s_i}}{\sum\limits^{C}_{j=1}e^{s_j}}\times \frac{(\sum\limits^{C}_{j=1}e^{s_j}-e^{s_j})}{\sum\limits^{C}_{j=1}e^{s_j}}=
p_i(1-p_j)=p_i(1-p_i) 
,\quad i=j\\
\frac{0 \sum\limits^{C}_{j=1}e^{s_j}-e^{s_i}e^{s_j}}{(\sum\limits^{C}_{j=1}e^{s_j})^2}=
\frac{-e^{s_i}e^{s_j}}{(\sum\limits^{C}_{j=1}e^{s_j})^2}=
-\frac{e^{s_i}}{\sum\limits^{C}_{j=1}e^{s_j}}\times \frac{e^{s_j}}{\sum\limits^{C}_{j=1}e^{s_j}}=
-p_ip_j
, \quad i\ne j
\end{cases}
\\
J^{s\rarr p}=\frac{\partial p}{\partial s}|^{softmax}=
\left[
\begin{matrix}
\frac{\partial p_1}{\partial s_1} & \dots & \frac{\partial p_1}{\partial s_C} \\
\vdots & \ddots & \vdots \\
\frac{\partial p_C}{\partial s_1} & \dots & \frac{\partial p_C}{\partial s_C}
\end{matrix}
\right]
=\left[
\begin{matrix}
p_1(1-p_1)  & -p_1p_2     & \dots  & -p_1p_{c-1}        & -p_1p_{c}\\
-p_2p_1     & p_2(1-p_2)  & \dots  & -p_2p_{c-1}        & -p_2p_{c}\\
\vdots      & \vdots      & \ddots & \vdots             & \vdots   \\
-p_{c-1}p_1 & -p_{c-1}p_2 & \dots  & p_{c-1}(1-p_{c-1}) & -p_{c-1}p_{c}\\
-p_{c}p_1   & -p_{c}p_2   & \dots  & -p_{c}p_{c-1}      & p_{c}(1-p_{c})
\end{matrix}
\right]
$$
交叉熵的部分
$$
o = H(t,p)=-\sum\limits_i t_ilog(p_i)
$$

```python
def cross_entropy(x,y):
  '''
  x.shape = [batch, num_cls]
  y.shape = [batch, 1]
  '''
  n = y.shape[0]
  p = softmax(x, axis=-1)
  nagtive_log_likehood = - np.log(p[range(n), y]) # lookup p by y as index for each sample
  loss = np.sum(nagtive_log_likehood) / n
  return loss
```

交叉熵的求导部分
$$
\frac{\partial o}{\partial s_i}|_{s_1...s_C}
=\frac{\partial o}{\partial p_i}|_{p_1...p_C}\frac{\partial p_i}{\partial s_i}|_{s_1...s_C}
=\frac{\partial \big( -\sum\limits_i t_ilog(p_i) \big)}{\partial p_i}\times \frac{\partial p_i}{\partial s_i}
\\ =-\sum\limits^{C}t_i\frac{\partial log(p_i)}{\partial p_i}\times \frac{\partial p_i}{\partial s_i}
\\ =-\sum\limits^C t_i\frac{1}{p_i}\times \frac{\partial p_i}{\partial s_i}
=\bigg [-\sum\limits^C t_i\frac{1}{p_i}\times \frac{\partial p_i}{\partial s_1},-\sum\limits^C t_i\frac{1}{p_i}\times \frac{\partial p_i}{\partial s_2},...,-\sum\limits^C t_i\frac{1}{p_i}\times \frac{\partial p_i}{\partial s_C} \bigg]^T
\\ 这个部分\frac{\partial p_i}{\partial s_i}讨论参考softmax求导结果
\\ CE对logit\; s的偏导可以写成:
\\ \frac{\partial o}{\partial s_i}|^{f}_{s_1...s_C}=
\sum
\begin{cases}
- t_i\frac{1}{p_i}\times p_i(1-p_j)= - t_i\times (1-p_j), \quad j = i
\\
-\sum\limits^C_{j \ne i} t_i\frac{1}{p_i}\times (-p_ip_j)= \sum\limits^C_{j \ne i} t_i\times p_j, \quad j \ne i
\end{cases}
\\单拿一个出来举例 \frac{\partial o}{\partial s_1}=-\sum\limits^C t_i\frac{1}{p_i}\times \frac{\partial p_i}{\partial s_1}=-\bigg( \big(t_1\frac{1}{p_1}\times p_1(1-p_1) \big )_{i=1} + \sum\limits^C_{i\ne 1}\big(t_i\frac{1}{p_i}\times (-p_ip_1)\big) \bigg)=-\big(t_1-t_1p_1 - \sum\limits^C_{i\ne 1}(t_ip1)\big)
\\ = -t_1 + \sum\limits^C t_ip_1 = -t_1 + p_1
\\故：
\\ \frac{\partial o}{\partial s_i}|^{f}_{s_1...s_C}=\bigg[p_1 - t_1, p_2 - t_2,..., p_C - t_C \bigg]^T, \quad t_i = 1 \; if \; label \; is \; i \; else \; t_i=0
$$

```python
def delta_cross_entropy(x,y):
  '''
  x.shape = [batch, num_cls]
  y.shape = [batch, 1] y is in {1,2,3, ..., C}
  '''
  n = y.shape[0]
  p = softmax(x, axis=-1)
  p[range(n), y] -= 1
  grad = p/n
  return grad
```

