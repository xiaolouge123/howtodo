### NN和RNN的实现

https://peterroelants.github.io/posts/neural-network-implementation-part01/

https://peterroelants.github.io/posts/rnn-implementation-part01/

#### 1. Gradient Descent

$$
以简单的线性回归举例
\\y=W\cdot x\quad y\in\mathbb{R}^m\quad x\in\mathbb{R}^k
\\target : t \in\mathbb{R}^m
\\计算图：
$$

$$
\begin{matrix} 
\\ x & \longrightarrow & y & \longrightarrow & o
\\   & \nearrow        &   & \nearrow        & 
\\ w &                 & t &                 &  
\end{matrix}
\tag{figure 1}
$$

$$
Define:\quad MSE:\; o=\frac{1}{N}\sum\limits^{N}_{i=1}||t_i-y_i||^2\quad N:sample \; number
$$

全局Jacobian $J : \frac{\partial o}{\partial W} $ 在k+1次迭代时$w_{k+1}=w_k - \bigtriangleup w_k=w_k - \alpha\cdot\frac{\partial o}{\partial W}$

$\alpha 是学习率，由链式法则 \frac{\partial o_i}{\partial W}=\frac{\partial o_i}{\partial y_i}\frac{\partial y_i}{\partial W}$

$\frac{\partial o_i}{\partial y_i} = \frac{\partial (t_i-y_i)^2}{\partial y_i}=-2(t_i-y_i)=2(y_i-t_i)$

$\frac{\partial y_i}{\partial W}=\frac{\partial (x_i\cdot W)}{\partial W}=x_i$

则有：$\bigtriangleup w=\alpha\cdot\frac{\partial o}{\partial W}=\alpha\cdot 2x_i(y_i-t_i)\quad 为简化公式，x，y，w都是标量$

则一次batch训练：$\bigtriangleup w=\alpha\cdot 2\frac{1}{N}\sum\limits^{N}_{i=1}x_i(y_i-t_i)$

然后就是不停batch更新迭代

#### 2. 分类问题

$$
\begin{matrix} 
\\ x   & \rarr    & z & \rarr & y & \rarr    & o
\\     & \nearrow &   &       &   & \nearrow &
\\ w   &          &   &       & t &          &
\end{matrix}
\tag{figure 2}
\\ x\in\mathbb{R}^2 \quad t\in\{0,1\}
\\ z,y\; are \; scalar
\\ z = x\cdot w^T
\\ y = \sigma(z) = \frac{1}{1+e^{-z}}
\\ crossentropy: \quad o=-\sum\limits^{\{0,1\}}_{1...N}tlog(y)
$$

