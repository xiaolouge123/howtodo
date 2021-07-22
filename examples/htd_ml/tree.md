# Machine Learning

ref：《统计学习方法》

## 决策树

>  框架：特征选择 $\Rightarrow$ 决策树生成 $\Rightarrow$ 决策树剪枝

利用信息增益、信息增益比这类指标作为指导筛选特征

期望叶节点的成员有更高的纯度，再决定属性划分时就有不同的划分标准。

- 信息增益 information gain （ID3）
- 信息增益率 gain ratio （C4.5）
- 基尼指数 Gini Index（CART）

以上指标可以用来度量分割后的成员的纯度或不确定性(熵)



**Background**

熵 $H(X)=-\sum_{i=0}^{n}{p(X=x_i)logp(X=x_i)}$

条件熵$H(Y|X)=\sum_{i=1}^{n}p(X=x_i)H(Y|X=x_i)$

对于一个数据集合D来说，样本中有k个类别。每个类别的经验概率是$\frac{|C_k|}{|D|}$, $|C_k|$ 代表k类的样本个数 ，$|D|$ 代表数据集合的样本总数。

对于数据集合D的经验熵就是：

$H(D)=-\sum_{k=1}^{K}\frac{|C_k|}{|D|}log(\frac{|C_k|}{|D|})$

- **ID3**

  特征选择标准

  - 假设取特征A作为划分特征。可以对数据集合D求(决策树学习中的信息增益等价于数据集中类与特征的互信息)

  - $g(D,A)=H(D)-H(D|A)$

  - 数据集中有K个类别，特征A有$\{a_1, a_2, ..., a_n\}$种取值，则可以根据A特征的取值将数据进一步划分为$\{D_1,D_2,...,D_n\}$，在$D_i$中属于$C_k$的样本集为$D_{ik}$，则有

  - $H(D|A)=\sum_{i=1}^{n}\frac{|D_i|}{|D|}H(Di|A=a_i)=-\sum_{i=1}^{n}\frac{|Di|}{|D|}\sum_{k=1}^{K}\frac{|D_{ik}|}{|D_i|}log(\frac{|D_{ik}|}{|D_i|})$

  - 对于所有的特征都可以计算个$g(D|A)$，选个信息增益最大的特征作为当前步骤的划分特征

  生成流程：

  - 在决策树各个节点上利用信息增益准则选择特征，递归的构建决策树。具体是：从根节点开始，对节点计算所有可能特征的信息增益，选择信息增益最大的特征作为该节点的特征，由该特征的不同取值建立子节点，在对子节点递归调用以上过程，直到信息增益很小或者没有特征可用停止。

- **C4.5**

  特征选择标准

  - $g_R(D,A)=\frac{g(D,A)}{H_A(D)}$

  - 这个算法算是对ID3的优化。再考虑用特征A作为分割特征的同时，还要考虑A取值的情况。$H_A(D)=-\sum_{i=0}^{n}\frac{|D_i|}{|D|}log(\frac{|D_i|}{|D|})$ , $|D_i|$代表特征A取$a_i$的样本数。

  生成流程：

  - 和ID3类似

过拟合的问题：上述两种算法对于训练集可能表现不错，但泛化个能不好。决策树过于复杂，很适配训练集的结果。

这个时候需要对决策树进行简化，**剪枝（pruning）**

最小化决策树的loss function

loss怎么定义？

假设决策树T有$|T|$个叶节点，每个叶节点t又有$N_t$个样本，其中$N_{tk}$为该叶节点上属于k类的样本数。$k=1,2,...,K$，定义$H_t(T)$为叶节点t上的经验熵，$\alpha\geq0$，则决策树的loss可以定义成：

$C_{\alpha}(T)=\sum_{t=i}^{|T|}N_tH_t(T)+\alpha|T|$

经验熵： $H_t(T)=-\sum_{k=1}^{K}\frac{N_{tk}}{N_t}log{\frac{N_{tk}}{N_t}}$

令：$C(T)=\sum_{t=i}^{|T|}N_tH_t(T)=-\sum_{t=i}^{|T|}\sum_{k=1}^{K}N_{tk}log{\frac{N_{tk}}{N_t}}$

损失函数可以化简为： $C_{\alpha}(T)=C(T)+\alpha|T|$

$C(T)$：可以表示模型对训练数据的拟合效果，越小越好，但是模型的复杂度会升高。

$\alpha|T|$：代表模型的复杂度，叶节点分的越细，越能过拟合训练数据，复杂度就会升高。

剪枝就是在$\alpha$确定的情况下，选择损失函数最小的模型。

- 剪枝过程：
  - 输入：生成的决策树T和$\alpha$；输出：剪枝后的子树$T_{\alpha}$
    1. 计算每个节点的经验熵
    2. 递归的从叶节点向上回溯；设原来树为$T$，子节点归并到上一父节点后的树是$T^{'}$。如果这两棵树的损失函数有 $C_{\alpha}(T^{'})\leq C_{\alpha}(T)$，则进行剪枝，退回到父节点。
    3. 重复2，直到不能进行下去为止。

- **CART**
  - 思路：对$P(Y|X)$建模，每个节点对特征做二分（选取特征A的某个取值$a_i$，等价于递归地二分每个特征，将数据空间划分为有限个单元，在这些单元上确定预测的概率分布。
  - 步骤：
    1. 生成：训练集生产，生成的树要尽量大（什么叫大？怎么保证大？均分某个特征的空间，保证左右两子节点样本数均衡？）
    2. 剪枝：用验证集数据进行剪枝，保证损失函数最小。
  - CART生成
    - 筛选特征准则，对于回归树用Square Error指标，对于分类树用Gini index
      - 回归树：rv： X， Y， Y连续。对于训练集$D=\{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}$

$Gini(p)=\sum_{k=1}^{K}p_k(1-p_k)=1-\sum_{k=1}^{K}p_k^2$

- $p_k$ 样本属于k类别的概率， $(1-p_k)$ 样本不是k类的概率
- 这个经验概率$p_k$可以用频率来计算，这样就有$Gini(D)=1-\sum_{k=1}^{K}(\frac{|C_k|}{|D|})^2$
- 由于Gini考虑的是对给定特征给定样本是否分类正确，即2项分类。所以如果有一个特征A，有$a_1,a_2,a_3$，三种取值。所以有三个划分选择。$Gini_{index}(D,A)=\sum_{i=1}^{3}\frac{|D_{a_i}|}{|D|}Gini({D_{a_i}})$



### Ensemble Learning

> 学习一系列弱学习器，然后通过一种结合策略把弱学习器集合成一个强学习器。
>
> 1. 弱学习器可以是同质的（都是树模型或者都是神经网络），也可以是异质的（SVM，LR，NB模型的组合）
> 2. 对于同质的弱学习器，按学习器之间是否有依赖关系可以分为bagging&boosting。

#### Boosting

ref：https://www.cnblogs.com/pinard/p/6133937.html

>  框架：
>
> 1. 先从训练集用初始权重训练一个弱学习器1。
> 2. 用弱学习器1的学习误差率来更新训练集样本的权重（误差率高的样本权重变高，后续更多关注。
> 3. 从训练集用更新权重训练一个弱学习器2，迭代到2步。
> 4. 直到弱学习器的个数达到目标T，最终通过策略集合T个弱学习器，得到强学习器。

需要解决的问题：

1. 如何计算学习误差率$e$
2. 如何得到弱学习器权重系数$\alpha$
3. 如何更新样本权重$D$
4. 选择什么结合策略

>  **Adaboost**
>
> 训练集 $T = \{(x_1,y_1), (x_2,y_2), (x_3,y_3), ..., (x_m,y_m)\}$
>
> 在第k个弱学习器对训练集的输出权重为$D_k = (w_{k1}, w_{k2}, ...,w_{km})$ ，初始训练样本权重$D_1 = (w_{11}, w_{12}, ...,w_{1m}); w_{1i} = \frac{1}{m}, i\in{1,2,3,..,m}$
>
> 

> **GBDT**

> **XGBoost**

> **LightGBM**

> **CatBoost**

#### Bagging

ref: https://www.cnblogs.com/pinard/p/6156009.html

> 框架：
>
> 1. 从训练集随机采样（有放回；bootstrap sampling）m个样本作为弱学习器的训练集。
> 2. 采样T次，得到T个采样集，分别训练T个弱学习器。
> 3. 通过策略集合T个弱学习器，得到强学习器。

> **Random Forest**



#### 集成策略

如何组合学习到的T个学习器$\{h_1,h_2, ... ,h_T\}$

> **平均法**
>
> $H(x)=\frac{1}{T}\sum\limits_{i=1}^{T}h_i(x)$
>
> $H(x)=\sum\limits_{i=1}^{T}w_ih_i(x); \sum w_i = 1 ; w_i \ge0$

> **投票**
>
> top1投票
>
> top1投票还要过半
>
> 加权投票

> **学习法**
>
> stacking
>
> 对于弱学习器$\{h_1,h_2, ... ,h_T\}$的输出作为stacking模型的输入，用训练集的输出做为stacking模型的输出，在训练个次级学习器。
>
> 模型的选择可以直接LR走起。



> **Tree Ensemble Model**
> $$
> D= \{(\bold{x}_i, y_i)\}(|D|=n, \bold{x}_i \in \mathbb{R}^m, y_i\in\mathbb{R})
> \\ use \; K \; additive \; functions \; to \; predict \; output
> \\
> \hat{y}_i = \phi(\bold{x}_i)=\sum\limits_{k=1}^{K}f_k(\bold{x}_i), f_k\in\psi
> \\
> \psi=\{f(\bold{x})\} = w_{q(\bold{x})}(q:\mathbb{R}^m \rarr T, w\in \mathbb{R}^T)
> \\T\; is\; the\; number\; of\; leaves\; in\; the\; tree.
> \\ Each\; f_k\; corresponds \; to\; an\; independent\; tree\; structure\; q \;and \;leaf\; weights\; w
> \\regularized \; objectives
> \\
> L(\phi) = \sum\limits_i l(\hat{y}_i, y_i) + \sum\limits_k\Omega(f_k)
> \\Here\; l\; is\; a\; differentiable\; convex\; loss\; function\;
> \\
> \Omega(f_k) = \gamma T+\frac{1}{2}\lambda||w||^2
> $$
> **Gradient Tree Boosting**
>
> 