分词技术的演进

1. 基于辞典的方法（unknown words problem）

   https://medium.com/@phylypo/nlp-text-segmentation-using-dictionary-based-algorithms-6d0a45a76c08

   1. 最大匹配（Maximal matching）
   2. 双向最大匹配
   3. 最大匹配（Maximum matching）

2. 基于N-gram的方法

   https://medium.com/@phylypo/nlp-text-segmentation-with-ngram-b5506dbb514c

   segment the sequence of any possible results, choose the biggest likelihood of the input sequence possible tokens combination

   1. unigram

      $p(w_{1:n})=\prod\limits_{k=1}^n p(w_k)$

   2. bigram

      add <b> before sequence and <e> after sequence

      $p(w_{1:n})=\prod\limits_{k=1}^n p(w_k|w_{k-1})$ 

   3. ngram

      same way.

3. 基于Naive Bayes的方法(每个char-label pair之间条件独立)

   https://medium.com/@phylypo/nlp-text-segmentation-using-naive-bayes-bccdd08ccf6f

   character level sequence labeling, **SBME** as tags

   $p(y|\vec x) \propto p(y,\vec x) = \prod\limits_{i=1}^m p(x_i|y)p(y)$ 
   $$
   \begin{matrix}
   t & h & i & s & i & s & a & t & e & s & t \\
   \darr & \darr & \darr & \darr & \darr & \darr & \darr & \darr & \darr & \darr & \darr \\
   B & M & M & E & B & E & S & B & M & M & E
   \end{matrix}
   $$
   

4. 基于HMM的方法

   https://medium.com/@phylypo/nlp-text-segmentation-using-hidden-markov-model-f238743d87eb

   $p(x,y) = \prod\limits_{t=1}^T p(x_t|y_t)p(y_t|y_{t-1})$
   $$
   \begin{matrix}
   <b> & \rarr & t & \rarr & h & \rarr & i & \rarr & s & \rarr & i & \rarr & s & \rarr & a & \rarr & t & \rarr & e & \rarr & s & \rarr & t & \rarr & <e> \\
    & & \darr & & \darr & & \darr & & \darr & & \darr & & \darr & & \darr & & \darr & & \darr & & \darr & & \darr \\
   & & B & & M & & M & & E & & B & & E & & S & & B & & M & & M & & E \\
   \end{matrix}
   $$
   解码方法：viterbi algo

   每一个step取概率最大的结果

   shortcoming:1.HMM只捕捉到相邻状态间和对应观测值的依赖，没办法获得全部领域的信息。2.HMM建模目标是$P(Y,X)$，但是预测时，我们需要的是$P(Y｜X)$。这就导致解码方法的结果还是有偏差。

5. 基于MEMM的方法

   https://medium.com/@phylypo/nlp-text-segmentation-using-maximum-entropy-markov-model-c6160b13b248

   maximum likelihood

   maximum conditional likelihood

   log-linear model

   maximum entropy model

   maximum entropy markov model
   $$
   P_w(y|x) = \frac{1}{Z_w(y_{i-1},x)}exp(\sum_{i=1}^n w_if_i(y_i,y_{i-1},x))
   \\
   Z_w(y_{i-1},x) = \sum_{y\in Y}exp(\sum_{i=1}^n w_if_i(y_i,y_{i-1},x))
   $$
   

   **MEMM models the dependencies between each state and the full observation sequences x.**

6. 基于CRF的方法

   https://medium.com/@phylypo/nlp-text-segmentation-using-conditional-random-fields-e8ff1d2b6060
   $$
   P_w(y|x) = \frac{1}{Z_w(x)}exp(\sum_{j=1}^n\sum_{i=1}^m w_if_i(y_{j-1},y_j,x,j))
   \\
   Z_w(x) = \sum_{y\in Y}exp(\sum_{j=1}^n\sum_{i=1}^m w_if_i(y_{j-1},y_j,x,j))
   $$
   

