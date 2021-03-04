## 面试A&Q

> **零散问题**
>
> epoch和batch的设置意义
>
> 如何调整learning rate
>
> 孪生网络
>
> 分词，POS 
>
> word2vec fasttext glove
>
> 贝叶斯网络

> **Focal Loss**
>
> Focal Loss for Dense Object Detection

> **CNN/ResNet/Inception**

> **BatchNormalization/LayerNormalization**
> Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
>
> Layer Normalization

> **Overfitting&Underfitting**
>
> >  **Dropout**
>
> Dropout: A Simple Way to Prevent Neural Networks from Overfitting
>
> > **Early Stopping**
>
> > **weight penalties(L1/L2)**
>
> > **Soft Weight Sharing**
> >
> > Simplifying neural networks by soft weight-sharing
>
> >  **Gradient Vanishing & Exploding**
> >
> >  On the difficulty of training Recurrent Neural Networks
> >
> >  Understanding the exploding gradient problem
>
> >  **Generalization**

> **Activations**
>
> > ReLU/Softmax/LeakyReLU/PReLU/ELU/ThresholdedReLU

> **Metrics & Loss**
>
> > Probabilistic losses
> >
> > >  binary crossentropy/categorical crossentropy/sparse categorical crossentropy/poisson/kl divergence
> >
> > Regression losses
> >
> > > MSE/MAE/MAPE/MSLE/cosine similarity/huber/logCosh
> >
> > Hinge losses
> >
> > > Hinge/Squared Hinge/CategoricalHinge

> **Optimizers**
>
> https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9
>
> https://remykarem.github.io/blog/gradient-descent-optimisers.html
>
> > SGD/RMSprop/Adam/Adadelta/Adagrad/Adamax/Nadam/Ftrl

> **Initializers**
>
> > RandomNormal/RandomUniform/TruncatedNormal/Zeros/Ones/GlorotNormal/GlorotUniform/Identity/Orthogonal/Constant/VarianceScaling/Custom

> **Regularizers**
>
> > L1/L2/l1_l2/Custom

> **RNN/LSTM/GRU**
>
> [RNN/LSTM/GRU animation](https://towardsdatascience.com/animated-rnn-lstm-and-gru-ef124d06cf45)
>
> On the difficulty of training Recurrent Neural Networks

> **Model Distilling & Compression**
>
> Distilling the Knowledge in a Neural Network

> **Few Shot & Zero Shot**

> **CRF/MC/HMM**
>
> http://www.cs.columbia.edu/~mcollins/crf.pdf
>
> **MRF**
>
> > **Markov Random Field** undirected graphical model.
> >
> > 性质：
> >
> > 1. $G=(V,E)$ V顶点代表随机变量 ，E边代表随机变量间的依赖关系
> > 2. 大图可以分级为子图（clique/factors）集合。如果用一个分解函数$\phi_j$表示子图$D_j$。对于每一种子图而言$\phi_j(d_j)$都要严格为正。
> > 3. 子图中的节点是俩俩联通。子图的集合等于所有节点的集合。
> > 4. 对于如下MRF $V=(A,B,C,D)$ , 联合概率可以写成： $Pr(A=a,B=b,C=c,D=d)=\frac{\phi_1(a,b)\phi_2(b,c)\phi_3(c,d)\phi_4(d,a)}{\sum_{a'}\sum_{b'}\sum_{c'}\sum_{d'}\phi_1(a',b')\phi_2(b',c')\phi_3(c',d')\phi_4(d',a')}$
> >
> > $$ \begin{matrix} A&—&D\\ |&&|\\ B&—&C\\ \end{matrix} $$ 
>
> 
>
> > **misc**
> >
> > Gibbs Distribution 
> >
> > $\beta(d_j)=log(\phi(d_j))$
> >
> > Energy: $E(x)=-\sum_{j=1}^J\beta_j(d_j),\ d_j\subseteq X$
> >
> > Gibbs:  $P(x)=\frac{e^{-E(x)}}{Z}, where\ Z=\sum_{x'\subseteq X}e^{-E(x')}$
>
> 

> **Topic Model/Latent Dirichlet Allocation **

> **Decision Tree/Random  Forest/Gradient Boosting Tree/LGBM/XGBoost/CatBoost**

> **BERT**
>
> BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
>
> > 1. 为什么BERT在第一句前会加一个[CLS]标志？（[unusedXX],[SEP],[UNK],[MASK]）
> >
> >    > [CLS], [SEP], [MASK]这些特殊的占位符的设计主要还是依从任务设计而设置的。在原语料中引入这些占位符可以很方便设计目标函数。比如[CLS] token所对应的模型输出可以用于pretaining阶段的NSP任务也可用于finetune阶段的分类任务。[MASK]token是在MLM任务中用于预测target token id。[CLS]作为一个设定的占位符，经过各层attention layer的计算后，压缩了整个句子的语义信息，所以可以作为一个比较方便的分类特征使用。[CLS] token attend到了后面的每一个token，学到了语义表示。[SEP] token用于区分上下句。
> >
> > 2. BERT的三个Embedding直接相加会对语义有影响吗？
> >
> > 3. 在BERT中，token分3种情况做mask，分别的作用是什么？
> >
> >    > 构造MLM任务的训练数据是会mask掉一个句子中15%的token。替换策略是1. 80%的几率替换成[MASK]token用于训练语言模型；2. 10%几率替换成随机的token；3. 10%为原token
> >
> > 4. 为什么BERT选择mask掉15%比例的词，可以是其他比例吗？
> >
> > 5. 针对句子语义相似度/多标签分类/机器翻译/文本生成的任务，利用BERT结构怎么做finetuning？
> >
> >    > 下游任务的设计，魔改模型的任务输出。1. 句子相似度可以将待比较的句子A和B用[SEP]拼接起来一起送入BERT。可以用最后层的[CLS] token作为分类特征过一层sigmoid激活函数的Dense Layer。分类标签就是0/1是否相似。binary crossentropy作为损失函数，finetune整个BERT和分类层的权重。2. 文本直接送入BERT，可以用最后层的[CLS] token作为分类特征过一层softmax激活函数的Dense Layer。categorical crossentropy作为损失函数，finetune整个BERT和分类层的权重。（有关分类特征的获取还可以从多个角度入手，可以用最后的[CLS]作为分类特征，可以对最后一层的输出进行max/avg等操作，可以获取倒数多层的输出进行max/avg等操作并进行concate。或者将上述的特征统统进行拼接，用作分类特征。）3. 机器翻译任务：**TODO**   4. 文本生成任务： **TODO** 
> >
> > 6. BERT非线性的来源在哪里？multihead-attention是线性的吗？
> >
> > 7. BERT的输入是什么，哪些是必须的，为什么position-id不用给，type-id和attention-mask没有给定的时候，默认是什么样？
> >
> >    > 输入input_ids(必须), input_mask(1填充), token_type_ids(0填充) , attention-mask: Tensor of shape [batch_size, from_seq_length, to_seq_length]
> >
> > 8. BERT是如何区分一词多义的？
> >
> > 9. BERT训练时使用的学习率warm-up策略是怎么样的？为什么要这么做？
> >
> > 10. BERT采用哪种Normalization结构，LayerNorm和BatchNorm的区别，LayerNorm结构有参数吗，参数的作用是什么？
> >
> >     > Bert采用LayerNorm。**TBD** 
> >
> > 11. 为什么说ELMO是伪双向，BERT是真双向？产生这种差异的原因是什么？
> >
> >     > ELMo的语言模型LTR-LSTM和RTL-LSTM是同时独立训练的，将拼接得到的特征用于下游任务。BERT的Transformer结构每个token之间都彼此计算过attention，所以说是真双向语言模型。ELMo还是属于AR方式建模似然函数$p(\chi)=\prod\limits_{t=1}^{T}p(x_t|\chi_{<t})$，目标函数就限制了模型计算过程在time step上的依赖性。Bert是AE模型，通过重建掩码的句子，把无监督转换为自监督，目标函数不含方向限制的成分。
> >
> > 12. BERT和Transformer Encoder的差异有哪些？这些差异的目的是什么？
> >
> >     > 
> >
> > 13. BERT训练过程中的损失函数是什么？
> >
> >     > 预训练阶段的损失函数由MLM和NSP任务得到。MLM任务的损失函数是词表大小的categorical crossentropy，NSP任务用[CLS]token的输出作为分类特征，损失函数是binary crossentropy。总loss=batchavg(NSP_loss)+batchavg(MLM_loss)
> >     >
> >     > finetune阶段的损失函数随任务定义：分类问题可以是crossentropy
> >
> > 14. BERT的两个任务MLM和NSP是先后训练还是交替训练？
> >
> >     > MLM 任务和NSP任务是的loss直接相加，总的loss反向传播同时训练。

>**Transformer/Attention**
>
>Attention Is All You Need
>
>> 1. Transformer在哪里做了权重共享，为什么可以做权重共享？好处是什么？
>> 2. Transformer的点积模型做缩放的原因是什么？
>> 3. Transformer中是怎么做multihead-attention的，这样做multihead-attention会增加它的时间复杂度吗？
>> 4. 为什么Transformer要做multihead-attention？好处在哪？
>> 5. Transformer的Encoder和Decoder是如何进行交互的？和一般的seq2seq有什么差别？

>  **Self-Attention**
>
>  1. self-attention的本质是什么？包括哪几个步骤？和普通Attention的本质差异在哪里？
>
>  2. 不考虑多头的原因，self-attention中词向量不乘QKV参数矩阵，会有什么问题？
>
>  3. 在普通attention中，一般有k=v，那self-attention中这样可以吗？
>
>  4. self-attention在计算过程中，如何对padding位做mask？
>
>  5. bert的mask为何不学习transformer在attention出进行屏蔽score的技巧？
>
>  6. XLNet为什么不直接在attention掩码矩阵中只把当前的单词掩盖来获取上下文信息？直接mask往左上到右下的对角线构建双向语言模型不行吗？

> **BERT家族**
>
> > RoBerta
> >
> > ERINE
> >
> > Electra
> >
> > T5
> >
> > BART 
> >
> > XLNet
> >
> > > 1. XLNet是如何实现在不加[MASK]的情况下利用上下文信息的呢？
> > >
> > > 2. XLNet为什么要用双流注意力？两个流的差别是什么？分别的作用是什么？分别的初始向量是什么？
> > >
> > > 3. 虽然不需要改变输入文本的顺序，但XLNet通过PLM采样输入文本的不同排列去学习，这样不会打乱或者丢失词汇的时序信息吗？
> > >
> > > 4. AutoRegressive（AR）和AutoEncoder（AE）这两种模式分别是怎样的，各自的优缺点是什么，XLNet又是怎样融合着两者的？
> > >
> > >    > AR是利用条件概率对序列进行建模，以最大化似然函数$p(\chi)=\prod\limits_{t=1}^{T}p(x_t|\chi_{<t})$作为目标（反向公式稍变）。AE是通过重建被掩码的句子，建模目标不包含限制方向的成分。AR不能构建深度的双向语境（正反拼接只是浅层的语境），而下游任务又往往需要双向语境，比如MRC。AE中引入MLM任务，[MASK] token造成了在pretrain和finetune阶段输入数据上的差异。只预测[MASK]token不去对token序列的联合概率建模，弱化了NL序列中普遍存在的长程依赖的性质。
>
> >  ALBERT
> >
> >  https://arxiv.org/abs/1909.11942
> >
> >  Tricks: 1. factorized embedding parameters; 2. cross-layer parameter sharing; 3. self-supervised loss for sentence-order prediction(SOP)
> >
> >  > 1. ALBERT的小具体小在哪里？对实际储存和推理有帮助吗？
> >  >
> >  >    > 压缩参数的处理：1.分解embedding table，把原来的$M_{V\times H}$变成$M_{V\times E} \cdot M_{E\times H}$
> >  >
> >  > 2. BERT的NSP为什么被认为是没用的？ALBERT采样的SOP（sentence order prediction）任务是怎么样的？相比NSP有什么优势？
> >  >
> >  >    > NSP任务相较于MLM任务过于简单，NSP任务的训练数据只用ISNEXT/NOTNEXT这种构造方式，可能就是把AB句子的topic和coherence一起打包建模了。句子的topic信息其实较为容易见面，从而导致NSP任务对coherence的信息学习到的较少。SOP任务主要考察的是coherence，构建数据时正例就是连续的AB句子，负例是颠倒AB顺序变成BA。强制模型关注句子间的连贯性，从而带来模型性能的提升。优势虽然ALBERT没有NSP任务，但是SOP任务在做NSP任务时也有较好的得分78（预SOP-测NSP）：90（NSP-NSP），但是NSP模型在做SOP任务是表现就和瞎猜差不多了。52（NSP-SOP）：86（SOP-SOP），同时下游的相关任务也有提升。
> >
> >  DistilBERT

> **工具框架**
>
> Tensorflow/Pytorch/Keras/Keras4bert/sklearn/Gensim/SpaCy

> **Ensamble learning**
>
> > Bagging/Boosting/Stacking

> **Decoding Strategies**
>
> > Verbit/TopK/BeamSearch/Sampling

> **MLE/MAP**
>
> $\hat{\theta}_{MLE}=argmax_{\theta}(P(D|\theta))$
>
> $\hat{\theta}_{MAP}=argmax_{\theta}(P(\theta|D))=argmax_{\theta}(\frac{P(D|\theta)P(\theta)}{P(D)})=argmax_{\theta}(P(D|\theta)P(\theta))$

> **AR&AE**
>
> AutoRegression LM: $max_{\theta \:}logp_{\theta}(\chi)=\sum\limits_{t=1}^{T}logp_{\theta}(x_t|\chi_{<t})=\sum\limits_{t=1}^{T}log\frac{exp(h_{\theta}(\chi_{1:t-1}))^{\top}e(x_t)}{\sum\limits_{x'}exp(h_{\theta}(\chi_{1:t-1}))^{\top}e(x'))}$ 
>
> $\chi$代表token序列，$x_t$代表t步的token，$\chi_{<t}$代表t步之前的token序列，$h_{\theta}(\chi_{1:t-1})$代表t步之前序列传到的hidden state，$e(x_t)$ t步token对应的词向量， $e(x‘)$ 词表里某个token对应的词向量。
>
> AutoEncoding LM: $max_{\theta \:}logp_{\theta}(\bar{\chi}|\hat{\chi}) \approx \sum\limits_{t\in M}logp_{\theta}(x_t|\hat{\chi})= \sum\limits_{t\in M}log\frac{exp(H_{\theta}(\hat{\chi})^{\top}_t e(x_t))}{\sum\limits_{x'}exp(H_{\theta}(\hat{\chi})^{\top}_t e(x'))}$ 
>
> $\chi$代表原token序列， $\hat{\chi}$代表有掩码的序列，$\bar{\chi}$代表序列中被掩码的M个tokens，

