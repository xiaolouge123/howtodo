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
> 

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
> > 4. 为什么BERT选择mask掉15%比例的词，可以是其他比例吗？
> >
> > 5. 针对句子语义相似度/多标签分类/机器翻译/文本生成的任务，利用BERT结构怎么做finetuning？
> >
> >    > 下游任务的设计，魔改模型的任务输出。1. 橘子
> >
> > 6. BERT非线性的来源在哪里？multihead-attention是线性的吗？
> >
> > 7. BERT的输入是什么，哪些是必须的，为什么position-id不用给，type-id和attention-mask没有给定的时候，默认是什么样？
> >
> > 8. BERT是如何区分一词多义的？
> >
> > 9. BERT训练时使用的学习率warm-up策略是怎么样的？为什么要这么做？
> >
> > 10. BERT采用哪种Normalization结构，LayerNorm和BatchNorm的区别，LayerNorm结构有参数吗，参数的作用是什么？
> >
> > 11. 为什么说ELMO是伪双向，BERT是真双向？产生这种差异的原因是什么？
> >
> > 12. BERT和Transformer Encoder的差异有哪些？这些差异的目的是什么？
> >
> > 13. BERT训练过程中的损失函数是什么？
> >
> > 14. BERT的两个任务MLM和NSP是先后训练还是交替训练？

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
> > > 2. XLNet为什么要用双流注意力？两个流的差别是什么？分别的作用是什么？分别的初始向量是什么？
> > > 3. 虽然不需要改变输入文本的顺序，但XLNet通过PLM采样输入文本的不同排列去学习，这样不会打乱或者丢失词汇的时序信息吗？
> > > 4. AutoRegressive（AR）和AutoEncoder（AE）这两种模式分别是怎样的，各自的优缺点是什么，XLNet又是怎样融合着两者的？
>
> >  ALBERT
> >
> >  https://arxiv.org/abs/1909.11942
> >
> >  Tricks: 1. factorized embedding parameters; 2. cross-layer parameter sharing; 3. self-supervised loss for sentence-order prediction(SOP)
> >
> >  > 1. ALBERT的小具体小在哪里？对实际储存和推理有帮助吗？
> >  > 2. BERT的NSP为什么被认为是没用的？ALBERT采样的SOP（sentence order prediction）任务是怎么样的？相比NSP有什么优势？
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
>
> 