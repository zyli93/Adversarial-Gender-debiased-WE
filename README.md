# Adversarial Gender-debiased Word Embedding
This is the repository of Fall 2017 CS269 Project, _Adversarial Gender-debiased Word Embedding_. The honored group members are Yichao Zhou, Jieyu Zhao, and Zeyu Li.

## Abstract
Word embedding has shown the ability to boost the performance in many NLP tasks such as  syntactic parsing and sentiment analysis. Despite its promising performance, word embedding  runs the risk of discovering and exploiting the gender stereotype problem. Our work aims at using the adversarial techniques to help reduce the gender bias in  current word embedding.

### Word Embedding
Word Embedding is an emerging technique for learning word representations in Natural Language Processing (NLP) tasks. It uses the context of a certain word to learn its representation as well as the those of the contextual words, i.e. words present within the ``context'' window. The learned vectorization of words has following properties. First, similar words are projected closely in the semantic space. Such proximity can be evaluated mathematically such as Cosine Similarity and Euclidean distance.

The first implementation was proposed in (Mokolov et al., 2013).The authors describe two basic models, Skip-gram and Continuous Bag-of-Words (CBOW), to learn the vectorization of words. In order to enhance the learning efficiency, Hierarchical Softmax and Negative Sampling strategies are utilized. Skip-gram aims at maximizing the probability of context words given the input word, i.e. the word in the middle of the sliding window. While CBOW functions the opposite way, context words are given to maximize the probability of the middle word.

### Bias in Word Embedding
Word embedding can be used in various applications, such as curriculum vitae parsing or machine translation. Words with similar semantic meanings will be mapped to close space. It can be used to show the relationship between different words, such as $\overrightarrow{man} - \overrightarrow{woman} \approx \overrightarrow{king} - \overrightarrow{queen}$. Such linearity can be help in many applications such as  sentiment analysis and question retrieval (Bolukbasi et al., 2016).

However, such relationship can sometimes reflect the stereotype, such as $\overrightarrow{man} - \overrightarrow{woman} \approx \overrightarrow{computer\ programmer} - \overrightarrow{homemaker}$ (Bolukbasi et al., 2016). When people build some applications such as the curriculum vitae filtering system based on such word embedding, it will incorrectly rank  male candidates ahead female ones based on the gender rather than their abilities. In our work we want to build a gender neutral word embedding to reduce such stereotype. For simplicity, in this work, we only consider the binary gender bias stereotype. However, the method we proposed can be adopted to other bias such as race or rage bias. 

### Adversarial Learning
Learning meaningful word representations that maintain the content necessary for a particular task while filtering away detrimental variations is a problem of great interest in natural language processing. In this project, we tackle the problem of learning word representations invariant to gender bias, leading to better generalization. inspired by the recent advancement of adversarial learning (Goodfellow et al., 2014), we formulate the word embedding representation learning as a minimax game among two players: an encoder which maps the observed text data deterministically into a feature space and a discriminator which looks at the word representation and tries to identify gender bias we hope to eliminate from the feature. Afterwards, we provide an intuitive interpretation through the analogy generation task.

## Related Work
1. Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeff Dean. "Distributed representations of words and phrases and their compositionality." In Advances in neural information processing systems, pp. 3111-3119. 2013.
2. Xie, Qizhe, Zihang Dai, Yulun Du, Eduard Hovy, and Graham Neubig. "Controllable Invariance through Adversarial Feature Learning." arXiv preprint arXiv:1705.11122 (2017).
3. Bolukbasi, Tolga, Kai-Wei Chang, James Y. Zou, Venkatesh Saligrama, and Adam T. Kalai. "Man is to computer programmer as woman is to homemaker? Debiasing word embeddings." In Advances in Neural Information Processing Systems, pp. 4349-4357. 2016.
	
## Deliverables
* We propose a novel algorithm that leverage adversarial learning to tackle the gender bias problem in word embeddings. 
* We make use of and give interpretation for the invariant word representation by experimenting on analogy generation task. We automatically generate pairs of words that are analogous to \textsl{she-he} and evaluate whether these pairs reflect gender stereotypes.
* We finally provide a sample gender-unbiased embedding for test use.


## Structure of this repo
__Todo__