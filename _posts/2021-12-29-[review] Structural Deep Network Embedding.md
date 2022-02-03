---
title: "Structural Deep Network Embedding"
categories:
  - paper review
classes: wide
excerpt: "SDNE reivew"
tags: 
  - embedding
  - graph
---





![논문제목](/assets/review_SDNE/ppt01.JPG)

## Abstract

how to find a method that is able to effectively capture the highly non-linear network structure and preserve the global and local structure is an open yet important problem.

non-linear한 구조를 파악하고, global과 local한 구조를 보존할 수 있는 문제는 아직 해결되지 않았습니다. 

그에 대한 방법으로 SDNE를 제안했습니다. 

1. semi-supervised model 사용했습니다. : multiple layers of non-linear functions을 사용하였습니다. 
2. exploit the first-order and second-order proximity jointly to preserve the network structure:  first-order과 second-order proximity 결합하여 network의 structure를 보존하고자 했습니다. 



---



## 1. Introduction

- networks는 어디나 있습니다. 트위터에서 tweets을 추천하는 추천 시스템과  SNS에서 communities를 clustering 하는 등 어디에서나 사용되어집니다.

- One of the fundamental problems is how to learn useful **network representations**.

  가장 중요한 문제는 network를 어떻게 표현하는가입니다. network를 표현하는데 다음과 같은 중요한 문제가 있습니다. 

  

- Great challenges

  - **High non-linearity:** network는 근본적으로 High non-linearity입니다. 이를 포착할 수 있는 모델을 디자인하는 것은 어렵습니다.  
  - **Structure-preserving**: network는  복잡합니다. 그래서  Structure를 보존하는 것이 필요합니다. vertexd의 유사도는 local과 global에 양쪽에 의존합니다. 이를 동시에 보존한다는 것은 어려운 문제입니다.
  - **Sparsity:**  많은 현실의 graph는 Sparsity합니다. 그래서 성능이 안 나오는 경우가 많습니다. 

  

- 이런 문제를 해결하기 위해 제안된 기존의 방법론들은 다음과 같은 것들이 있습니다. 이는 Experiments에서 baseline algorithm으로서 비교해줍니다. 간단히 설명하면 다음과 같습니다. 

  - IsoMAP

    > ISOMAP 은 manifold 에서의 점들 간의 거리를 nearest neighbor graph 에서의 점들 간의 최단 경로로 정의합니다. 그림 (b) 처럼 표면을 따라 이동하는 거리로 두 점 사이의 거리를 정의합니다. 그리고 이 정보를 보존하는 2 차원 임베딩 공간을 학습합니다.

  - Laplacian Eigenmaps 

    > SDNE에서도 등장하는 해당 임베딩은, Edge weigh가 높을 수록 두 노드간의 거리가 가까워질 수 있는 것에 집중한 방법론입입니다. Normalized laplacian matrix를 기반으로 계산되며 아래 수식을 최소화하는 방향으로 학습하게 됩니다.

    다음 모델들은 highly non-linear structure를 파악하기 어렵습니다.

    추가적으로 Kernel trick을 이용한 모델들도 사용되어지기는 하지만 이 또한 피상적인 모델(shallow model)로서 highly non-linear structure을 파악하기 어렵습니다.

    

- In order to capture the highly non-linear structure well, in this paper we propose a new deep model to learn vertex representations for networks

  - 비선형성을 잘 파악하기 위해서  **vertex representations** for networks를 하는 딥러닝 모델을 제안합니다.

    

- In order to address the structure-preserving and sparsity problems in the deep model, we further propose to exploit the first-order and second-order proximity 

  - Structure-preserving과 Sparsity 문제를 해결하기 위해서 first-order and second-order proximity를 이용했습니다. 

  > J. Tang, M. Qu, M. Wang, M. Zhang, J. Yan, and Q. Mei. Line: Large-scale information network embedding. In Proceedings of the 24th International Conference on World Wide Web, pages 1067–1077. International World Wide Web Conferences Steering Committee, 2015.

- The first-order proximity is the local pairwise similarity 오직 vertexes와 연결된 edge들만 가지고 local similarity를 구하였습니다. 하지만 실제로는 links없는 Sparsity 경우가 존재합니다. 그래서 vertexes만을 가지고 representation하는 것은 부족합니다. 따라서 second-order proximity를 제안했습니다.

- Second order proximity는 similarity of the vertexes’ neighborhood structures를 포함하고 있습니다. Second order proximity는 first-order proximity을 정답지로 하여 semi-supervised architecture로 구하였습니다.

- 본 논문에서는 실제  5개의 networked datasets을 통해 4가지 현실 문제를 적용시켜 다른 알고리즘과 비교해봅니다. 결과를 통해서 다양한 작업과과 다양한 네트워크에서 더 좋은 성능을 보여줌을 나타냅니다.



> 간단히 결과를 정리해보면 다음과 같습니다. 
>
> (1) network embedding 하는 Structural Deep Network Embedding method을 제안한다.
>
> (2)  semi-supervised의 구조를 포함하는 deep model을 통해 first-order and second-order proximity를 동시에 최적화합니다.
>
> (3) 실제 사례를 통해 성능을 검증합니다. 

---



## 2. Related Work

### Deep Neural Network

Deep Neural Network는 강력한 powerful representations abilities 방법입니다.

네. 강력하답니다. 

### Network Embedding 

- Our work solves the problem of network embedding, which aims to learn representations for networks.

Local Linear Embedding(LLE)

> 지역 선형 임베딩, PCA와 달리 투영(projection)이 아닌 **매니폴드 학습**(manifold learning) 이다.
>
> 출처: https://excelsior-cjh.tistory.com/168 [EXCELSIOR]

최근에  local and global network structure 각자 loss functions 사용 

- Deep Walk = random walk and skip-gram 

>  그래프 형태의 데이터를 효과적으로 임베딩할 수 있는 방법 중 하나인 **`DeepWalk`** 에 대해서 써보려고 합니다. 정확히 말하자면 그래프의 `노드` 를 임베딩할 수 있는 방법인데요. 핵심을 요약하면 다음과 같습니다.
>
>  `skip-gram` 은 `word2vec` 의 방법 중 하나입니다. 대상 단어와 주변 단어를 관계를 이용해 함께 자주 등장한 단어일수록 유사한 백터공간에 표현하는 것이 목적입니다. 입력은 단어 뭉치이고 학습을 완료했을 때 최종 결과 값은 가중치 행렬입니다. 이 가중치 행렬은 각 단어의 벡터 값을 포함하고 있습니다.
>
>  skip gram 은 **대상 단어가 입력으로 들어왔을 때 주변 단어를 예측하는 식**으로 학습이 진행됩니다. `window size` 는 대상 단어가 주어졌을 때 주변 단어를 몇 개 까지 확인할 것인지를 나타내는 하이퍼파라미터 입니다. 만약 window size 가 2라면 대상 단어의 양 옆 2개의 단어까지를 학습합니다.



---



## 3. Structural Deep Network Embedding

### Problem Definition

**Network Embedding aims to map the graphs data into a low dimensional latent space** 

DEFINITION 1. (Graph) A graph is denoted as $G = (V, E)$, where $V = {v1, ..., vn}$ represents n vertexes and $E = {ei,j} n i,j=1$ represents the edges. Each edge ei,j is associated with a weight i,j ≥ 0 1 . For vi and vj not linked by an edge, si,j = 0. Otherwise, for unweighted graph si,j = 1 and for weighted graph, si,j > 0.





그래프는 꼭지점 V와 edges로 표기된다. 연결된 것은 ei >0 이며 연결되지 않은 것은 0이다. unweighted graph의 경우 s는 0보다 크고, weighted graph의 경우  s는 1이다. 

Network embedding aims to map the graph data into a low dimensional latent space, where each vertex is represented as a low-dimensional vector and the network computing can be directly realized.

네트워크 임베딩의 목적은 컴퓨터에서 계산하기 위해 graph 데이터를  low dimensional latent로 바꾸는데 목적이 있다. 



DEFINITION 2. (First-Order Proximity) The first-order proximity describes the pairwise proximity between vertexes. For any pair of vertexes, if si,j > 0, there exists positive first-order proximity between vi and vj . Otherwise, the first-order proximity between vi and vj is 0.

vertexes간의  pairwise proximity(쌍별 유사도)가 first-order proximity이다. 

si,j가 0 이상이면 positive first-order proximity가 존재한다. 만약 0이라면 존재하지 않는다. 

1차 근접성은 실제 네트워크의 두 정점이 관찰된 가장자리로 연결된 경우 항상 유사함을 의미하기 때문입니다.

하지만 서로 비슷하지만 가장자리로 연결되지 않은 정점들이 많기 떄문에 global network structure를 보존하기는 어렵다.



DEFINITION 3. (Second-Order Proximity) The second-order proximity between a pair of vertexes describes the proximity of the pair’s neighborhood structure. Let Nu = {su,1, ..., su,|V |} denote the first-order proximity between vu and other vertexes. Then, secondorder proximity is determined by the similarity of Nu and Nv.



2차 근접성은 vertexes describes the proximity of the pair’s neighborhood structure이다. 간단힌 말하면 인접 행렬에 의해서 계산된다는 것을 의미한다. 1차 근접성과 2차 근접성을 이용하여 구조를 보존하고 이들을 통합하는 임베딩을 찾습니다. 

두 정점은 많은 공통 이웃을 공유할수록 유사할 것이다라는 가정으로 계산됩니다.

Nu의 유사도에 의해서 2차 근접성이 결정됩니다. 따라서 2차 근접성을 도입하여 글로벌 네트워크 구조를 특성화하고 희소성 문제를 완화할 수 있습니다.



DEFINITION 4. (Network Embedding) Given a graph denoted as G = (V, E), network embedding aims to learn a mapping function f : vi 7 → yi ∈ R d , where d  |V |. The objective of the function is to make the similarity between yi and yj explicitly preserve the first-order and second-order proximity of vi and vj .

최종적으로 network Embedding 문제를 다음과 같이 정의하였습니다. 차원을 d로 축소하는 과정에 있어서 the first-order and second-order proximity 보존하면서, yi와 yj가 같게끔 만드는 것이 이 논문의 목표(objective function) 입니다. 



### The Model

- Framework

이 페이퍼에서는 다음과 같은 semi-supervised deep model을 제시합니다.

아래 그림과 같이 빨간색과 파란색, Unsupervised 부분과 Supervised 부분으로 구성되어 있습니다.

![Framework](/assets/review_SDNE/ppt08.JPG)



![Unsupervised component](/assets/review_SDNE/ppt09.JPG)

Unsupervised 부분은 traditional autoencoder의 형태를 확장한 모델입니다. encoder와 decoder로 구성되어 있으며 encoder은 non-linear function로 되어 있습니다. Unsupervised 부분에서는 Vertex의 인접행렬 한 row가 들어가게 됩니다. 그리고 autoencoder를 통해서 latent vector를 추출합니다. i,j vertex에서 이를 수행합니다. 



![Supervised component](/assets/review_SDNE/ppt10.JPG)

추출된 latent representation들간의 관계를 비교해서 Laplacian Eigenmaps을 진행합니다.

이부분에서 first-order proximity를 supervised information으로 사용합니다. 





### Loss Functions

최종적인 Loss Function은 first-order and second-order proximity에 대한 Loss function과 regularizer term으로 구성됩니다.

![Loss Functions](/assets/review_SDNE/ppt12.JPG)



### Optimization 

![Optimization](/assets/review_SDNE/ppt13.JPG)

![Optimization](/assets/review_SDNE/ppt14.JPG)



### Algorithm 

![Algorithm](/assets/review_SDNE/ppt15.JPG)

---



## 4. Experiments

실험은 5개의 데이터셋을 가지고 Network Reconstruction, Multi-label Classification, Link Prediction, Visualization Task를 기존의 알고리즘과 비교했습니다. 



### Datasets 

- Three social networks
  - BLOGCATALOG: 39 different categories
  - FLICKR: 195 categories
  -  YOUTUBE: 47 categories
- One citation network 
  - ARXIV GR-QC: General Relativity and Quantum Cosmology from arXiv
- One language network
  - 20-NEWSGROUP: e 20000 newsgroup documents and each document is labelled by one of the 20 different groups.



### Baseline Algorithms 

다음과 같은 기존의 Representation Algorithm과 비교 실험을 진행했습니다. 

![기존의 Representation Algorithm](/assets/review_SDNE/ppt17.JPG)

- DeepWalk

  > 네트워크 표현을 생성하기 위해 랜덤 워크 및 스킵 그램 모델을 채택합니다.

- LINE

  > 1차 또는 2차 근접성을 별도로 유지하기 위해 손실 함수를 정의합니다. 손실 함수를 최적화한 후 이러한 표현을 연결합니다

- GraRep

  > 고차 근접도로 확장하고 모델을 훈련시키기 위한 SVD. 또한 1차 및 고차 표현을 직접 연결합니다.

- Laplacian Eigenmaps (LE)

  >인접 행렬의 라플라시안 행렬을 분해하여 네트워크 표현을 생성합니다. 네트워크 구조를 보존하기 위해 1차 근접성만 이용합니다.

- Common Neighbor

  > 정점 간의 유사도를 측정하기 위해 공통 이웃의 수만 사용합니다. 링크 예측 작업에서만 기준선으로 사용됩니다.



### Evaluation Metrics

![Evaluation Metrics](/assets/review_SDNE/ppt19.JPG)

reconstruction and link prediction에서 사용한 평가 지표 입니다. 

- precision@k 
- MAP(Mean Average Precision)

multi-label classification task에서 사용한 평가 지표 입니다. 

- Macro-F1
- Micro-F1





### Parameter Settings

서로 다른 algorithms에 대한 비교 실험을 위해 parameter 설정을 어떻게 했는지를 이야기해줍니다. 

![Parameter Settings](/assets/review_SDNE/ppt18.JPG)



### Experiment Results

다음 task에 대해서 알고리즘에 대해서 기존의 알고리즘과 비교해서 결과를 분석합니다.

- Network Reconstruction

![Network Reconstruction](/assets/review_SDNE/ppt20.JPG)



- Multi-label Classification

![Multi-label Classification](/assets/review_SDNE/ppt21.JPG)

![Multi-label Classification](/assets/review_SDNE/ppt22.JPG)



- Link Prediction

![Link Predictionl Classification](/assets/review_SDNE/ppt23.JPG)

![Link Predictionl Classification](/assets/review_SDNE/ppt24.JPG)



- Visualization

![Visualizationl Classification](/assets/review_SDNE/ppt25.JPG)



### Parameter Sensitivity

Loss function의 parameter 조정에 대해서 실험을 진행하였습니다. 

![Parameter Sensitivityl Classification](/assets/review_SDNE/ppt26.JPG)

임베딩 차원에 대해서 크게 영향을 받지 않지만 다른 파라미터는 영향을 크게 받는 것을 확인할 수 있습니다. 



## 5. Conclusions

본 논문에서 Deep autoencoder를 **사용하여** **first and second order network proximities** 보존합니다. 

Deep Learning 기반의 autoencoder를 사용했기 때문에 **High non-linearity**를 잘 파악합니다. 

Dataset을 활용하여, Reconstruction, Classification 등의 Task 진행했을 때 기존의 알고리즘보다 향상된 성능을 보입니다.

SDNE가 다른 알고리즘보다 **Sparsity**에 대해서 **Robust**한 것을 실험을 통해서 확인했습니다. 

추후 연구로서 기존의 연결되어 있지 않은 vertex에 대한 연구를 진행할 것이라고 이야기합니다.  







## 참조 자료 

위 포스팅은 같은 랩실의 김명준 박사님의 도움을 많이 받았습니다. 

- https://lovit.github.io/nlp/representation/2018/09/28/mds_isomap_lle/
- https://junklee.tistory.com/113 
- https://towardsdatascience.com/graph-embeddings-the-summary-cc6075aba007
