---
title: "Chapter 3. Analytic Geometry(1) 요약 정리"
categories:
  - paper review
classes: wide
excerpt: "graph reivew"
tags: 
  - Mathematics for Machine Learning
  - study

 
---



# Chapter 3. Analytic Geometry

가짜 연구소에서 Mathematics for Machine Learning 스터디를 하고 있습니다. 본 글은 다음 학습한 내용을 간단하게 정리한 글입니다. 모두 한 글에 정리하고자 했으나, Chapter 3의 9. Rotations까지 한꺼번에 정리하게 되면 너무 길어져서 두 부분으로 나누어서 정리했습니다. 

# Chapter 3. Analytic Geometry

지난 Chapter 2에서는 Linear Algebra에 관한 전반적인 개념에 대해서 살펴보았습니다. vector와 vector space, linear map 등에 대해서 간략히 개념을 학습해보았습니다. 3장은 vector의 길이부터 시작해서, Angles, Inner product, Orthogonality에 대해서 살펴볼 것입니다.

## Norms

첫번째 나오는 개념은 Norms 입니다. 간단히 설명하면 vector의 길이를 표현합니다. 시작점과 끝점이 있는 화살표의 길이를 표현한다고 생각하면 쉽습니다. 

![Norm의 정의(하나 하나 쳐보고 미친 짓이란 걸 깨달았습니다) ](Chapter%203%20%201ad9f/Untitled.png)

Norm의 정의(하나 하나 쳐보고 미친 짓이란 걸 깨달았습니다) 

이러한 Norm에는 대표적으로 Manhattan Norm과  Euclidean Norm이 존재합니다. 

![Untitled](Chapter%203%20%201ad9f/Untitled%201.png)

다음 왼쪽의 그래프에 원점으로 부터 Manhattan Norm이 1인 모든 vector의 trace(자취)를 그린 그래프입니다. 오른쪽의 그림이 원점으로 부터 Euclidean Norm이 1인 모든 vector의 trace(자취)를 그린 그래프입니다. 일반적으로 왼쪽의 Euclidean Norm을 많이 사용합니다.

## Inner Products

두번째 개념은 Inner Products 입니다. 내적이 중요한 이유는 두개의 vector의 Length, angle, distance에 대해서 기하학적으로 접근할 수 있기 때문입니다. 이 Inner Products의 가장 큰 목표는 vector간의 orthogoanl한 관계를 유추하는 것입니다.

우선 고등학교 때 배웠던 Dot Product를 기억해봅시다. 

![Untitled](Chapter%203%20%201ad9f/Untitled%202.png)

사실 Dot Proudct = 내적이 아닙니다. 엄밀히 말하면 Inner Poduct의 한 형태라고 볼 수 있습니다. 

Inner Product의 정의 우선에 일반적으로 두개의 vector를 하나의 real number 값으로  표현하는 것을 Lilinear mapping 이라고 하며  다음과 같은 정의를 가집니다. 

![Untitled](Chapter%203%20%201ad9f/Untitled%203.png)

그리고 $\Omega$ 의 symmetric(대칭성)과 positive definite의 특성을 보여줍니다. 

- 이때 $\Omega (x,y)=\Omega(y,x)$라면 $\Omega$는 **symmetric**하다고 정의합니다
- 다음과 같은 조건을 만족하면 positive definite 라고 정의합니다.

그리고 General Inner Products의 경우 선형성을 띄기 때문에 다음과 같은 성질을 보입니다.

![Untitled](Chapter%203%20%201ad9f/Untitled%204.png)

최종적으로 Vector space V에 대한 inner product는 다음과 표현됩니다. 

A positive definite, symmetric한 bilinear mapping을 Vecotr space V에 대한 내적이라고 정의합니다.

![Untitled](Chapter%203%20%201ad9f/Untitled%205.png)

### Symmetric, Positive Definite Matrices

Symmetric, Positive Definite Matrices은 Inner product를  통해 정의됩니다.

하지만 이는 매우 중요한 요소입니다. 다음 내용은 Chapter 3에는 없지만 Support Vector Machine의 Kernel trick에서 매우 중요한 요소로서 작용됩니다. **저차원 공간(low dimensional space)을 고차원 공간(high dimensional space)으로 매핑해주는 작업을 커널 트릭(Kernel Trick)이라고 합니다.** 

> **Kernel 함수 K 가** 실수 scalar 를 출력하는 **continuous function**일 때, Kernel 함수값으로 만든 행렬이 **Symmetric**(대칭행렬)이고, **Positive semi-definite**(대각원소>0)라면, **K(xi, xj) = K(xj, xi) = <Φ(xi), Φ(xj)>를 만족하는 mapping Φ 가 존재**한다.
> 

이 부분은 나중에 Support Vector Machine과 Kernel에 대해서 소개할 때 자세히 설명하겠습니다. 

아무튼. 이 모든 것이 시작이 Inner Product 입니다 . 

이  Positive Definite Matrices 핵심 정의는 다음과 같습니다.

![Untitled](Chapter%203%20%201ad9f/Untitled%206.png)

![Untitled](Chapter%203%20%201ad9f/Untitled%207.png)

(3.11)의 식을 따르면 Symmetric matrix A는 Symmetric, Positive Definite하다. 

Positive Definite Matrix 모든 positive eigenvalues을 갖는 행렬입니다.

기학적으로 표현하면 n차원 공간의 타원체 (Ellipsoid)의 형태이며, 그 안의 n개의 축은 A의 eigenvectors, 축의 길이는 eigenvalues으로 생각할 수 있습니다.

다음 식의 간단한 문제를 통해서 예시를 들어보면 

![Untitled](Chapter%203%20%201ad9f/Untitled%208.png)

$A_1: x^TA_1x = (완전제곱꼴) > 0$  의 형태이므로 positive definite라고 할 수 있습니다.

반면 $A_2: x^TA_1x \not= (완전제곱꼴)$ 의 형태이므로   positive definite라고 할 수 없습니다.

## Lengths and Distances

길이는 다음과 같이 Inner product로 정의됩니다.

![Untitled](Chapter%203%20%201ad9f/Untitled%209.png)

그리고 다음과 같은 Mapping을 meric이라고 합니다

![Untitled](Chapter%203%20%201ad9f/Untitled%2010.png)

Inner product를 통해 norm을 도출한 다는 점에서 두개는 밀접한 관련이 있습니다. 

## Angle and Orthogonality

 Inner products를 통해서 두 vector 간의 angle $w$를 정의할 수 있습니다. 

$x \not= 0, y\not=0$ 이라면 Cauchy-Schwarz inequality에 (3.24)가 성립합니다. 

![Untitled](Chapter%203%20%201ad9f/Untitled%2011.png)

따라서 unique  $w \in [0, \pi]$로 나타낼 수 있으며 이는 아래와 같이 표현됩니다 .

![Untitled](Chapter%203%20%201ad9f/Untitled%2012.png)

Inner products를 통해서 Orthogonality와 Orthonormal을 정의할 수 있습니다. 

vector x와 y가 Inner products가 0 일 경우 orthogonal이라고 하고, x와 y의 norm이 1이 경우 Orthonormal이라고 정의합니다.

![Untitled](Chapter%203%20%201ad9f/Untitled%2013.png)

![Untitled](Chapter%203%20%201ad9f/Untitled%2014.png)

→ 다음 직교성은 중요한 성질로 데이터로 생각해보면 x, y를  데이터로 가정해서 생각해보면 두 데이터의특성이 완전 독립이란 것을 의미합니다. 

이 부분에서 Orthogonal Matrix를 함께 정의해보면 다음과 같이 정의할수 있습니다

![Untitled](Chapter%203%20%201ad9f/Untitled%2015.png)

다음 정방행렬 A columns이 Orthonormal하면  Orthogonal Matrix라고 하고 (3.29)와 (3.30)을 만족하게 됩니다. 

# 마무리

본 글에서 기본적인 정의에 대해서 정리해보았습니다. 다음으로 정의를 기본으로 Orthonormal Basis를 알아본 후 Inner Product를 Function으로  확장한 Inner Product of Functions에 대해서 살펴볼 예정입니다. 그리고 PCA 기법 등에 대해서 가장 기본이 되는 Projection과 Rotation에 대해서 정리해보겠습니다. 18학점 + 졸프 + ... 기타 등 많지만 가짜 연구소의 스터디를 잘 정리해보겠습니다. 

다음 번에는 조금 더 완성도 있는 글을 가져오겠습니다. 짧은 글 읽어주셔서 감사합니다.