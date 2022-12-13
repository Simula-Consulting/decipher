# Inverse Propensity Weighting
## Overview
In the matrix factorization paper ([Langberg et al., 2022](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04949-8)), the Weighted CMF (WCMF) and Shifted WCMF (SWCMF) weigh each observation by its inverse propensity score, using the method in [W. Ma and G. Chen (2019)](https://arxiv.org/abs/1910.12774).

This method aims to reduce bias from entries missing not at random (MNAR). In our case, there could be other underlying conditions that cause females to get tested and perhaps also affect results.

## Method
The matrix factorisation algorithm works by reducing the loss:
$$
\min_{U, V}{\{ \lVert \textbf{W} \odot (\textbf{Y - UV}^T)  \rVert + \dots \}} \; \; ,
$$
where $\textbf{W}$ is the weight matrix. The unweighted approaches uses $W_{i, j} = M_{i, j} \in \{ 0, 1\}$ which is simply a mask of observed entries.

Now we instead want to use $\hat{W}_{i, j}=M_{i, j}/\hat{P}_{i, j}$ where $\hat{P}_{i, j} \in [0, 1]$ is the estimated probability of an entry being revealed.

$\hat{P}_{i, j}$ is given by:

$$
\hat{P}_{i, j} := \sigma(\hat{A}_{i, j}) \;\; ,
$$

where $\sigma$ is the sigmoid function and $\hat{A}$ is estimated by solving the maximum likelihood problem (note how the terms vanish depending on if $M_{i,j}$ is 0 or 1):

$$
\hat{A} = \arg \max_{\Gamma \in \mathcal{F}_{\tau, \gamma}}\left\{
     \sum_{i=1}^m\sum_{j=1}^n
     \left[ 
        M_{i, j} \log \sigma(\Gamma_{i, j}) + (1 - M_{i, j}) \log(1 -  \sigma(\Gamma_{i, j}))
     \right]
     \right\} \;,
$$

under the constraint:

$$
A \in \mathcal{F}_{\tau, \gamma}:= \left\{
    \Gamma \in \mathbb{R} ^{m \times n} : 
    \lVert \Gamma \rVert_{*} \leq \tau \sqrt{mn},\; \lVert \Gamma\rVert_{\max} \leq \gamma
\right\} \; \; ,
$$

where $\tau$ and $\gamma$ are user specified. This is an optimisation problem and can be solved using projected gradient descent in the preprocessing step (before training the MatFact model).

