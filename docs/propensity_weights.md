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

where $\tau$ and $\gamma$ are user specified. This is an optimization problem and can be solved using projected gradient descent in the preprocessing step (before training the MatFact model). The only requirement is the binary observation mask $M_{i,j}$ which is known.

## Implementation
The optimization problem is solved with `tensorflow`. 

### Constraints
The projected gradient descent is implemented by imposing a [`tf.keras.constraints.Constraint`](https://www.tensorflow.org/api_docs/python/tf/keras/constraints/Constraint) on our optimisation [`Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable), `A`:

```python
import tensorflow as tf
A = tf.Variable(tf.random.uniform(M.shape), constraint=my_constraint(tau, gamma))
```
The constraint is a custom class enforces the two normalisations from above. 

### Solving the Equation
To maximize the likelihood function, we first define a loss function as the opposite of the likelihood:
```python
from tensorflow.math import log, reduce_sum, sigmoid

M_one_mask, M_zero_mask = (M==1), (M==0)
def loss():
    term1 = reduce_sum(log(sigmoid(A[M_one_mask])))
    term2 = reduce_sum(log(1 - sigmoid(A[M_zero_mask])))
    return -(term1 + term2)
```
and then minimizing `A` based on this loss function with a normal Adam optimizer, automatically subject to the constraints:
```python
opt = tf.optimizers.Adam()

for _ in range(n_iterations):
    opt.minimize(loss, var_list=[A])
```
