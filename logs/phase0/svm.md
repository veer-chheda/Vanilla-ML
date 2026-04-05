Support Vector Machines are supervised algorithms that have the objective of finding the best separating hyperplane. The distance from the hyperplane to the nearest support vectors is maximised. Support vectors are the lines that pass through the closest points to the hyperplane such that the distance i.e. margin is maximum. SVMs use kernels for projecting data into a higher space to handle non-linear data. The goal is to find a hyperplane defined by $wx - b = 0$ where,
```math
y = 
\begin{cases} 
1 & wx - b \geq 0 \\ 
-1 & wx - b < 0 
\end{cases}
```
This can be written as a combined equation $y \cdot (wx + b) \geq 1$.Distance of a point from a line can be written as $\frac{wx + b}{\|w\|}$. Also, support vectors are defined as $|wx + b| = 1$. Therefore, distance from support vector is $\frac{1}{\|w\|}$ and the total margin becomes $\frac{2}{\|w\|}$. (Since there is a margin on either side of the hyperplane).  
The objective function is to maximise this margin $\frac{2}{\|w\|}$, which is the same as minimising $\frac{\|w\|}{2}$. To make it mathematically easier, the objective is written as:  
```math
\min \frac{1}{2} \|w\|^2
```
subject to the constraint $y \cdot (wx + b) \geq 1$.  

The cost function effectively becomes,
```math
J(w) = \frac{1}{2} \|w\|^2 + \max(0, 1 - y(wx - b))
```
The function has two modes according to the following cases:
* Point is correctly classified i.e, is outside the margin:
    Loss from the point is 0 as $y \cdot (wx + b) \geq 1$. Only minimisation left is the regularisation term:
```math
    \frac{\partial J}{\partial w} = w 
```
As the derivative doesn't include $x$, this condition introduces sparsity by weight shrinking.
* Point is classified incorrectly i.e. inside the margin:
    The point margin is $y \cdot (wx + b) < 1$. Hence, the derivative becomes:
```math
    \frac{\partial J}{\partial w} = w - y . x
```
This is the primal form of SVM. The loss function is a combination of hinge loss and L2 regularisation.
```math
L = \lambda \|w\|^2 + \sum_{i=1}^N \max(0, 1 - y_i(w x_i + b))
```


## Dual Form

The objective $\min \frac{1}{2} \|w\|^2$ and the constraint $y \cdot (wx + b) \geq 1$ are the same. But here, we look at it as a constraint optimisation problem. If both equations are combined by introducing Lagrange multipliers,
```math
L \left(w, b, \alpha \right) = \frac{1}{2} \|w\|^2 + \sum_{i=1}^N \alpha_i \left(1 - y_i\left(w x_i + b\right)\right)
```
Therefore, to find the optimal conditions, we differentiate this equation,
* With respect to $w$:
```math
    \frac{\partial L}{\partial w} = w - \sum_{i=1}^N \alpha_i y_i \ x_i \\
    w = \sum_{i=1}^N \alpha_i y_i \ x_i 
```
* With respect to $b$:
```math
    \frac{\partial L}{\partial b} = - \sum_{i=1}^N \alpha_i y_i = 0 \\
    \sum_{i=1}^N \alpha_i y_i = 0
```

We can rewrite the Langrangian equation as:
```math
L \left(w, b, \alpha \right) = \frac{1}{2} w^T w - \sum_{i=1}^N \alpha_i y_i w^T x_i - \sum_{i=1}^N \alpha_i y_i b + \sum_{i=1}^N \alpha_i \\
= \frac{1}{2} w^T w - \sum_{i=1}^N \alpha_i y_i w^T x_i - b \sum_{i=1}^N \alpha_i y_i  + \sum_{i=1}^N \alpha_i \\
= \frac{1}{2} \left( \sum_{i=1}^N \alpha_i y_i \ x_i \right)^T \left( \sum_{j=1}^N \alpha_j y_j \ x_j \right) - \sum_{i=1}^N \alpha_i y_i \left( \sum_{j=1}^N \alpha_j y_j \ x_j \right)^T x_i - 0 + \sum_{i=1}^N \alpha_i \\
= \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j - \sum_{i=1}^N \sum_{j=1}^N  \alpha_i \alpha_j y_i y_j x_i x_j^T + \sum_{i=1}^N \alpha_i \\
W \left(\alpha \right) = \sum_{i=1}^N \alpha_i - \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j \\
```

To solve this dual problem, the solution should satisfy the Karush-Kuhn-Tucker conditions, which in our case is $\alpha_i \left[y_i \left(w^Tx_i + b - 1 \right) \right] = 0$.
* If $\alpha_i = 0$, then the point must lie outside the margin.
* If $alpha_i > 0$, then $\left[y_i \left(w^Tx_i + b - 1 \right) \right]$ = 0, meaning point lies exactly on the margin.

Since, the actual data is only present in the form of $x_i^T x_j$, SVMs are memory efficient and this expression can be replaced with a Kernel Function.
* Linear: $K(x_i, x_j) = x_i^T x_j$
* Polynomial: $K(x_i, x_j) = (x_i^T x_j + c)^d$
* Radial Basis Function (Gaussian): $K(x_i, x_j) = \exp{(-\gamma \|x_i - x_j\|^2)}

So in practical examples, people generally use Quadratic Programming to solve this problem.
A QP solver expects an optimization problem of this form:
```math
\text{Minimise: } \frac{1}{2} \alpha^T P \alpha + q^T \alpha \\
\text{Subject to: } G \alpha \leq h \text{ and }  A \alpha = b
```
In our case the objective is:
```math
\text{Maximise: } \sum_{i=1}^N \alpha_i - \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j \\
\text{OR} \\ 
\text{Minimise: }\sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j - \sum_{i=1}^N \alpha_i \\
```

The substitutions to be made to fit the QP form are:
* $P =  y_i y_j x_i^T x_j$ to get the $\frac{1}{2} \alpha^T P \alpha$.
* $q=-1$ to get q^T \alpha part as the second part of our Lagrangian equation is $\sum_{i=1}^N \alpha_i$.
* $A \alpha = b$, here we can take $\sum_{i=1}^N \alpha_i y_i = 0$. Hence, $A = y$ and $b = 0$.
* $G \text{and} h$ are concerned with soft margins. To maintain equalities of $\alpha_i \geq 0 \text{ or } -\alpha \leq 0$ amd $\alpha \leq C$, $G$ and $h$ are taken as stacked $I \text{ and } -I$ matrices and stacked 0s and Cs respectively. 

Links:
https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/support_vector_machine.py 
