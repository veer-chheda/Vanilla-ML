Support Vector Machines are supervised algorithms that have the objective of finding the best separating hyperplane. The distance from the hyperplane to the nearest support vectors is maximised. Support vectors are the lines that pass through the closest points to the hyperplane such that the distance i.e. margin is maximum. SVMs use kernels for projecting data into a higher space to handle non-linear data.The goal is to find a hyperplane defined by $wx - b = 0$ where,
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
    $\frac{\partial J}{\partial w} = w$
    As the derivative doesn't include $x$, this condition introduces sparsity by weight shrinking.
* Point is classified incorrectly i.e. inside the margin:
    The point margin is $y \cdot (wx + b) < 1$. Hence, the derivative becomes:
    ```math
        \frac{\partial J}{\partial w} = w - y . x
    ```
This is the primal form of SVM. The loss function is a combination of hinge loss and L2 regularisation.
```math
L = \lambda \|w\|^2 + \sum_{i=1}^N \max(0, 1 - y_i(wx_i + b))
```
