An activation function is a mathematical operation applied to the outputs of a neuron. The value resulting from the activation function determines whether the neuron will fire or not. This mechanism adds non-linearity that allows the network to learn complex patterns.  

Hence, a linear activation function would look like:  
$$f(x) = x$$   
And the range would be $(-\infty, \infty)$. Only passing linear values to the next layer would result into a linear model which won't be able to learn complex patterns.

This is why we use non-linear activations:   

- Sigmoid:
    * Function: $\sigma(z) = \frac{1}{1 + e^{-z}}$. Output can be either 0 or 1.
    * Gradient: $\sigma'(z) = \sigma(z) (1 - \sigma(z))$ (Previously derived in mlp.md in the second term for backpropagation.)

- Tanh:
    * Function: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$. Output can be either -1 or 1. As you can see, the output is 0-centered. This allows for balanced weight updates as compared to sigmoid. However, computing the extra $e^z$ term is more computationally expensive than sigmoid.
    * Gradient:  
    ```math
    \begin{aligned}
    \tanh(x) &= \frac{\sinh(x)}{\cosh(x)} \\
    \frac{d\tanh(x)} {dx} &=  \frac {d \frac{\sinh(x)}{\cosh(x)}} {dx} \\
    \text{ using quotient rule } \\
    &= \frac{\sinh'(x)\cosh(x) - \sinh(x)\cosh'(x)}{\cosh^2(x)} \\
    &= \frac{\cosh^2(x) - \sinh^2(x)}{\cosh^2(x)}        \dots  (\sinh'(x) = \cosh(x) \text{ and } \cosh'(x) = \sinh(x)) \\
    &= \frac{1}{\cosh^2(x)}    \dots (\cosh^2(x) - \sinh^2(x) = 1) \\
    &= \text{sech}^2(x) \\
    &= 1 - \tanh^2(x)   \dots   (\text{sech}^2(x) = 1 - \tanh^2(x))
    \end{aligned}
    ```

- ReLU:
    * Function: $\text{ReLU}(z) = \max(0, z)$. Range is $[0, \infty)$
    * Gradient:   
    ```math
    \text{ReLU}'(z) = 
    \begin{cases}
    0 & z \leq 0 \\
    1 & z > 0
    \end{cases}
    ```

- LeakyReLU:
    * Function: $\text{LeakyReLU}(z) = \max(\alpha z, z)$. Range is $(-\infty, \infty)$. Leaky ReLU was introduced to solve the problem of dying ReLU where the neurons would die if the activation is negative. Instead, LeakyReLU allows a small negative activation to pass through the neuron, so it's weights are updated continuously.
    * Gradient:    
    ```math
    \text{LeakyReLU}'(z) = 
    \begin{cases}
    \alpha & z \leq 0 \\
    1 & z > 0
    \end{cases}
    ```

- Softmax:
    * Function: $\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^N e^{z_j}}$. Output is a vector of probabilities of size N. Each term corresponds to the probability of the respective class such that sum of all the terms is always 1.
    * Gradient:   
    ```math
    \begin{aligned}
    \sigma(z_i) &= \frac{e^{z_i}}{\sum_{j=1}^N e^{z_j}} \\
    &= \frac{e^{z_i}} {S}     \dots   (S = \sum_{j=1}^N e^{z_j}) \\
    \end{aligned}
    ```
    We want to find the partial derivative of the output $S_i$ with respect to the input logit $z_k$, denoted as $\frac{\partial S_i}{\partial z_k}$.   
    Because $S_i$ depends on all $z$, we must evaluate two separate cases using the **Quotient Rule**:   
    $$\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$$   
 
    ---
    ### Case 1: When $i == k$ (Diagonal terms)   
    - $f = e^{z_i} \implies f' = e^{z_i}$   
    - $g = S \implies g' = \frac{\partial}{\partial z_k}(\sum_{j=1}^{N} e^{z_j}) = e^{z_i}$    

    $\text{Applying the quotient rule:}$    
    $$\frac{\partial \sigma(z_i)}{\partial z_k} = \frac{(e^{z_i} \cdot S) - (e^{z_i} \cdot e^{z_i})}{S^2}$$    

    $\text{Split the fraction: }$   
    $$\frac{\partial \sigma(z_i)}{\partial z_k} = \frac{e^{z_i} \cdot S}{S^2} - \frac{e^{z_i} \cdot e^{z_i}}{S^2}$$   

    $\text{Simplify by substituting back }\sigma(z_k) = \frac{e^{z_i}}{S}$:   
    $$\frac{\partial \sigma(z_i)}{\partial z_k} = \left(\frac{e^{z_i}}{S}\right) - \left(\frac{e^{z_i}}{S}\right)\left(\frac{e^{z_i}}{S}\right)$$    
    $$\frac{\partial \sigma(z_i)}{\partial z_k} = \sigma(z_i) - \sigma(z_i)^2 = \sigma(z_i)(1 - \sigma(z_i))$$    

    ---
    ### Case 2: When $i \neq k$ (Rest of the terms)

    We are differentiating $\sigma(z_i)$ with respect to a different input $z_k$.   

    - $f = e^{z_i} \implies f' = 0$ (since $z_i$ is treated as a constant relative to $z_k$)   
    - $g = S \implies g' = \frac{\partial}{\partial z_k}(\sum_{j=1}^{K} e^{z_j}) = e^{z_k}$   

    Applying the quotient rule:   
    $$\frac{\partial \sigma(z_i)}{\partial z_k} = \frac{(0 \cdot S) - (e^{z_i} \cdot e^{z_k})}{S^2}$$   

    $$\frac{\partial \sigma(z_i)}{\partial z_k} = \frac{- e^{z_i} e^{z_k}}{S^2}$$   

    Separate the terms into individual softmax forms:   
    $$\frac{\partial \sigma(z_i)}{\partial z_k} = - \left(\frac{e^{z_i}}{S}\right) \left(\frac{e^{z_k}}{S}\right)$$   

    $$\frac{\partial \sigma(z_i)}{\partial z_k} = - \sigma(z_i) \sigma(z_k)$$    

    ---

    If you map out this derivative for all inputs and outputs, it forms a Jacobian matrix:

    ```math
    J = \begin{bmatrix} 
    \sigma(z_1)(1-\sigma(z_1)) & -\sigma(z_1)\sigma(z_2) & \dots & -\sigma(z_1)\sigma(z_k) \\
    -\sigma(z_2)\sigma(z_1) & \sigma(z_2)(1-\sigma(z_2)) & \dots & -\sigma(z_2)\sigma(z_k) \\
    \vdots & \vdots & \ddots & \vdots \\
    -\sigma(z_k)\sigma(z_1) & -\sigma(z_k)\sigma(z_2) & \dots & \sigma(z_k)(1-\sigma(z_k))
    \end{bmatrix}
    ```