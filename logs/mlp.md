A multilayer perceptron or MLP is a type of artificial neural network that is made up of neurons that typically use non-linear activations. This allows the network to learn complex patterns of data. 
Parts of a neural network are:
* Input layer - the first layer of neurons that receives the data
* Hidden layer - between the input and output layers, there can be more than one layers of neurons. Each hidden layer receives input from the neurons of the previous layer and pass an output to the next layer
* Output layer - the last layer that produces the final output of the network
* Weights - Neurons are connected in adjacent layers with an associated weight. This weight denotes the strength of the connectio. It is a signal of how much information a neuron will propagate to the next layer.
* Activation functions - Each neuron in the hidden and output layers apply certain mathematical transformation to the processed input. This is typically done to add non-linearity to the network which allows it to learn complex patterns of data. 
* Feed forward propagation - Each pass in the network from the input at input layer to the output processed at the output layer. 
* Back Propagation - The network applies a backward pass to compute gradients of a loss function with respect to the model parameters. The gradients are used to update the models' parameters to minimise a loss function.

![Multilayer Perceptron](/images/mlp.png)

For a perceptron, 
```math
a = \phi \left( \sum_{i} w_i x_i + b \right)
```
where, $x_i$ are the inputs to the neuron, $w_i$ are the weights, $b$ is the bias and $\phi$ is the non-linear activation function to compute $a$, the neuron's activation.

## Forward Propagation

![Neural Network](/images/mlp_with_weights.png)

So, for a given input x where,
```math
x =
\begin{bmatrix}
x_1 \\ 
x_2 \\ 
x_3 \\ 
\end{bmatrix}
```
We have a network consisting of three input neurons (one for each input) and two hidden neurons. Hence, the weight matrix can be represented as,
```math
W = 
\begin{bmatrix}
w_{11} & w_{12} \\ 
w_{21} & w_{22} \\ 
w_{31} & w_{32} \\ 
\end{bmatrix}
\text{ or }
W^T = 
\begin{bmatrix}
w_{11} & w_{21} & w_{31}\\ 
w_{12} & w_{22} & w_{32}\\ 
\end{bmatrix}
\text{ and }
b =
\begin{bmatrix}
b_1 \\ 
b_2 \\
\end{bmatrix}
```

We transpose the $W$ matrix to perform dot product with the input $X$. Along with the weights, each neuron also has an associated bias $b$.
```math
\begin{aligned}
z = W^T x + b &= 
\begin{bmatrix}
w_{11} & w_{21} & w_{31}\\ 
w_{12} & w_{22} & w_{32}\\ 
\end{bmatrix}
\begin{bmatrix}
x_1 \\ 
x_2 \\ 
x_3 \\ 
\end{bmatrix} 
+
\begin{bmatrix}
b_1 \\ 
b_2 \\
\end{bmatrix}
\\[15pt]
&= 
\begin{bmatrix}
w_{11}x_1 + w_{21}x_2 + w_{31}x_3 + b_1 \\ 
w_{12}x_1 + w_{22}x_2 + w_{32}x_3 + b_2 \\ 
\end{bmatrix}
\end{aligned}
```

This brings me to the concept of activation functions. Activation functions are equations essentially used to determine whether a neuron should activate or not. By activating, a neuron passes some information forward that depends on the type of activation function used. This introduces non-linearity which enables the network to learn complex patterns.  
Examples of activation functions are:
* Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$
* ReLU (Rectified Linear Unit): $ReLU(z) = \max(0, z)$
* Tanh (Hyperbolic tangent): $tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$  

So passing the $z$ vector through an activation function results into an activation which is basically the output of the hidden unit.
```math
\begin{aligned}
a = \sigma(z) &= \frac{1}{1 + e^{-z}} \\
=
\begin{bmatrix}
a_1 \\ 
a_2 \\
\end{bmatrix}
&=
\begin{bmatrix}
\frac{1}{1 + e^{-(w_{11}x_1 + w_{21}x_2 + w_{31}x_3 + b_1)}} \\ 
\frac{1}{1 + e^{-(w_{12}x_1 + w_{22}x_2 + w_{32}x_3 + b_2)}} \\
\end{bmatrix}
\end{aligned}
```

However, the output layer makes use of activations functions as thresholds to obtain outputs. For binary classification, it could be sigmoid whereas for multiclass classification, we use softmax function $\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^N e^{z_j}}$. For regression problems, we use the identity function $f(x) = x$

We also have cost functions or loss functions or objective functions (they have different meanings theoretically, but they refer to the same thing) that are used to measure the fit or "goodness" of a network's performance. Example: mean squared error, binary cross entropy, etc.

![Forward Propagation](/images/mlp_forward_prop.png)

If we pass the output of the hidden layer (i.e. it's activation that we derived before) into the threshold function (let's say sigmoid), we get the output $\hat{y}$.

## Backward Propagation
Back propagation is the process of weight updates according to the resulting error. We propagate the weight updates backward, from the output layer to the input layer. The basic idea is:
* The error depends on the output function's output $a$ (in our case, the output function is sigmoid).
* Value of the sigmoid function depends on the value of the activations $z$.
* Value of $z$ depends on the weights $w$ from the previous layer.

Thus, we can identify a chain. Now, we must find the answer to:
* How does the error change when we change activation output $a$.
* How does activation output $a$ change when we change the input activation $z$.
* How does the input activation $z$ change with a small change in the weights $w$.

This can be summarized by the chain rule from calculus:
```math
 \frac{\partial E} {\partial w^L} = \frac{\partial E} {\partial a^L} \times \frac{\partial a^L} {\partial z^L} \times \frac{\partial z^L} {\partial w^L} 
 ```

Now let's break this down from left to right. To make it simpler for understanding the derivation, let's remove the layered generalization (not consider the superscript $L$). For the outermost layer, $a$ is essentially $\hat{y}$.

```math
\begin{aligned}
E &= -(y \ln(\hat{y}) + (1 - y) \ln(1-\hat{y})) \\
\frac{\partial E} {\partial \hat{y}} &= -\left(\frac{y}{\hat{y}} - \frac{1-y}{1-\hat{y}}\right) \\
&= \frac{\hat{y}-y}{\hat{y}(1-\hat{y})} \\[15pt]
&\text{let's plug back the generalized notation:} \\
\frac{\partial E} {\partial a^L} &= \frac{a^L-y}{a^L(1-a^L)}
\end{aligned}
```

Now the second term,

```math
\begin{aligned}
\hat{y} = \sigma{(z)} &= \frac{1}{1 + e^{-z}} \\[10pt]
\frac{\partial \hat{y}} {\partial z} &= \frac{\partial \sigma{(z)}} {\partial z} \\
&= \frac {e^{-z}} {(1 + e^{-z})^2} \\
&= \frac {e^{-z}} {(1 + e^{-z})} \times \frac {1} {(1 + e^{-z})} \\
&= (1 - \sigma(z)) \sigma(z)  \qquad \left(\text{since } 1 - \sigma(z) = 1 - \frac{1}{1 + e^{-z}} = \frac{e^{-z}}{1+e^{-z}}\right) \\[10pt]
&\text{so the generalized form becomes:} \\
\frac{\partial a^L} {\partial z^L} &= a^L (1 - a^L)
\end{aligned}
```

And for the last term,

```math
\begin{aligned}
z &= w \cdot x + b \\
\frac{\partial z} {\partial w} &= x \\[10pt]
&\text{or can be generalized as:} \\
\frac{\partial z^L} {\partial w^L} &= a^{L-1}
\end{aligned}
```

Altogether, 
```math
\begin{aligned}
\frac{\partial E} {\partial w^L} &= \frac{a^L-y}{a^L(1-a^L)} \times a^L (1 - a^L) \times a^{L-1} \\
&= (a^L-y) \times a^{L-1} \\
\end{aligned}
```

Similarly for the bias term,
```math
\begin{aligned}
\frac{\partial E} {\partial b^L} &= \frac{\partial E} {\partial a^L} \times \frac{\partial a^L} {\partial z^L} \times \frac{\partial z^L} {\partial b^L} \\
&= a^L-y
\end{aligned}
```

Now let's see how the error propagates weight updates from the hidden layer to the input layer.
```math
\begin{aligned}
\frac{\partial E} {\partial w^{L-1}} &= \frac{\partial E} {\partial a^L} \times \frac{\partial a^L} {\partial z^L} \times \frac{\partial z^L} {\partial a^{L-1}} \times \frac{\partial a^{L-1}} {\partial z^{L-1}}  \times \frac{\partial z^{L-1}} {\partial w^{L-1}} \\
&= (a^L-y) \times w^L \times a^{L-1}(1-a^{L-1}) \times a^{L-2} \\[10pt]
&\text{this is generalised as:} \\
\delta_L &= (a^L-y) \times w^L \times a^{L-1}(1-a^{L-1}) \\
\frac{\partial E} {\partial w^{L-1}} &= \delta_L . a^{L-2} \qquad \text{ for the input layer } a^{L-2} \text{ is } x \text{.} \\
\end{aligned}
```

This is how backpropagation works. But how exactly are the weights updated? This is where gradient descent comes into picture.
```math
w_{new} = w_{old} - \eta \frac{\partial E} {\partial w^{L}}
```
Let's not forget the bias term. The gradient stays the same and the bias update becomes:
```math
\begin{aligned}
\delta_L &= (a^L-y) \times w^L \times a^{L-1}(1-a^{L-1}) \\
b_{new} = b_{old} - \eta \frac{\partial E} {\partial b^{L}}
\end{aligned}
```

## Summary
* Initialize weights and biases randomly.
* Perform a forward pass.
* Calculate loss.
* Perform backward pass.
* Update weights and biases.
* Repeat for $n$ epochs.

### Next: Activation and loss functions and their derivations

https://com-cog-book.github.io/com-cog-book/features/multilayer-perceptron.html  
https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning  
https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/readings/L05%20Multilayer%20Perceptrons.pdf