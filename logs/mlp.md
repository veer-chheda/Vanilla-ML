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
* Sigmoid: $\sigma(z) = \frac{1}{1 + e^-z}$
* ReLU (Rectified Linear Unit): $ReLU(z) = \max(0, z)$
* Tanh (Hyperbolic tangent): $tanh(z) = \frac{e^z - e^-z}{e^z + e^-z}$  

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

However, the output layer makes use of activations functions as thresholds to obtain outputs. For binary classification, it could be sigmoid whereas for multiclass classification, we use softmax function $\sigma(z)_i = \frac{e^{z_i}}{\sum_{i=1}^N e^{z_j}}$. For regression problems, we use the identity function $f(x) = x$

We also have cost functions or loss functions or objective functions (they have different meanings theoretically, but they refer to the same thing) that are used to measure the fit or "goodness" of a network's performance. Example: mean squared error, binary cross entropy, etc.

![Forward Propagation](/images/mlp_forward_prop.png)

If we pass the output of the hidden layer (i.e. it's activation that we derived before) into the threshold function (let's say sigmoid), we get the output $\hat{y}$.

## Backward Propagation



https://com-cog-book.github.io/com-cog-book/features/multilayer-perceptron.html
https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning
https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/readings/L05%20Multilayer%20Perceptrons.pdf