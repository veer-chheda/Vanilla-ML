(Used Claude to help plan and learn this)

Basically, I have two options to go forward with - 
i. Either hand-derive the backprop for each kind of neural network architecture (ANN, CNN, RNN, etc.)
ii. Or create a tiny autograd engine.

going with i. would mean that i can start with neural layers like Dense, Linear, etc. but it would get complicated going forward. so i guess ii. makes more sense even though its more work right now.

### Autograd Engine
An autograd engine is two-part - a data structure to record the order of operations (a computation graph) and a traversal algorithm to apply chain rule at each node.  
It can be implemented two ways - scalar or array.  
Scalar - like Karpathy's micrograd (https://github.com/karpathy/micrograd) where the DAG (directed acyclic graph) breaks a neuron's values into individual additive or multiplicative operations (hence, scalar). It becomes slower as network complexity increases.  
Array - each node holds a vector/array instead of a float. It is faster as handles entire matrices but more complicated.  

A node would typically have the following: data, gradient, reference to the operation that created it, reference to parent nodes, marker about its topological position.

Autograd works on automatic differentiation.

### Automatic Differentiation
It is a technique that is used to calculate the gradients on the inputs in a computational graph. It has two modes, forward and reverse. Forward calculates the gradient while calculating the result of the function whereas reverse calculates the function first and then the gradient, starting from the output. Reverse mode is more efficient since the number of outputs is generally less than the number of inputs [1]. [1] is a good source to read up on autodiff.

Since a scalar autograd would be a good starting point, I will be building that first and then move on to the array-valued autograd engine.

Check individual log files in /autograd folder.

[1] https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html  
[2] https://github.com/eduardoleao052/Autograd-from-scratch/blob/main/neuralforge/tensor_operations.py  
[3] https://github.com/arthurdjn/nets  
[4] https://github.com/joelgrus/autograd/tree/part06  (https://www.youtube.com/watch?v=RxmBukb-Om4 - youtube live code)
[5] https://pytorch.org/blog/overview-of-pytorch-autograd-engine/  
[6] https://cs231n.github.io/optimization-2/
