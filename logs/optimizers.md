Optimizers are algorithms that are used to minimize a loss function to maximize learning/efficiency. For neural networks, the optimizers change the weights and biases of the network to reduce losses.

## Gradient Descent
It is an optimization algorithm that iteratively updates the weights and biases to reduce loss. These iterations of changes or updates are called 'gradients' and their magnitude or effect is determined by the learning rate. This process continues till the objective function is minimized or the desired number of iterations are completed. The idea behind gradient descent is taking steps in a downhill direction (negative gradient) such that the minima is achieved.
$$\theta_{new} = \theta_{old} - \alpha . \nabla J(\theta)$$

### Stochastic Gradient Descent
It is the variant of gradient descent where a single data point is used to calculate the gradient. The key difference between gradient descent and stochastic gradient descent is that the former uses the entire dataset to calculate the gradient for an iteration, and the latter picks a random data point from the dataset and updates weights as per the gradient for that particular data point.

### Mini-batch Gradient Descent
It's a combination of stochastic and vanilla gradient descent where instead of a single data point, a batch of points are used to calculate the gradient.

### Momentum
The goal of optimization is finding the global minimum of a function. However, this can be difficult due to existence of many local minima. Or it could so be that the optimizer can get stuck in a saddle point, where the function can neither be maximized or minimized in any direction. Thus, we include momentum which can be expressed as the tendency to continue moving in the same direction as the previous gradients. This accelerates the convergence by ensuring a smoother and faster progression towards the minimum.
$$v = \beta . v_{old} + (1 - \beta).\nabla J(\theta)$$
$$\theta_{new} = \theta_{old} - \alpha . v$$

## AdaGrad
AdaGrad or Adaptive Gradient adjusts the learning rate while training. Since different parameters might require different learning rates, AdaGrad assigns a unique learning rate to each parameter and change it based on its previous gradient. A disadvantage is that it can overly decrease the gradient overtime, thus slowing down the convergence of parameters.
$$\theta_{new} = \theta_{old} - \frac{\alpha}{\sqrt{G_{diagonal}} + \epsilon} . \nabla J(\theta)$$ where, $G_{diagonal}$ is a diagonal matrix where the diagonal elements are the sum of past gradients for the respective parameter.

## RMSProp
AdaGrad keeps adding the previous gradients which makes the sum too large over time, thus effectively reducing the learning rate. RMSProp solves this by maintaining a moving average of the squared gradients. This average eventually 'forgets' old gradients so the learning rate is not reduced too much.
$$v = \beta v_{old} + (1 - \beta).(\nabla J(\theta))^2$$
$$\theta_{new} = \theta_{old} - \frac{\alpha}{\sqrt{v} + \epsilon}. \nabla J(\theta)$$

## Adam
Adam or Adaptive Moment Estimation combines RMSProp and momentum. It integrates momentum by using the moving averages of gradients and updating the learning rate for each parameter on the basis of the moving average of squared gradients.
$$m = \beta_1 . m_{old} + (1 - \beta_1).\nabla . J(\theta)$$
$$v = \beta_2 . v_{old} + (1 - \beta_2). (\nabla . J(\theta))^2$$
$$\hat{m} = \frac{m}{1 - \beta_1}$$
$$\hat{v} = \frac{v}{1 - \beta_2}$$
$$\theta_{new} = \theta_{old} - \frac{\alpha}{\sqrt{\hat{v}} + \epsilon}.\hat{m}$$
$m$ is the first-order moment (mean of gradients) which works as momentum like in RMSProp. $v$ is the second-order moment (variance of the gradients) which tells us how much the gradients are fluctuating. This helps prevent the optimizer from taking too large steps.