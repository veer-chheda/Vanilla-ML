import numpy as np

class SGD:
    def __init__(self, lr=0.02, momentum=0):
        self.lr = lr
        self.momentum = momentum
        self.weight_update = None

    def update(self, w, gradient):
        if self.weight_update is None:
            self.weight_update = np.zeros(np.shape(w))
        self.weight_update = self.momentum * self.weight_update + (1 - self.momentum) * gradient
        return w - self.lr * self.weight_update

class AdaGrad:
    def __init__(self, lr=0.02):
        self.lr = lr
        self.G_diag = None
    
    def update(self, w, gradient):
        if self.G_diag is None:
            self.G_diag = np.zeros(np.shape(w)) #In theory, G_diag is a diagonal matrix. However, it is computationally very expensive to store the gradient in a diagonal matrix. This essentially is a dot product of [1] and G_diag and works the same.
        self.G_diag += np.square(gradient)
        return w - self.lr * gradient / np.sqrt(self.G_diag + 1e-8)

class RMSProp:
    def __init__(self, lr=0.02, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.moving_average = None

    def update(self, w, gradient):
        if self.moving_average is None:
            self.moving_average = np.zeros(np.shape(gradient))
        
        self.moving_average = self.beta * self.moving_average + (1 - self.beta) * np.square(gradient)

        return w - self.lr * gradient / (np.sqrt(self.moving_average) + 1e-8)

class Adam:
    def __init__(self, lr=0.02, b1=0.9, b2=0.9):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.m = None
        self.v = None

    def update(self, w, gradient):
        if self.m is None and self.v is None:
            self.m = np.zeros(np.shape(gradient))
            self.v = np.zeros(np.shape(gradient))
        
        self.m = self.b1 * self.m + (1 - self.b1) * gradient
        self.v = self.b2 * self.v + (1 - self.b2) * np.square(gradient)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        return w - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


# def loss_fn(w): # SGD should work best with this
#     return np.sum(w ** 2)

# def gradient_fn(w):
#     return 2 * w

# weights = np.random.uniform(low=-1000.0, high=1000.0, size=(10,))

# iterations = 10
# learning_rate = 0.02

# optimizers = {
#     "SGD (no momentum)": SGD(lr=learning_rate, momentum=0.0),
#     "SGD (with momentum)": SGD(lr=learning_rate, momentum=0.9),
#     "AdaGrad": AdaGrad(lr=learning_rate),
#     "RMSProp": RMSProp(lr=learning_rate),
#     "Adam": Adam(lr=learning_rate)
# }

# print(f"Initial Randomized Weights:\n{weights}")
# print(f"Initial Total Loss: {loss_fn(weights):.4f}\n")
# print("=" * 65)

# # Run evaluation loop
# for name, opt in optimizers.items():
#     w = np.copy(weights) 
#     print(f"Testing Optimizer: {name}")
    
#     for step in range(1, iterations + 1):
#         grad = gradient_fn(w)
#         w = opt.update(w, grad)
#         loss = loss_fn(w)
#         weights_str = ", ".join([f"{val:6.2f}" for val in w])
#         print(f"  Step {step} | Loss: {loss:8.3f} | Weights: [{weights_str}]")
        
#     print("-" * 65)
