import numpy as np

class Scalar:
    def __init__(self, data, children=None):
        self.data = data
        self.previous = children
        self.backward = None
        self.gradient = 0

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, node):
        return Scalar(self.data + node.data, (self, node))

    def __mul__(self, node):
        return Scalar(self.data * node.data, (self, node))
    

a = Scalar(3.5)
b = Scalar(2.0)

print(a + b) 
