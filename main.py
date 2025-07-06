import numpy as np

def sigmoid(x):
    #Activation function is f(x) = 1/(1+e^(-x))
    return 1/(1+np.exp(-x))

class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def feedforward(self, inputs):
        #Weight inputs, add bias, then use the activation function
        total = np.dot(self.weight, inputs) + self.bias
        return sigmoid(total)

weight = np.array([0, 1])
bias = 4
n = Neuron(weight, bias)

x = np.array([2, 3])
print(n.feedforward(x))