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
#print(n.feedforward(x))

############################################################################################

class OurNeuralNetwork:
    def __init__(self):
        weight = np.array([0,1])
        bias = 0

        # The Neuron class here is from the previous section
        self.h1 = Neuron(weight, bias)
        self.h2 = Neuron(weight, bias)
        self.o1 = Neuron(weight, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

network = OurNeuralNetwork()
x = np.array([2, 3])
#print(network.feedforward(x))


############################################################################################

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

