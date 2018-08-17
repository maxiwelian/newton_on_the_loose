import numpy as np

### Activation function and derivatives
class sigmoid():
    def fActivation(self, x):
            return 1. / (1. + np.exp(-x))
    def dfActivation(self,x):
        #(1. / (1. + np.exp(-x)))*(1. - (1. / (1. + np.exp(-x))))
        return x*(1-x)
    def ddfActivation(self, x):
        #(1. / (1. + np.exp(-x))) * (1. - (1. / (1. + np.exp(-x)))) * (1. - (2. / (1. + np.exp(-x))))
        return x*(1-x)*(1-2*x)

class Leaky():
    def fActivation(self, x):
        epsilon = 0.01
        return np.maximum(epsilon * x, x)
    def dfActivation(self, x):
        epsilon = 0.01
        z = 1. * (x > epsilon)
        try:
            z[np.where(z == 0.)] = epsilon
        except:
            z = epsilon
        return z
    def ddfActivation(self, x):
        return 0

class softmax():
    def fActivation(self, x):
        return None

    def dfActivation(self, x):
        return None

    def ddfActivation(self, x):
        return None






                    ### Same functions. Use the second one if the input is the activations
# def daFunc(x): # Derivative of the sigmoid
#     tmp = aFunc(x)
#     return tmp * (1 - tmp)
# def dfda(x):# Derivative of the sigmoid
#     return x * (1 - x)
# def dfdada(x):
#     return x * (1 - x) * (1-2*x)
# 
# def ddaFunc(x):
#     tmp = aFunc(x)
#     return tmp * (1 - tmp) * (1 - 2 * tmp)