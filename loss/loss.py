import numpy as np

class mse():
    def loss(self, x, y):
        z = 0.5 * (x - y) ** 2
        return z

    def dLoss(self,x, y):
        z = x - y
        return z

    def ddLoss(self,x, y):
        # Activations, batchy
        ### Sum of squared errors
        z = np.ones(x.shape)
        return z


class cross_entropy():
    def __init__(self):
        return

    def loss(self, x, y):
        ### Binary cross entropy loss
        # z = np.sum(y * np.log(x) + (1-y) * np.log(1-x))

        ### Multiclass cross entropy loss
        z = - y * np.log(x)
        # loss = -tf.reduce_sum(y_target * tf.log(output))
        ### Sum of squares
        # diff = x - y
        # diff2 = (x - y)**2
        # summ = np.sum(diff2)
        return z

    def dLoss(self,x, y):
        z = - y / x
        ### Sum of squares
        return z

    def ddLoss(self,x, y):
        ### Cross entropy
        z = y / np.power(x, 2)
        return z


class loss():
    def __init__(self,func='mse'):
        if func=='mse':
            def loss(self, x, y):
                z = 0.5 * (y - x)**2
                return z

            def dLoss(x, y):
                z = x - y
                return z

            def ddLoss(x, y):
                #Activations, batchy
                ### Sum of squared errors
                z = - np.ones(x.shape)
                return z
        if func=='log':
            def loss(x, y):

                ### Binary cross entropy loss
                # z = np.sum(y * np.log(x) + (1-y) * np.log(1-x))

                ### Multiclass cross entropy loss
                z = - y * np.log(x)
                # loss = -tf.reduce_sum(y_target * tf.log(output))
                ### Sum of squares
                # diff = x - y
                # diff2 = (x - y)**2
                # summ = np.sum(diff2)
                return z
            def dLoss(x, y):
                z = - y / x
                ### Sum of squares
                return z
            def ddLoss(x, y):

                ### Cross entropy
                z = y / np.power(x,2)
                return z
