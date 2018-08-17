
### Saddle points http://www.santanupattanayak.com/2017/12/19/newtons-method-vs-gradient-descent-method-in-tacking-saddle-points-in-non-convex-optimization/
import numpy as np
import numpy.linalg as LA
from NN import NN
from functions import check_symmetric
import pickle as p
from data.data import dataset
from loss.loss import loss
import time
from loss.loss import mse, cross_entropy
from activation.activations import sigmoid, Leaky, softmax
import matplotlib.pyplot as plt

test = True
dataset = dataset()
dataset.load(test=test,dataset = 'iris')
# dataset.normalize()
# dataset.shuffle()
x_train = dataset.x_train
y_train = dataset.y_train
x_test = dataset.x_test
y_test = dataset.y_test

### Neural network topology
# nNodes + bias node
topology = np.array([x_train.shape[1]+1, x_train.shape[1]+1, 3])

### Initialise Neural Network
loss = mse()
activation = sigmoid()
NN = NN(topology, loss=loss, activation=activation, test=test)

epochs = range(10000)
loss_tracker = []
start = time.time()
for epoch in epochs:
    ### Init Hessian
    Hessian = np.zeros((NN.nWeights, NN.nWeights))
    dEdw = np.zeros(NN.nWeights)

    for data, batchy in zip(x_train, y_train):
        data = np.asarray([data])
        batchy = np.asarray([batchy])

        ### *** Forward propagation *** ###
        NN.fwdProp(data)

        ### *** Calculation of the Hessian *** ###
        Hessian += - NN.Hessian(batchy)

        ### *** Calculation of the gradients *** ###
        # dEdw += NN.dEdw_solo(batchy)
        dEdw += NN.dEdw()
    # print('Epoch %i' % epoch, ' complete')

    print(check_symmetric(Hessian))

    # dEdw = dEdw# / len(y_train)
    ### Levenberg gradient descent method
    # alpha = 0.1
    # Hessian = Hessian + np.identity(nWeights) * alpha

    ### Egvs
    # eig, v = LA.eigh(Hessian)
    # invHessian = np.identity(NN.nWeights) / np.abs(eig)
    # Hessian = np.abs(eig * np.identity(nWeights))
    # eig = eig/np.min(np.abs(eig))
    # eig = eig*np.identity(nWeights)

    ### Getting the inverse
    invHessian = np.linalg.pinv(Hessian)

    ### Performing the updates
    rolledWeights = NN.rollWeights()
    rolledWeights += 0.15* np.dot(invHessian,dEdw)
    # rolledWeights += - 0.1 * dEdw
    NN.unrollUpdateW(rolledWeights)


    # batchy = [[y] for y in y_train]
    batchy = y_train
    data = x_train
    NN.fwdProp(data)
    print(NN.activations[2])
    print(batchy)
    print(NN.activations[2] - batchy)
    # print(batchy[0:10])
    x = np.sum(loss.loss(NN.activations[2], batchy)) / len(y_train)
    loss_tracker.append(x)
    print('Epoch:', epoch, 'Loss:', x)





print(time.time() - start)
print('Made it')

### For saves
# with open('weight_bishop.csv', 'w+') as f:
#     np.savetxt(f, Hessian, delimiter=',')
# with open('grads_bishop.csv', 'w+') as f:
#     np.savetxt(f, dEdw, delimiter=',')