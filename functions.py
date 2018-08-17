import numpy as np
import pickle as p

def rollDict(dictionary):
    return np.array(sum(dictionary.values), [])

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)





# ### Activation function and derivatives
# def aFunc(x): # Activatation function sigmoid
#     return 1. / (1. + np.exp(-x))
#
#
#
# ### Same functions. Use the second one if the input is the activations
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
#
#
#
#

# #
# def load_p(path):
#     with open(path, 'rb') as f:
#         x = p.load(f)
#     return x
#
# def rollWeights(weights):
#     w = np.array([])
#     for (k, v) in weights.iteritems():
#         tmp = np.reshape(v, (v.size, 1))
#         w = np.append(w, tmp)
#     return w
#
# def unrollWeights(rolledWeights, nL, topology):
#     weights = {}
#     counter = 0
#     for i in range(nL):
#         if i == nL-1:
#             layer2 = topology[i+1]
#         else:
#             layer2 = topology[i+1]-1
#         nWeightsLayer = topology[i] * layer2
#         weights[i] = np.reshape(rolledWeights[counter:counter+nWeightsLayer], (topology[i], layer2))
#         counter += nWeightsLayer
#     return weights
#
# def fwdProp(inputs, weights, sizeBatch, nL):
#     ## Initialise first layer
#     activations = {}
#     activations[0] = np.concatenate([inputs[0], np.ones((sizeBatch, 1))], axis=1)
#     for i in range(nL):
#         ## Activations
#         inputs[i + 1] = np.dot(activations[i], weights[i])
#         if i == nL - 1:
#             activations[i + 1] = aFunc(inputs[i + 1])
#         else:
#             activations[i + 1] = np.concatenate([aFunc(inputs[i + 1]), np.ones((sizeBatch, 1))], axis=1)
#     return inputs, activations
#
#
# def det(l):
#     n=len(l)
#     if (n>2):
#         i=1
#         t=0
#         sum=0
#         while t<=n-1:
#             d={}
#             t1=1
#             while t1<=n-1:
#                 m=0
#                 d[t1]=[]
#                 while m<=n-1:
#                     if (m==t):
#                         u=0
#                     else:
#                         d[t1].append(l[t1][m])
#                     m+=1
#                 t1+=1
#             l1=[d[x] for x in d]
#             sum=sum+i*(l[0][t])*(det(l1))
#             i=i*(-1)
#             t+=1
#         return sum
#     else:
#         return (l[0][0]*l[1][1]-l[0][1]*l[1][0])
#
# def is_pos_def(x):
#     return np.all(np.linalg.eigvals(x) > 0)
#
# def check_symmetric(a, tol=1e-8):
#     return np.allclose(a, a.T, atol=tol)
