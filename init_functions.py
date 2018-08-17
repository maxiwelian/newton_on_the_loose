from functions import *

def init_idxlistLayer(nLayers, topology):
    counter = 0
    idxLayers = {}
    idxlistLayers = {}
    for layer in range(nLayers):
        if not layer == nLayers-1:
            idxlistLayers[layer] = range(counter, counter + topology[layer])
        else:
            idxlistLayers[layer] = range(counter, counter + topology[layer])
        for _ in range(topology[layer]):
            idxLayers[counter] = layer
            counter += 1
    return idxlistLayers

### Initialise the weights to random values
def init_Weights(nL, topology):
    weights = {}
    for layer in range(nL):
        # b1 = np.zeros(topology[layer+1])
        # If not the second to last layer
        if not layer + 1 == nL:
            weights[layer] = np.clip(np.random.normal(0., 0.1, (topology[layer], topology[layer + 1] - 1)), -1, +1)
            # b2 = np.zeros(topology[layer+1]-1)
            b2 = np.clip(np.random.normal(0.,0.1, topology[layer+1]-1), -1., +1.)
            # weights[layer] = np.concatenate([weights[layer], [b1]], axis=0)
            weights[layer][-1, :] = b2
        else:
            weights[layer] = np.clip(np.random.normal(0., 0.1, (topology[layer], topology[layer + 1])), -1, +1)
            # b2 = np.zeros(topology[layer+1])
            # weights[layer] = np.concatenate([weights[layer], [b1]], axis=0)
            b2 = np.clip(np.random.normal(0., 0.1, topology[layer + 1] - 1), -1., +1.)
            weights[layer][-1, :] = b2
        # idxlistLayers is a dictionary where each entry is a layer and the corresponding list are the node indexes
        # idxLayers is a dictionary of node indexes : layer
    return weights

def init_WeightsTest(nL, topology):
    weights = {}
    # weights[0] = np.asarray([[0.1, 0.2], [0.1, 0.05], [-0.1, 0.1]])
    # weights[1] = np.asarray([[-0.2, -0.05], [-0.1, 0.2], [0.1, 0.1]])
    w = np.asarray([0.1, 0.2, 0.1, 0.05, -0.1, 0.1, 0.3, -0.2, 0.1,0.,0.,0., -0.2, -0.05, -0.1, 0.2, 0.1, 0.1, 0.3, -0.2, 0.1,0.,0.,0.])
    weights[0] = w[0:12].reshape((4, 3))
    weights[1] = w[12:24].reshape((4,3))
    return weights

def init_b(g, rolledActivations, idxlistLayers, nL, inputs, activations, batchy, nNodes):
    b = np.zeros_like(g)
    H = np.zeros_like(rolledActivations[0])
    H[idxlistLayers[nL]] = np.sum(
                                        dfdada(activations[nL]) * dLoss(activations[nL], batchy) +
                                        dfda(activations[nL])**2 * ddLoss(activations[nL],batchy), axis=0)

    for m in idxlistLayers[nL]:
        for i in range(nNodes):
            # Except for input nodes and units n which are in a lower layer than unit i
            b[i, m] = g[i, m] * H[m]

    return b