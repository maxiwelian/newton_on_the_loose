from functions import *

def calc_Hessian(nWeights, nL, idxlistLayers, rolledActivations, sigma, b, g):
    # Take the first node of the first layer
    Hessian = np.zeros((nWeights, nWeights), dtype=np.float16)
    axis_0 = 0
    for layer_1 in idxlistLayers:
        if layer_1 == nL:
            continue
        elif layer_1 == nL - 1:
            for i in idxlistLayers[layer_1]:
                for j in idxlistLayers[layer_1+1]:
                    axis_1 = 0
                    for layer_2 in idxlistLayers:
                        if layer_2 == nL:
                            continue
                        elif layer_2 == nL-1:
                            for n in idxlistLayers[layer_2]:
                                for l in idxlistLayers[layer_2+1]:
                                    if axis_0 <= axis_1:
                                        if i == 4 and j == 8 and n == 4 and l == 9:
                                            d = dfda(rolledActivations[:, n])
                                            e = sigma[l]
                                            f = g[j, n]
                                            a = b[j, l]
                                            s = rolledActivations[:, n]
                                            p = rolledActivations[:, i]

                                        derivative = dfda(rolledActivations[:, n])
                                        Hessian[axis_0, axis_1] = np.sum(
                                            rolledActivations[:, i] * sigma[l] * derivative * g[j, n] + \
                                            rolledActivations[:, i] * rolledActivations[:, n] * b[j, l], axis=0)
                                    else:

                                        Hessian[axis_0, axis_1] = Hessian[axis_1, axis_0]
                                    axis_1 += 1
                        elif layer_2 < nL-1:
                            for n in idxlistLayers[layer_2]:
                                for l in idxlistLayers[layer_2 + 1][:-1]:
                                    if axis_0 <= axis_1:
                                        derivative = dfda(rolledActivations[:, n])
                                        Hessian[axis_0, axis_1] = np.sum(
                                            rolledActivations[:, i] * sigma[l] * derivative * g[j, n] + \
                                            rolledActivations[:, i] * rolledActivations[:, n] * b[j, l], axis=0)
                                    else:
                                        Hessian[axis_0, axis_1] = Hessian[axis_1, axis_0]
                                    axis_1 += 1
                    axis_0 += 1

        elif layer_1 < nL-1:
            for i in idxlistLayers[layer_1]:
                for j in idxlistLayers[layer_1+1][:-1]:
                    axis_1 = 0
                    for layer_2 in idxlistLayers:
                        if layer_2 == nL:
                            continue
                        elif layer_2 < nL-1:
                            for n in idxlistLayers[layer_2]:
                                for l in idxlistLayers[layer_2 + 1][:-1]:
                                    if axis_0 <= axis_1:
                                        derivative = dfda(rolledActivations[:, n])
                                        Hessian[axis_0, axis_1] = np.sum(
                                            rolledActivations[:, i] * sigma[l] * derivative * g[j, n] + \
                                            rolledActivations[:, i] * rolledActivations[:, n] * b[j, l], axis=0)
                                    else:
                                        Hessian[axis_0, axis_1] = Hessian[axis_1, axis_0]
                                    axis_1 += 1
                        elif layer_2 == nL - 1:
                            for n in idxlistLayers[layer_2]:
                                for l in idxlistLayers[layer_2 + 1]:
                                    if axis_0 <= axis_1:
                                        derivative = dfda(rolledActivations[:, n])
                                        Hessian[axis_0, axis_1] = np.sum(
                                            rolledActivations[:, i] * sigma[l] * derivative * g[j, n] + \
                                            rolledActivations[:, i] * rolledActivations[:, n] * b[j, l], axis=0)
                                    else:
                                        Hessian[axis_0, axis_1] = Hessian[axis_1, axis_0]
                                    axis_1 += 1
                    axis_0 += 1



    return Hessian

def calc_dEdw(idxlistLayers, nL, nWeights, rolledActivations, sigma):
    counter = 0
    dEdw = np.zeros(nWeights)
    for layer_1 in idxlistLayers:
        if layer_1 < nL - 1:
            for i in idxlistLayers[layer_1]:
                for j in idxlistLayers[layer_1 + 1][:-1]:
                    dEdw[counter] = np.sum(sigma[j] * rolledActivations[:, i], axis=0)
                    counter += 1
        elif layer_1 == nL-1:
            for i in idxlistLayers[layer_1]:
                for j in idxlistLayers[layer_1+1]:
                    dEdw[counter] = np.sum(sigma[j] * rolledActivations[:, i],axis=0)
                    counter += 1
        elif layer_1 == nL:
            continue
    return dEdw

    # Calculate the g values
    # gDict is a dictionary len(layers) each entry containing the indexes (i, l) i -> l of nonzero g terms split into layers
    # this dictionary should first use the first layer then the second etc
    # For each layer saved by the gdict
    # Here rolledInput has been substituted for rolled activations to account for bias nodes, will need to change derivatives if activation changes
def calc_g(nNodes, idxlistLayers, weightMatrix, rolledActivations):
    g = np.identity(nNodes)
    for idx1, layer1 in idxlistLayers.iteritems():
        for i in layer1:
            for idx2, layer2 in idxlistLayers.iteritems():
                for l in layer2:
                    if idx1 >= idx2:
                        pass
                    else:
                        for r in idxlistLayers[idx2-1]:
                            weight = weightMatrix[r, l]  # All weights connected to l. weight r -> l
                            derivative = np.sum(dfda(rolledActivations[:, r]), axis=0)
                            g[i, l] += weight * derivative * g[i, r]
    return g

def calc_sigma(nL, idxlistLayers, weightMatrix, rolledActivations, activations, batchy):
    sigma = np.zeros_like(rolledActivations[0])
    x = dfda(activations[nL]) * dLoss(activations[nL], batchy)
    test = batchy - activations[nL]
    sigma[idxlistLayers[nL]] = np.sum(x, axis=0)
    x = list(reversed(range(nL)))
    for i in x:
        if i == nL -1:
            for n in idxlistLayers[i]:
                for r in idxlistLayers[i+1]:
                    weight_sigma = weightMatrix[n, r] * sigma[r]
                    derivative = np.sum(dfda(rolledActivations[:, n]), axis=0)
                    sigma[n] += derivative * weight_sigma
                    ## Use the two print functions below to prove the equivalence of dfda function and the full notation
                    # print 'dfda', dfda(rolledActivations[:, r])
                    # print 'orig', rolledActivations[:, r] * (1 - rolledActivations[:, r])
        else:
            for n in idxlistLayers[i]:
                for r in idxlistLayers[i+1][:-1]:
                    weight_sigma = weightMatrix[n, r] * sigma[r]
                    derivative = np.sum(dfda(rolledActivations[:, n]), axis=0)
                    sigma[n] += derivative * weight_sigma
                    ## Use the two print functions below to prove the equivalence of dfda function and the full notation
                    # print 'dfda', dfda(rolledActivations[:, r])
                    # print 'orig', rolledActivations[:, r] * (1 - rolledActivations[:, r])

    return sigma

def calc_b(b, g, nL, idxlistLayers, nNodes, rolledActivations, weightMatrix, sigma):
    x = list(reversed(range(nL)))
    for layer in x:
        if layer == nL-1:
            for i in range(nNodes):
                for n in idxlistLayers[layer]:
                    derivative = dfda(rolledActivations[:, n])
                    dderivative = dfdada(rolledActivations[:, n])
                    for r in idxlistLayers[layer+1]:
                        sum1 = np.sum(dderivative * g[i,n] * weightMatrix[n, r] * sigma[r] + derivative * weightMatrix[n, r] * b[i, r])
                        b[i, n] += sum1
        else:
            for i in range(nNodes):
                for n in idxlistLayers[layer]:
                    dderivative = dfdada(rolledActivations[:, n])
                    derivative = dfda(rolledActivations[:, n])
                    for r in idxlistLayers[layer+1][:-1]:
                        sum1 = np.sum(dderivative * g[i,n] *  weightMatrix[n, r] * sigma[r] + derivative * weightMatrix[n, r] * b[i, r])
                        b[i, n] += sum1

    return b

def calc_matWeights(nNodes, nL, weightsDict, idxlistLayers):
    # Matrix 1 node to all with the self weight = 0 and any mapping i -> j where lay(j) <= lay(i) = 0
    weights = np.zeros((nNodes, nNodes))
    # Take the layers in order and fill in the matrix with weights from the layer weight matrices
    for layer in idxlistLayers:
        if layer == nL:
            pass
        elif layer == nL-1:
            for idx_0, axis_0 in enumerate(idxlistLayers[layer]):
                for idx_1, axis_1 in enumerate(idxlistLayers[layer + 1]):
                    weight = weightsDict[layer][idx_0, idx_1]  # wij
                    weights[axis_0, axis_1] = weight
        else:
            for idx_0, axis_0 in enumerate(idxlistLayers[layer]):
                for idx_1, axis_1 in enumerate(idxlistLayers[layer + 1][:-1]):
                    weight = weightsDict[layer][idx_0, idx_1] #wij
                    weights[axis_0, axis_1] = weight
    return weights

def calc_nWeights(weightsDict):
    nWeights = 0
    for (k, v) in weightsDict.iteritems():
        nWeights += v.size
    return nWeights