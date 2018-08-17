import numpy as np

class NN():

    def __init__(self, topology, loss, activation, test=False):

        self.topology = [x-1 for x in topology[:-1]]
        self.topology.append(topology[-1])
        self.topologyBias = topology
        self.nNodes = int(np.sum(topology))
        self.listNodes = range(int(self.nNodes))
        # self.nLayers = topology.size
        self.nL = int(topology.size-1)
        self.layers = range(self.nL+1)

        ### Algorithm parameters
        self.init_W(test=test)
        self.calc_nWeights()
        self.idxlistLayers = self.init_idxlistLayer()

        ### Rolled matrices
        self.rollAct = []
        self.rollInput = []

        ### Hessian Calculation
        self.g = np.identity(int(self.nNodes))
        self.b = np.zeros_like(self.g)
        self.WMatrix = np.zeros_like(self.g)
        self.sigma = np.zeros(self.nNodes)

        ### Activations & inputs
        self.activations = {}
        self.inputs = {}

        ### Loss
        self.loss = loss

        ### Activation
        self.activation = activation

    def fwdProp(self, data):
        self.sizeBatch = len(data)
        self.inputs = {}
        self.inputs[0] = data
        self.activations = {}
        self.activations[0] = np.concatenate([self.inputs[0], np.ones((self.sizeBatch,1))], axis=1)
        for i in self.layers[:-1]:
            ## Activations
            self.inputs[i + 1] = np.dot(self.activations[i], self.W[i])
            self.activations[i + 1] = np.concatenate([self.activation.fActivation(self.inputs[i + 1]),  np.ones((self.sizeBatch,1))], axis=1)
        ### Last layer
        self.activations[self.nL] = self.activation.fActivation(self.inputs[self.nL])
        for i in self.layers[:-1]:
            self.inputs[i] = np.concatenate([self.inputs[i], np.zeros((self.sizeBatch, 1))], axis=1)
        return

    def Hessian(self, batchy):
        self.rollInput = self.rollDict(self.inputs)
        self.rollAct = self.rollDict(self.activations)
        ## WMatrix
        self.calc_WMatrix()
        ## g
        self.calc_g()
        ## sigma
        self.calc_sigma(batchy)
        ## b
        self.calc_b(batchy)
        ##WMatrix
        self.calc_WMatrix()
        ## Hessian
        Hessian = self.calc_Hessian()

        return Hessian

    def calc_g(self):
        self.g = np.identity(self.nWeights)
        for layer1 in self.layers[1:]:
            for i in self.idxlistLayers[layer1]:

                for layer2 in self.layers[layer1+1:]:
                    for l in self.idxlistLayers[layer2]:

                        for r in self.idxlistLayers[layer2-1]:
                            weight = self.WMatrix[r, l]  # All weights connected to l. weight r -> l
                            derivative = self.activation.dfActivation(self.rollAct[r]) # Minus one bias for every layer
                            self.g[i, l] += weight * derivative * self.g[i, r]
        return

    def calc_sigma(self, batchy):
        self.sigma = np.zeros_like(self.rollAct)
        self.sigma[self.idxlistLayers[self.nL]]= self.activation.dfActivation(self.activations[self.nL]) * self.loss.dLoss(self.activations[self.nL], batchy)
        for layer1 in self.layers[:-1]:
            layer2 = layer1+1
            for n in self.idxlistLayers[layer1]:
                derivative = self.activation.dfActivation(self.rollAct[n])
                for r in self.idxlistLayers[layer2][:self.topology[layer2]]:
                    weight_sigma = self.WMatrix[n, r] * self.sigma[r]
                    self.sigma[n] += derivative * weight_sigma
        return

    def calc_b(self, batchy):
        H = np.zeros_like(self.rollAct)
        self.b = np.zeros_like(self.g)

        # print(H[self.idxlistLayers[self.nL]])
        # print(self.idxlistLayers[self.nL])
        H[self.idxlistLayers[self.nL]] = self.activation.ddfActivation(self.activations[self.nL]) * \
                                         self.loss.dLoss(self.activations[self.nL], batchy) + \
                                         self.activation.dfActivation(self.activations[self.nL])**2 * \
                                         self.loss.ddLoss(self.activations[self.nL], batchy)

        for m in self.idxlistLayers[self.nL]:
            for i in self.listNodes:
                # Except for input nodes and units n which are in a lower layer than unit i
                self.b[i, m] = self.g[i, m] * H[m]

        for layer1 in self.layers[::-1][1:]:
            layer2 = layer1+1
            for i in self.listNodes:
                for n in self.idxlistLayers[layer1]:
                    derivative = self.activation.dfActivation(self.rollAct[n])
                    dderivative = self.activation.ddfActivation(self.rollAct[n])
                    for r in self.idxlistLayers[layer2][:self.topology[layer2]]:
                        sum1 = dderivative * self.g[i, n] * self.WMatrix[n, r] * self.sigma[r] + derivative * self.WMatrix[n, r] * self.b[i, r]
                        self.b[i, n] += sum1
        return

    def calc_Hessian(self):
        # Take the first node of the first layer
        Hessian = np.zeros((self.nWeights, self.nWeights))
        axis_0 = 0
        for layer1 in self.layers[:-1]:
            layer2 = layer1+1
            for i in self.idxlistLayers[layer1]:
                for j in self.idxlistLayers[layer2][:self.topology[layer2]]:
                    axis_1 = 0


                    for layer3 in self.layers[:-1]:
                        layer4 = layer3 + 1
                        for n in self.idxlistLayers[layer3]:
                            for l in self.idxlistLayers[layer4][:self.topology[layer4]]:
                                if axis_0 <= axis_1:
                                    derivative = self.activation.dfActivation(self.rollAct[n])

                                    Hessian[axis_0, axis_1] = self.rollAct[i] \
                                                              * self.sigma[l] \
                                                              * derivative \
                                                              * self.g[j, n] \
                                                              + self.rollAct[i] * self.rollAct[n] * self.b[j, l]
                                else:
                                    Hessian[axis_0, axis_1] = Hessian[axis_1, axis_0]
                                axis_1 += 1
                    axis_0 += 1
        return Hessian

    def calc_WMatrix(self):
        # Matrix 1 node to all with the self weight = 0 and any mapping i -> j where lay(j) <= lay(i) = 0
        self.WMatrix = np.zeros((self.nNodes, self.nNodes))
        # Take the layers in order and fill in the matrix with weights from the layer weight matrices
        for layer1 in self.layers[:-1]:
            layer2 = layer1+1
            for idx_0, axis_0 in enumerate(self.idxlistLayers[layer1]):
                for idx_1, axis_1 in enumerate(self.idxlistLayers[layer2][:self.topology[layer2]]):
                    weight = self.W[layer1][idx_0, idx_1]  # wij
                    self.WMatrix[axis_0, axis_1] = weight

    def calc_nWeights(self):
        self.nWeights = 0
        for (k, v) in self.W.items():
            self.nWeights += v.size

    def dEdw(self):
        counter = 0
        dEdw = np.zeros(self.nWeights)
        for layer1 in self.layers[:-1]:
            layer2 = layer1+1
            for i in self.idxlistLayers[layer1]:
                for j in self.idxlistLayers[layer2][:self.topology[layer2]]:
                    dEdw[counter] = self.sigma[j] * self.rollAct[i]
                    counter += 1
        return dEdw

    def dEdw_solo(self, batchy):

        self.rollInput = self.rollDict(self.inputs)
        self.rollAct = self.rollDict(self.activations)
        ## WMatrix
        self.calc_WMatrix()
        ## sigma
        self.calc_sigma(batchy)
        counter = 0
        dEdw = np.zeros(self.nWeights)
        for layer1 in self.layers[:-1]:
            layer2 = layer1+1
            for i in self.idxlistLayers[layer1]:
                for j in self.idxlistLayers[layer2][:self.topology[layer2]]:
                    dEdw[counter] = self.sigma[j] * self.rollAct[i]
                    counter += 1
        return dEdw

    def rollWeights(self):
        w = np.array([])
        for (k, v) in self.W.items():
            tmp = np.reshape(v, (v.size, 1))
            w = np.append(w, tmp)
        return w

    def rollDict(self, D):
        w = np.array([])
        for (k, v) in D.items():
            tmp = np.reshape(v, (v.size, 1))
            w = np.append(w, tmp)
        return w

    def unrollUpdateW(self, rolledWeights):
        counter = 0
        for layer1 in self.layers[:-1]:
            layer2 = layer1+1
            nwlayer1 = self.topologyBias[layer1]
            nwlayer2 = self.topology[layer2]
            nWeightsLayer = nwlayer1 * nwlayer2
            self.W[layer1] = np.reshape(rolledWeights[counter:counter + nWeightsLayer], (nwlayer1, nwlayer2))
            counter += nWeightsLayer
        return self.W

    def init_W(self, test=False):
        self.W = {}
        if test==False:
            for layer in range(self.nL):
                self.W[layer] = np.clip(np.random.normal(0, 0.2, (self.topologyBias[layer], self.topology[layer+1])), -1,+1)
                b2 = np.zeros(self.topology[layer+1])
                self.W[layer][-1, :] = b2
        else:
            w = np.asarray([0.1, 0.2, 0.1, 0.05, -0.1, 0.1, 0.3, -0.2, 0.1, 0, 0, 0, -0.2, -0.05, -0.1, 0.2, 0.1, 0.1, 0.3, -0.2, 0.1,
         0, 0, 0])
            self.W[0] = w[0:12].reshape((4, 3))
            self.W[1] = w[12:24].reshape((4, 3))

    def init_idxlistLayer(self):
        ### Create a dictionary containing the indexes of the nodes in the neural network to be referenced for the Hessian later
        counter = 0
        idxlistLayers = {}
        for layer in self.layers:
            idxlistLayers[layer] = range(counter, counter + self.topologyBias[layer])
            counter += self.topologyBias[layer]
        return idxlistLayers
