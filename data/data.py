
from sklearn import datasets
import pickle as p
import numpy as np

class dataset():

    def save(x, path):
        with open(path, 'wb') as f:
            p.dump(x, f)

    def load_p(name):
        with open(name, 'rb') as f:
            data = p.load(f)
        return data

    def load(self, dataset='iris', test=False):
        if test == True:
            dataset=None
            self.x_train = np.asarray([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
            self.y_train = np.asarray([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
            self.x_test = None
            self.y_test = None


        if dataset=='iris':
            # import some data to play with
            iris = datasets.load_iris()
            self.x_train = iris.data # we only take the first two features.
            self.y_train = iris.target
            y = []
            for val in self.y_train:
                if val == 0:
                    y.append([1,0,0])
                elif val == 1:
                    y.append([0,1,0])
                elif val == 2:
                    y.append([0,0,1])
            self.y_train = np.asarray(y)

            self.x_min,self.x_max = self.x_train[:, 0].min() - .5, self.x_train[:, 0].max() + .5
            y_min, y_max = self.x_train[:, 1].min() - .5, self.x_train[:, 1].max() + .5
            self.x_test = None
            self.y_test = None

        if dataset=='boston':
            import keras
            boston_housing = keras.datasets.boston_housing

            (self.x_train, self.y_train), (self.x_test, self.y_test) = boston_housing.load_data()


    def shuffle(self):
        # Shuffle the training set
        order = np.argsort(np.random.random(len(self.y_train)))
        self.x_train = self.x_train[order]
        self.y_train = self.y_train[order]


    def normalize(self):
        mean = self.x_train.mean(axis=0)
        std = self.x_train.std(axis=0)
        self.x_train = (self.x_train - mean) / std
        if not self.x_test == None: self.x_test = (self.x_test - mean) / std

        mean = self.y_train.mean(axis=0)
        std = self.y_train.std(axis=0)
        self.y_train = (self.y_train - mean)/std
        if not self.y_test == None: self.y_test = (self.y_test - mean)/std
