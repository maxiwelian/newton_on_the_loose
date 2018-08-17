import tensorflow as tf
import matplotlib as plt
import numpy as np
import math
import pickle as p

from tensorflow.examples.tutorials.mnist import input_data

def getHessianMLP(x_train, y_train, n_input, n_hidden, n_output):
    batch_size = 3
    # Each time getHessianMLP is called, we create a new graph so that the default graph (which exists a priori) won't be filled with old ops.
    g = tf.Graph()
    with g.as_default():
        # First create placeholders for inputs and targets: x_input, y_target
        x_input = tf.placeholder(tf.float32, shape=[batch_size, n_input])
        y_target = tf.placeholder(tf.float32, shape=[batch_size, n_output])

        # Start constructing a computational graph for multilayer perceptron
        ###  Since we want to store parameters as one long vector, we first define our parameters as below and then
        ### reshape it later according to each layer specification.

        parameters = tf.get_variable("other_variable", dtype=tf.float32,initializer=tf.constant(w))
        # parameters = tf.Variable(tf.concat(0, [tf.truncated_normal([n_input * n_hidden, 1]), tf.zeros([n_hidden, 1]),
        #                                        tf.truncated_normal([n_hidden * n_output, 1]), tf.zeros([n_output, 1])]))

        # parameters = tf.Variable(w)

        with tf.name_scope("hidden") as scope:
            idx_from = 0
            weights = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[ n_input *n_hidden, 1]), [n_input, n_hidden])
            idx_from = idx_from + n_input* n_hidden
            biases = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[n_hidden, 1]),
                                [n_hidden])  # tf.Variable(tf.truncated_normal([n_hidden]))
            hidden = tf.nn.sigmoid(tf.matmul(x_input, weights) + biases)
        with tf.name_scope("linear") as scope:
            idx_from = idx_from + n_hidden
            weights = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[n_hidden * n_output, 1]),
                                 [n_hidden, n_output])
            idx_from = idx_from + n_hidden * n_output
            biases = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[n_output, 1]), [n_output])
            output = tf.nn.sigmoid(tf.matmul(hidden, weights) + biases)
        # Define cross entropy loss
        # loss =  tf.reduce_sum(tf.pow(output - y_target,2))
        loss = tf.reduce_sum(tf.nn.l2_loss(output - y_target))
        # loss = tf.reduce_sum(y_target * tf.log(output) + (1-y_target) * tf.log(1-output))
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=output)
        # loss = -tf.reduce_sum(y_target * tf.log(output))

        ### Note: We can call tf.trainable_variables to get GraphKeys.TRAINABLE_VARIABLES
        ### because we are using g as our default graph inside the "with" scope.
        # Get trainable variables
        tvars = tf.trainable_variables()
        # Get gradients of loss with repect to parameters
        dloss_dw = tf.gradients(loss, tvars)[0]
        dim, _ = dloss_dw.get_shape()
        hess = []
        for i in range(dim):
            # print(i, dim)
            # tf.slice: https://www.tensorflow.org/versions/0.6.0/api_docs/python/array_ops.html#slice
            dfx_i = tf.slice(dloss_dw, begin=[i, 0], size=[1, 1])
            ddfx_i = tf.gradients(dfx_i, parameters)[0]  # whenever we use tf.gradients, make sure you get the actual tensors by putting [0] at the end
            hess.append(ddfx_i)

        print('exit')
        hess = tf.squeeze(hess)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            x_train = np.asarray([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
            y_train = np.asarray([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
            # x_train = np.asarray([[0, 1, 1], [1, 0, 0], [0, 0, 1]])
            # y_train = np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            ex = np.random.random([batch_size, n_output])
            feed_dict = {x_input: x_train,
                         y_target: y_train}
            # print(sess.run(loss, feed_dict))
            # print(hess.get_shape())
            # hess2 = sess.run(hess, feed_dict)
            # print(hess2)
            hess2 = sess.run(hess, feed_dict)
            grads  = sess.run(dloss_dw, feed_dict)
            hidden2 = sess.run(hidden, feed_dict)
            # with open('weight.csv', 'w+') as f:
            #     np.savetxt(f, hess2, delimiter=',')
            with open('grads.csv', 'w+') as f:
                np.savetxt(f, grads, delimiter=',')
            return hess2, grads , sess.run(loss, feed_dict), hidden2

if __name__ =='__main__':
    from sklearn import datasets

    # import some data to play with
    iris = datasets.load_iris()
    x_train = iris.data  # we only take the first two features.
    y_train = iris.target
    y = []
    for val in y_train:
        if val == 0:
            y.append([1, 0, 0])
        elif val == 1:
            y.append([0, 1, 0])
        elif val == 2:
            y.append([0, 0, 1])
    y_train = np.asarray(y)

    x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5
    y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5
    x_test = None
    y_test = None

    epochs = range(100)
    for epoch in epochs:
        Hessian, dloss_dw, loss, hidden2 = getHessianMLP(x_train, y_train, n_input=3, n_hidden=3, n_output=3)
        # print(Hessian)
        # print(dloss_dw)
        print(loss)
        eig, v = np.linalg.eigh(Hessian)
        invHessian = np.linalg.pinv(Hessian)

        eig, v = np.linalg.eigh(Hessian)

        w += - np.dot(invHessian, dloss_dw)
        w = w.astype(np.float32)
        # print('end')

