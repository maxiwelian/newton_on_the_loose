import tensorflow as tf
import matplotlib as plt
import numpy as np
import math
import pickle as p
from numpy import linalg as LA

from tensorflow.examples.tutorials.mnist import input_data

def getHessianMLP(n_input, n_hidden, n_output):
    batch_size = 1
    # Each time getHessianMLP is called, we create a new graph so that the default graph (which exists a priori) won't be filled with old ops.
    g = tf.Graph()
    with g.as_default():
        # First create placeholders for inputs and targets: x_input, y_target
        x_input = tf.placeholder(tf.float32, shape=[batch_size, n_input])
        y_target = tf.placeholder(tf.float32, shape=[batch_size, n_output])

        # Start constructing a computational graph for multilayer perceptron
        ###  Since we want to store parameters as one long vector, we first define our parameters as below and then
        ### reshape it later according to each layer specification.
        parameters = tf.Variable(tf.concat([tf.truncated_normal([n_input * n_hidden, 1]), tf.zeros([n_hidden, 1]),
                                               tf.truncated_normal([n_hidden * n_output ,1]), tf.zeros([n_output, 1])], 0))
        # parameters = tf.Variable(tf.concat(0, [tf.truncated_normal([n_input * n_hidden, 1]), tf.zeros([n_hidden, 1]),
        #                                        tf.truncated_normal([n_hidden * n_output, 1]), tf.zeros([n_output, 1])]))

        with tf.name_scope("hidden") as scope:
            idx_from = 0
            weights = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[ n_input *n_hidden, 1]), [n_input, n_hidden])
            idx_from = idx_from + n_input* n_hidden
            biases = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[n_hidden, 1]),
                                [n_hidden])  # tf.Variable(tf.truncated_normal([n_hidden]))
            hidden = tf.matmul(x_input, weights) + biases
        with tf.name_scope("linear") as scope:
            idx_from = idx_from + n_hidden
            weights = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[n_hidden * n_output, 1]),
                                 [n_hidden, n_output])
            idx_from = idx_from + n_hidden * n_output
            biases = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[n_output, 1]), [n_output])
            output = tf.nn.softmax(tf.matmul(hidden, weights) + biases)
        # Define cross entropy loss
        loss = -tf.reduce_sum(y_target * tf.log(output))

        ### Note: We can call tf.trainable_variables to get GraphKeys.TRAINABLE_VARIABLES
        ### because we are using g as our default graph inside the "with" scope.
        # Get trainable variables
        tvars = tf.trainable_variables()
        # Get gradients of loss with repect to parameters
        dloss_dw = tf.gradients(loss, tvars)[0]
        dim, _ = dloss_dw.get_shape()
        hess = []
        for i in range(dim):
            # tf.slice: https://www.tensorflow.org/versions/0.6.0/api_docs/python/array_ops.html#slice
            dfx_i = tf.slice(dloss_dw, begin=[i, 0], size=[1, 1])
            ddfx_i = tf.gradients(dfx_i, parameters)[
                0]  # whenever we use tf.gradients, make sure you get the actual tensors by putting [0] at the end
            hess.append(ddfx_i)
        hess = tf.squeeze(hess)
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            feed_dict = {x_input: np.random.random([batch_size, n_input]),
                         y_target: np.random.random([batch_size, n_output])}
            # print(sess.run(loss, feed_dict))
            # print(hess.get_shape())
            # print(sess.run(hess, feed_dict))
            return sess.run(hess, feed_dict)

if __name__ == '__main__':
    hess = getHessianMLP(n_input=3, n_hidden=4, n_output=3)

    eig, v = LA.eigh(hess)

    print(eig)
