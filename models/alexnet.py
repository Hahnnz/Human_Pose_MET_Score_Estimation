import tensorflow as tf
import numpy as np

def _conv(data, ksize, filters, ssize, padding, name, bn=False):
    if not bn :
        outputs = tf.layers.conv2d(data, kernel_size=ksize, filters=filters, strides=(ssize,ssize), padding=padding, name=name, activation=tf.nn.relu)
    else : 
        with tf.variable_scope(name) as scope:
            conv = tf.layers.conv2d(data, kernel_size=ksize, filters=filters, strides=(ssize,ssize), padding=padding, name=scope.name)
            bn = tf.contrib.layes.batch_norm(conv)
            output = tf.nn.relu(bn)
    return output

def _max_pooling(data, name):
    return tf.nn.max_pool(data, ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID", name=name)

def _dropout(data, name):
    return tf.nn.dropout(x, 0.5, name=name)

def _lrn(data, depth_radius, alpha, beta, name):
    return tf.nn.local_response_normalization(data, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=1.0, name=name)

def _fc(data, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable("weights", shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable("biases", [num_output], trainable=True)
        output = tf.nn.xw_plus_b(data, weights, biases, name=scope.name)
    if relu : return tf.nn.relu(output)
    else : return act
    
class config_net:
    def AlexNet(data):
        conv1 = _conv(data, ksize=11, filters=96, ssize=4, padding='VALID', name='conv1')
        lrn1 = _lrn(conv1, 2, 2e-5, 0.75, name="lrn1")
        pool1 = _max_pooling(lrn1, "pool1")

        conv2 = _conv(pool1, ksize=5, filters=256, ssize=1, padding="VALID", name="conv2")
        lrn2 = _lrn(conv2, 2, 2e-5, 0.75, name= "lrn2")
        pool2 = _max_pooling(lrn2, "pool2")

        conv3 = _conv(pool2, ksize=3, filters=384, ssize=1, padding="VALID", name="conv3")
        conv4 = _conv(conv3, ksize=3, filters=384, ssize=1, padding="VALID", name="conv4")
        conv5 = _conv(conv4, ksize=3, filters=384, ssize=1, padding="VALID", name="conv5")
        pool3 = _max_pooling(conv4, "pool3")

        num_nodes=0
        for i in range(1,4): num_nodes*=int(pool3.get_shape()[i])
        rsz = tf.reshape(pool3, [-1, num_nodes])

        fc6 = _fc(rsz,num_nodes,4096,name="fc6")
        fc7 = _fc(fc6,4096,4096,name="fc7")
        fc8 = _fc(fc7,4096,Classes.shape[0],name="fc8")

    def ConvNet(X):
        conv1 = _conv(X, ksize=11, filters=96, ssize=4, padding='VALID', name='conv1', bn=True)
        pool1 = _max_pooling(conv1, "pool1")

        conv2 = _conv(pool1, ksize=5, filters=256, ssize=1, padding="VALID", name="conv2", bn=True)
        pool2 = _max_pooling(conv2, "pool2")

        conv3 = _conv(pool2, ksize=3, filters=384, ssize=1, padding="VALID", name="conv3", bn=True)
        conv4 = _conv(conv3, ksize=3, filters=384, ssize=1, padding="VALID", name="conv4", bn=True)
        conv5 = _conv(conv4, ksize=3, filters=384, ssize=1, padding="VALID", name="conv5", bn=True)
        pool3 = _max_pooling(conv4, "pool3")

        num_nodes=0
        for i in range(1,4): num_nodes*=int(pool3.get_shape()[i])
        rsz = tf.reshape(pool3, [-1, num_nodes])

        fc6 = _fc(rsz,num_nodes,4096,name="fc6")
        fc7 = _fc(fc6,4096,4096,name="fc7")
        fc8 = _fc(fc7,4096,Classes.shape[0],name="fc8")