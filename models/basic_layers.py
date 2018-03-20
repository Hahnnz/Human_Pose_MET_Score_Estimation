import tensorflow as tf

def conv(data, ksize, filters, ssize, padding, use_bias, conv_name=None, bn_name=None, bn=False, act=True):
    if not bn :
        if act : output = tf.layers.conv2d(data, kernel_size=ksize, filters=filters, strides=(ssize,ssize), padding=padding, name=conv_name, activation=tf.nn.relu,use_bias=use_bias)
        else : output = tf.layers.conv2d(data, kernel_size=ksize, filters=filters, strides=(ssize,ssize), padding=padding, name=conv_name,use_bias=use_bias)
    else : 
        conv = tf.layers.conv2d(data, kernel_size=ksize, filters=filters, strides=(ssize,ssize), padding=padding, name=conv_name,use_bias=use_bias)
        output = tf.contrib.layers.batch_norm(conv, axis=3, name=bn_name)
        if act : output = tf.nn.relu(output)
    return output

def max_pooling(data, name=None):
    return tf.nn.max_pool(data, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME", name=name)

def dropout(data, name=None):
    return tf.nn.dropout(data, 0.5, name=name)

def lrn(data, depth_radius, alpha, beta, name):
    return tf.nn.local_response_normalization(data, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=1.0, name=name)

def bn(data, name=None):
    return tf.contrib.layers.batch_norm(data, axis=3, name=name)

def fc(data, num_in, num_out, name=None, relu=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)
        output = tf.nn.xw_plus_b(data, weights, biases, name=scope.name)
    if relu : return tf.nn.relu(output)
    else: return output