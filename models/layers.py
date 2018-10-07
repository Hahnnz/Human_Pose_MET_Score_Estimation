import tensorflow as tf

def conv(data, ksize, filters, ssize, padding, use_bias, conv_name=None, bn_name=None, bn=False, act=True, is_train=None):
    if bn and is_train==None:
        raise ValueError('Phase should be declared "True" for train or "False" for test when you activate batch normalization.')
    
    output = tf.layers.conv2d(data, kernel_size=ksize, filters=filters,
                              strides=(ssize,ssize),
                              padding=padding.upper(),
                              name=conv_name,use_bias=use_bias,
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    if bn : output = batch_norm(output, is_train, bn_name)
    if act : output = tf.nn.relu(output)
    return output

def deconv(data, ksize, filters, ssize, padding, use_bias, deconv_name=None, bn_name=None, bn=False, act=True, is_train=None):
    if bn and is_train==None:
        raise ValueError('Phase should be declared "True" for train or "False" for test when you activate batch normalization.')
    
    output = tf.layers.conv2d_transpose(data, kernel_size=ksize, filters=filters,
                                        strides=(ssize,ssize),
                                        padding=padding.upper(),
                                        name=deconv_name, use_bias=use_bias,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    if bn : output = batch_norm(output, is_train, bn_name)
    if act : output = tf.nn.relu(output)
    return output

def max_pooling(data, ksize=3, ssize=2, name=None):
    return tf.nn.max_pool(data, ksize=[1,ksize,ksize,1], strides=[1,ssize,ssize,1], padding="SAME", name=name)

def dropout(data, ratio, name=None):
    return tf.nn.dropout(data, ratio, name=name)

def lrn(data, depth_radius, alpha, beta, name=None):
    return tf.nn.local_response_normalization(data, depth_radius=depth_radius, alpha=alpha, beta=beta, bias=1.0, name=name)

def batch_norm(data, is_train, name=None):
    return tf.layers.batch_normalization(data, training=is_train, name=name)

def fc(data, num_in, num_out, name=None, relu=True, bn=True, is_train=None):
    if bn and is_train==None:
        raise ValueError('Phase should be declared "True" for train or "False" for test when you activate batch normalization.')
        
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)
        output = tf.nn.xw_plus_b(data, weights, biases, name=scope.name)
    if bn : output = batch_norm(output, is_train)
    if relu : output = tf.nn.relu(output)
        
    return output