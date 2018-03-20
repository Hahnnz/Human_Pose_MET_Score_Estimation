import tensorflow as tf
from basic_layers import *

def AlexNet(data, labels):
    conv1 = conv(data, ksize=11, filters=96, ssize=4, use_bias=True, padding='VALID', conv_name='conv1')
    lrn1 = lrn(conv1, 2, 2e-5, 0.75, name="lrn1")
    pool1 = max_pooling(lrn1, "pool1")

    conv2 = conv(pool1, ksize=5, filters=256, ssize=1, use_bias=True, padding="VALID", conv_name="conv2")
    lrn2 = lrn(conv2, 2, 2e-5, 0.75, name= "lrn2")
    pool2 = max_pooling(lrn2, "pool2")
    
    conv3 = conv(pool2, ksize=3, filters=384, ssize=1, use_bias=True, padding="VALID", conv_name="conv3")
    conv4 = conv(conv3, ksize=3, filters=384, ssize=1, use_bias=True, padding="VALID", conv_name="conv4")
    conv5 = conv(conv4, ksize=3, filters=384, ssize=1, use_bias=True, padding="VALID", conv_name="conv5")
    pool3 = max_pooling(conv4, "pool3")
    
    num_nodes=0
    for i in range(1,4): num_nodes*=int(pool3.get_shape()[i])
    rsz = tf.reshape(pool3, [-1, num_nodes])
    
    fc6 = fc(rsz,num_nodes,4096,name="fc6")
    drop6 = dropout(fc6, name="drop6")
    fc7 = fc(drop6,4096,4096,name="fc7")
    drop7 = dropout(fc7, name="drop7")
    fc8 = fc(drop7,4096,int(labels.shape[1]),name="fc8")