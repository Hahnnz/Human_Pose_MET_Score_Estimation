import tensorflow as tf
import numpy as np
from models.layers import *

class alexnet:
    def __init__(self,input_shape,output_shape,batch_size=None, gpu_memory_fraction=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        with tf.variable_scope('input'):
            self.x = tf.placeholder(tf.float32, (batch_size,) + self.input_shape, name='X')
            self.y_gt = tf.placeholder(tf.int32, shape=(batch_size,)+ self.output_shape, name='y_gt')
            self.keep_prob = tf.placeholder(tf.float32)

        self.__create() 
        self.global_iter_counter = tf.Variable(0, name='global_iter_counter', trainable=False)
        
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
            
        self.sess = tf.Session(config=config)
        
        self.sess = tf.Session(config=config)
        self.graph = tf.get_default_graph()
    
    def __create(self):
        self.conv1 = conv(self.x, ksize=11, filters=96, ssize=4, use_bias=True, padding='VALID', conv_name='conv1')
        self.lrn1 = lrn(self.conv1, 2, 2e-5, 0.75, name="lrn1")
        self.pool1 = max_pooling(self.lrn1, 3, 2, "pool1")

        self.conv2 = conv(self.pool1, ksize=5, filters=256, ssize=1, use_bias=True, padding="VALID", conv_name="conv2")
        self.lrn2 = lrn(self.conv2, 2, 2e-5, 0.75, name= "lrn2")
        self.pool2 = max_pooling(self.lrn2, 3, 2, "pool2")

        self.conv3 = conv(self.pool2, ksize=3, filters=384, ssize=1, use_bias=True, padding="VALID", conv_name="conv3")
        self.conv4 = conv(self.conv3, ksize=3, filters=384, ssize=1, use_bias=True, padding="VALID", conv_name="conv4")
        self.conv5 = conv(self.conv4, ksize=3, filters=384, ssize=1, use_bias=True, padding="VALID", conv_name="conv5")
        self.pool3 = max_pooling(self.conv4, 3, 2, "pool3")

        num_nodes=1
        for i in range(1,4): num_nodes*=int(self.pool3.get_shape()[i])
        self.rsz = tf.reshape(self.pool3, [-1, num_nodes])

        self.fc6 = fc(self.rsz,num_nodes,4096,name="fc6")
        self.drop6 = dropout(self.fc6, name="drop6")
        self.fc7 = fc(self.drop6,4096,4096,name="fc7")
        self.drop7 = dropout(self.fc7, name="drop7")
        self.fc8 = fc(self.drop7,4096,int(self.output_shape[0]),name="fc8")
        
        self.layers=([self.conv1, self.lrn1, self.pool1, self.conv2, self.lrn2, self.pool2, self.conv3, self.conv4, self.conv5, self.pool3, self.fc6, self.drop6, self.fc7, self.drop7, self.fc8])
    def get_layers(self, layer_name):
        found=False

        for nb in range(len(self.layers)):
            idx=list(i for i in range(len(self.layers[nb].name)) if self.layers[nb].name[i]=='/' or self.layers[nb].name[i]==':')[0]
            if self.layers[nb].name[:idx] == layer_name : 
                found=True
                break
            else : continue

        if found : return self.layers[nb]
        elif not found : print("tensor named "+layer_name+" doesn't exist")
    def print_shape(self):
        for layers in self.layers:
            print(layers.name,list(int(layers.get_shape()[i]) for i in range(len(layers.get_shape()))))
