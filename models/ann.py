import numpy as np
import tensorflow as tf
from models.layers import *

class ann:
    def one_hot_encoding(labels,depth):
        return np.eye(depth)[labels].reshape(len(labels),depth)
    
    def __init__(self, input_shape, output_shape, batch_size=None, gpu_memory_fraction=None):
        self.input_shape = [batch_size,input_shape]
        self.output_shape = output_shape
        with tf.variable_scope('input', reuse=True):
            self.X = tf.placeholder(tf.float32, self.input_shape)
            self.Y = tf.placeholder(tf.int32, [None, 1])
            
            self.Y_one_hot=tf.one_hot(self.Y,depth=10)
            self.Y_ont_hot=tf.reshape(self.Y_one_hot,[-1,10])
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_train = tf.placeholder(tf.bool)

        self.__create() 
            
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config=config)
        self.graph = tf.get_default_graph()
        
    def __create(self):
        HN=200
        
        self.fc1 = fc(self.X, 28, 140, name='fc1', relu=True, bn=True, is_train=self.is_train)
        self.fc1 = tf.nn.dropout(self.fc1, self.keep_prob)
        
        self.fc2 = fc(self.fc1,140, 112, name='fc2', relu=True, bn=True, is_train=self.is_train)
        self.fc2 = tf.nn.dropout(self.fc2, self.keep_prob)
        
        self.fc3 = fc(self.fc2, 112, 84, name='fc3', relu=True, bn=True, is_train=self.is_train)
        self.fc3 = tf.nn.dropout(self.fc3, self.keep_prob)
        
        self.fc4 = fc(self.fc3, 84, 56, name='fc4', relu=True, bn=True, is_train=self.is_train)
        self.fc4 = tf.nn.dropout(self.fc4, self.keep_prob)
        
        self.fc5 = fc(self.fc4, 56,10, name='fc5', relu=True, bn=True, is_train=self.is_train)
        self.fc5 = tf.nn.dropout(self.fc5, self.keep_prob)
        
        self.logit = fc(self.fc5, 10, self.output_shape, name='logits', relu=False, bn=False)
        self.hypothesis=tf.nn.softmax(self.logit)