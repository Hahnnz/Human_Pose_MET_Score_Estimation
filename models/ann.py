import numpy as np
import tensorflow as tf
import pandas as pd

class ann:
    def one_hot_encoding(labels,depth):
        return np.eye(depth)[labels].reshape(len(labels),depth)
    
    def __init__(self, input_shape, gpu_memory_fraction=None):
        self.input_shape = input_shape
        #self.output_shape = output_shape
        #tf.default_graph()
        with tf.variable_scope('input'):
            self.X = tf.placeholder(tf.float32, self.input_shape)
            self.keep_prob = tf.placeholder(tf.float32)

        self.__create() 
            
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        
        self.sess = tf.Session(config=config)
        self.graph = tf.get_default_graph()

        
    def __create(self):
        tf.set_random_seed(777)
        training_epochs = 1000
        HN=200
        
        self.W1 = tf.get_variable("1", shape=[28,HN], initializer=tf.contrib.layers.xavier_initializer())
        self.b1 = tf.Variable(tf.random_normal([HN]))
        self.L1 = tf.matmul(self.X, self.W1) + self.b1
        self.bn1 = tf.contrib.layers.batch_norm(self.L1)
        self.L1=tf.nn.dropout(tf.nn.relu(self.bn1),keep_prob=self.keep_prob)

        self.W2 = tf.get_variable("2", shape=[HN,HN], initializer=tf.contrib.layers.xavier_initializer())
        self.b2 = tf.Variable(tf.random_normal([HN]))
        self.L2 = tf.matmul(self.L1, self.W2) + self.b2
        self.bn2 = tf.contrib.layers.batch_norm(self.L2)
        self.L2 =tf.nn.dropout(tf.nn.relu(self.bn2),keep_prob=self.keep_prob)
        
        self.W3 = tf.get_variable("3", shape=[HN,HN], initializer=tf.contrib.layers.xavier_initializer())
        self.b3 = tf.Variable(tf.random_normal([HN]))
        self.L3 = tf.matmul(self.L2, self.W3) + self.b3
        self.bn3 = tf.contrib.layers.batch_norm(self.L3)
        self.L3 =tf.nn.dropout(tf.nn.relu(self.bn3),keep_prob=self.keep_prob)
        
        self.W4 = tf.get_variable("4", shape=[HN,HN], initializer=tf.contrib.layers.xavier_initializer())
        self.b4 = tf.Variable(tf.random_normal([HN]))
        self.L4 = tf.matmul(self.L3, self.W4) + self.b4
        self.bn4 = tf.contrib.layers.batch_norm(self.L4)
        self.L4 =tf.nn.dropout(tf.nn.relu(self.bn4),keep_prob=self.keep_prob)
        
        self.W5 = tf.get_variable("5", shape=[HN,10], initializer=tf.contrib.layers.xavier_initializer())
        self.b5 = tf.Variable(tf.random_normal([10]))
        self.logit = tf.matmul(self.L4, self.W5) + self.b5
        self.hypothesis=tf.nn.softmax(self.logit)
        
    
'''
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=Y_one_hot))
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=1,rho=0.95,epsilon=1e-09).minimize(cost)

    prediction=tf.argmax(hypothesis,1)
    target=tf.argmax(Y,1)
    correct =tf.equal(prediction, target)
    accuracy=tf.reduce_mean(tf.cast(correct, tf.float32))

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())'''
