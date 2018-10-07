from models.layers import *
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np

class Regressionnet:
    def __init__(self, data_shape, num_joints, batch_size=None, gpu_memory_fraction=None, optimizer_type='adam', phase='train'):
        if phase not in ['train', 'inference'] : raise  ValueError("phase must be 'train' or 'inference'.")
        self.graph = tf.Graph()
        self.num_joints = num_joints
        
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        
        self.sess = tf.Session(config=config, graph=self.graph)
        
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, (batch_size,) + data_shape, name='input_images')
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_train = tf.placeholder(tf.bool)
            self.__create_model() 
            
            if phase=='train':
                self.y = tf.placeholder(tf.float32, [batch_size, num_joints*2], name="joints_ground_truth")
                self.valid = tf.placeholder(tf.float32, [batch_size, num_joints*2], name="joints_is_valid")
                self.lr = tf.placeholder(tf.float32, name="lr")

                diff = tf.subtract(self.y, self.fc_regression)
                diff_valid = tf.multiply(diff, self.valid)

                num_valid_joints = tf.reduce_sum(self.valid, axis=1) / tf.constant(2.0, dtype=tf.float32)

                self.pose_loss_op = tf.reduce_mean(tf.reduce_sum(tf.square(diff_valid), axis=1) / num_valid_joints, name="joint_euclidean_loss")

                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
                self.loss_with_decay_op = self.pose_loss_op + tf.constant(0.0005, name="weight_decay") * l2_loss

                tf.summary.scalar("loss_with_decay", self.loss_with_decay_op)
                tf.summary.scalar("loss", self.pose_loss_op)
                tf.summary.scalar("learning_rate", self.lr)

                self.sess.run(tf.global_variables_initializer())
                self.train_op = self.__set_op(self.pose_loss_op, self.lr, optimizer_type)

                uninit_vars = [v for v in tf.global_variables()
                              if not tf.is_variable_initialized(v).eval(session=self.sess)]
                self.sess.run(tf.variables_initializer(uninit_vars))

    def __create_model(self):
        padding_type='SAME'
        
        self.conv1 = conv(self.x, ksize=3, filters=16, ssize=1, use_bias=True, padding=padding_type,
                          conv_name='conv1', bn_name='bn1', bn=True, is_train=self.is_train)
        self.conv2 = conv(self.conv1, ksize=3, filters=16, ssize=1, use_bias=True, padding=padding_type,
                          conv_name="conv2", bn_name='bn2', bn=True, is_train=self.is_train)
        self.pool1 = max_pooling(self.conv2, name="pool1")

        self.conv3 = conv(self.pool1, ksize=3, filters=32, ssize=1, use_bias=True, padding=padding_type,
                          conv_name='conv3', bn_name='bn3', bn=True, is_train=self.is_train)
        self.conv4 = conv(self.conv3, ksize=3, filters=32, ssize=1, use_bias=True, padding=padding_type,
                          conv_name="conv4", bn_name='bn4', bn=True, is_train=self.is_train)
        self.pool2 = max_pooling(self.conv4, name="pool2")
        
        self.conv5 = conv(self.pool2, ksize=3, filters=64, ssize=1, use_bias=True, padding=padding_type,
                          conv_name='conv5', bn_name='bn5', bn=True, is_train=self.is_train)
        self.conv6 = conv(self.conv5, ksize=3, filters=64, ssize=1, use_bias=True, padding=padding_type,
                          conv_name="conv6", bn_name='bn6', bn=True, is_train=self.is_train)
        self.pool3 = max_pooling(self.conv6, name="pool3")

        num_nodes=1
        for i in range(1,4): num_nodes*=int(self.pool3.get_shape()[i])
        self.rsz = tf.reshape(self.pool3, [-1, num_nodes])
        
        self.fc6 = fc(self.rsz,num_nodes,4096,name="fc6", bn=False)
        self.drop6 = dropout(self.fc6, name="drop6", ratio=self.keep_prob)
        self.fc7 = fc(self.drop6,4096,4096,name="fc7", bn=False)
        self.drop7 = dropout(self.fc7, name="drop7", ratio=self.keep_prob)
        self.fc_regression = fc(self.drop7,4096,self.num_joints*2,name="fc_regression", relu=False, bn=False)

    def __set_op(self, loss_op, learning_rate, optimizer_type="adam"):
        with self.graph.as_default():
            if optimizer_type=="adam":
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif optimizer_type == "adagrad":
                optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.0001)
            elif optimizer_type == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            elif optimizer_type == "momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
            elif optimizer_type == "adadelta":
                optimizer = tf.train.AdadeltaOptimizer(learning_rate,rho=0.95,epsilon=1e-09)
            else : raise ValueError("{} optimizer doesn't exist.".format(optimizer_type))

            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            grads = tf.gradients(loss_op, trainable_vars)

            with tf.name_scope("grad_norms"):
                for v, grad in zip(trainable_vars, grads):
                    if grad is not None :
                        grad_norm_op = tf.nn.l2_loss(grad, name=format(v.name[:-2]))
                        tf.add_to_collection("grads", grad_norm_op)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(zip(grads, trainable_vars), name="train_op")

        return train_op