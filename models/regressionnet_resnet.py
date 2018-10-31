from models.layers import *
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np

def identity_block(data, ksize, filters, stage, block, use_bias=True, is_train=None):
    suffix=str(stage)+block+"_branch"
    conv_name_base = "res"+suffix
    bn_name_base = "bn"+suffix
    
    filter1, filter2, filter3 = filters
    
    conv1 = conv(data, ksize, filter1, ssize=1, padding="SAME", conv_name=conv_name_base+"2a",
                  bn_name=bn_name_base+"2a", use_bias=use_bias, bn=False, act=False, is_train=is_train)
    conv2 = conv(conv1, ksize, filter2, ssize=1, padding="SAME",conv_name=conv_name_base+"2b",
                  bn_name=bn_name_base+"2b", use_bias=use_bias, bn=False, act=False, is_train=is_train)
    conv3 = conv(conv2, ksize, filter3, ssize=1, padding="SAME",conv_name=conv_name_base+"2c",
                  bn_name=bn_name_base+"2c", use_bias=use_bias, bn=False, act=False, is_train=is_train)
    if int(data.shape[-1])!=filter3:
        shortcut = conv(data, 1, filter3, ssize=1, padding="SAME",
                        conv_name=conv_name_base+"shortcut", use_bias=False, bn=False, act=False)
    else :
        shortcut = data
    #addx_h = batch_norm(tf.add(conv3, shortcut), is_train=is_train)
    addx_h = tf.add(conv3, shortcut)
    
    return tf.nn.relu(addx_h, name="res"+str(stage)+block+"_out")

def conv_block(data, kernel_size, filters, stage, block, ssize, use_bias=True, is_train=None):
    suffix=str(stage)+block+"_branch"
    conv_name_base = "res"+suffix
    bn_name_base = "bn"+suffix
    
    conv1 = conv(data, kernel_size, filters[0], ssize=ssize, padding="SAME",conv_name=conv_name_base+"2a",
                 bn_name=bn_name_base+"2a",use_bias=use_bias,bn=False,act=False, is_train=is_train)
    conv2 = conv(conv1, kernel_size, filters[1], ssize=1, padding="SAME",conv_name=conv_name_base+"2b",
                 bn_name=bn_name_base+"2b",use_bias=use_bias,bn=False,act=False, is_train=is_train)
    conv3 = conv(conv2, kernel_size, filters[2], ssize=1, padding="SAME",conv_name=conv_name_base+"2c",
                 bn_name=bn_name_base+"2c",use_bias=use_bias,bn=False,act=False, is_train=is_train)
    
    if int(data.shape[-1])!=filters[2]:
        shortcut = conv(data, 1, filters[2], ssize=1, padding="SAME",
                        conv_name=conv_name_base+"shortcut", use_bias=False, bn=False, act=False)
    else :
        shortcut = data
    #addx_h = batch_norm(tf.add(conv3, shortcut), is_train=is_train)
    addx_h = tf.add(conv3, shortcut)
    
    return tf.nn.relu(addx_h, name="res"+str(stage)+block+"_out")

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
        BIAS = False
        
        conv1 = conv(self.x,filters=64,ksize=9,ssize=2,padding="SAME",use_bias=BIAS,conv_name="conv1",
                     bn_name="bn_conv1",bn=True,act=True, is_train=self.is_train)

        # Stage 2
        stage2_filters = [64,64,256]
        convblock_1 = conv_block(conv1,7,stage2_filters, stage=2, block="a", ssize=1, is_train=self.is_train, use_bias=BIAS)
        id_block_2 = identity_block(convblock_1, 7, stage2_filters, stage=2, block="b", is_train=self.is_train, use_bias=BIAS)
        id_block_3 = identity_block(id_block_2, 7, stage2_filters, stage=2, block="c", is_train=self.is_train, use_bias=BIAS)
        conv2 = conv(id_block_3,filters=stage2_filters[-1],ksize=7,ssize=2,padding="SAME",use_bias=BIAS,conv_name="conv2",
                     bn_name="bn_conv2",bn=True,act=True, is_train=self.is_train)
        
        # Stage 3
        stage3_filters = [128,128,512]
        convblock_4 = conv_block(conv2,5,stage3_filters, stage=3, block="a", ssize=1, is_train=self.is_train, use_bias=BIAS)
        id_block_5 = identity_block(convblock_4, 5, stage3_filters, stage=3, block="b", is_train=self.is_train, use_bias=BIAS)
        id_block_6 = identity_block(id_block_5, 5, stage3_filters, stage=3, block="c", is_train=self.is_train, use_bias=BIAS)
        id_block_7 = identity_block(id_block_6, 5, stage3_filters, stage=3, block="d", is_train=self.is_train, use_bias=BIAS)
        conv3 = conv(id_block_7,filters=stage3_filters[-1], ksize=5,ssize=2,padding="SAME",use_bias=False,conv_name="conv3",
                     bn_name="bn_conv3",bn=True,act=True, is_train=self.is_train)
        
        # Stage 4
        stage4_filters = [256,256,1024]
        stage4 = conv_block(conv3,3,stage4_filters, stage=4, block="a", ssize=1, is_train=self.is_train, use_bias=BIAS)
        for i in range(5):
            stage4 = identity_block(stage4, 3, stage4_filters, stage=4, block=chr(ord('b')+i),
                                    is_train=self.is_train, use_bias=BIAS)
        conv4 = conv(stage4,filters=stage4_filters[-1],ksize=3,ssize=2,padding="SAME",use_bias=False,conv_name="conv4",
                     bn_name="bn_conv4",bn=True,act=True, is_train=self.is_train)
        # Stage 5
        stage5_filters = [512,512,2048]
        convblock_14 = conv_block(stage4,3,stage5_filters, stage=5, block="a", ssize=1, is_train=self.is_train, use_bias=BIAS)
        id_block_15 = identity_block(convblock_14, 3, stage5_filters, stage=5, block="b", is_train=self.is_train, use_bias=BIAS)
        id_block_16 = identity_block(id_block_15, 3, stage5_filters, stage=5, block="c", is_train=self.is_train, use_bias=BIAS)
        
        conv1x1 = conv(id_block_16,filters=stage5_filters[-1],ksize=8,ssize=8,padding="SAME", use_bias=False,
                       conv_name="conv1x1", bn_name="bn_conv1x1",bn=True,act=True, is_train=self.is_train)
        
        num_nodes=1
        for i in range(1,4): num_nodes*=int(conv1x1.get_shape()[i])
        self.rsz = tf.reshape(conv1x1, [-1, num_nodes])
        self.fc_regression = fc(self.rsz,num_nodes,self.num_joints*2,name="fc_regression", relu=False, bn=False)

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
