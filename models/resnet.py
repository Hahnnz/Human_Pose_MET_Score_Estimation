import tensorflow as tf
from models.layers import *

def identity_block(data, ksize, filters, stage, block, use_bias=True):
    suffix=str(stage)+block+"_branch"
    conv_name_base = "res"+suffix
    bn_name_base = "bn"+suffix
    
    filter1, filter2, filter3 = filters
    
    conv1 = conv(data, ksize, filter1, ssize=1, padding="SAME", conv_name=conv_name_base+"2a",
                  bn_name=bn_name_base+"2a", use_bias=use_bias, bn=True)
    conv2 = conv(conv1, ksize, filter2, ssize=1, padding="SAME",conv_name=conv_name_base+"2b",
                  bn_name=bn_name_base+"2b", use_bias=use_bias, bn=True)
    conv3 = conv(conv2, ksize, filter3, ssize=1, padding="SAME",conv_name=conv_name_base+"2c",
                  bn_name=bn_name_base+"2c", use_bias=use_bias, bn=True)
    if int(data.shape[-1])!=filter3:
        shortcut = conv(data, 1, filter3, ssize=1, padding="SAME",
                        conv_name=conv_name_base+"shortcut", use_bias=False, bn=False, act=False)
    else :
        shortcut = data
    addx_h = tf.add(conv3, shortcut)
    
    return tf.nn.relu(addx_h, name="res"+str(stage)+block+"_out")

def conv_block(data, kernel_size, filters, stage, block, ssize, use_bias=True):
    suffix=str(stage)+block+"_branch"
    conv_name_base = "res"+suffix
    bn_name_base = "bn"+suffix
    
    conv1 = conv(data, kernel_size, filters[0], ssize=ssize, padding="SAME",conv_name=conv_name_base+"2a",
                 bn_name=bn_name_base+"2a",use_bias=use_bias,bn=True,act=True)
    conv2 = conv(conv1, kernel_size, filters[1], ssize=1, padding="SAME",conv_name=conv_name_base+"2b",
                 bn_name=bn_name_base+"2b",use_bias=use_bias,bn=True,act=True)
    conv3 = conv(conv2, kernel_size, filters[2], ssize=1, padding="SAME",conv_name=conv_name_base+"2c",
                 bn_name=bn_name_base+"2c",use_bias=use_bias,bn=True,act=False)
    
    if int(data.shape[-1])!=filters[2]:
        shortcut = conv(data, 1, filters[2], ssize=1, padding="SAME",
                        conv_name=conv_name_base+"shortcut", use_bias=False, bn=False, act=False)
    else :
        shortcut = data
    addx_h = tf.add(conv3, shortcut)
    
    return tf.nn.relu(addx_h, name="res"+str(stage)+block+"_out")

class ResNet:
    def __init__(self,input_shape,output_shape, batch_size=None, gpu_memory_fraction=None):
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
        self.graph = tf.get_default_graph()

    def __create(self, stage5=False):
        # Stage 1
        #padded = ZeroPadding2D(self.x, psize=[3,3])
        
        conv1 = conv(self.x,filters=64,ksize=7,ssize=2,padding="SAME",use_bias=True,conv_name="conv1",
                     bn_name="bn_conv1",bn=True,act=True)
        pool1 = max_pooling(conv1, 3, 2)

        # Stage 2
        convblock_1 = conv_block(pool1,3,[64,64,256], stage=2, block="a", ssize=1)
        id_block_2 = identity_block(convblock_1, 3, [64,64,256], stage=2, block="b")
        id_block_3 = identity_block(id_block_2, 3, [64,64,256], stage=2, block="c")
        pool2 = max_pooling(id_block_3, 3, 2)
        
        # Stage 3
        convblock_4 = conv_block(pool2,3,[128,128,512], stage=3, block="a", ssize=1)
        id_block_5 = identity_block(convblock_4, 3, [128,128,512], stage=3, block="b")
        id_block_6 = identity_block(id_block_5, 3, [128,128,512], stage=3, block="c")
        id_block_7 = identity_block(id_block_6, 3, [128,128,512], stage=3, block="d")
        
        # Stage 4
        convblock_8 = conv_block(id_block_7,3,[256,256,1024], stage=4, block="a", ssize=1)
        id_block_9 = identity_block(convblock_8, 3, [256,256,1024], stage=4, block="b")
        id_block_10 = identity_block(id_block_9, 3, [256,256,1024], stage=4, block="c")
        id_block_11 = identity_block(id_block_10, 3, [256,256,1024], stage=4, block="d")
        id_block_12 = identity_block(id_block_11, 3, [256,256,1024], stage=4, block="e")
        id_block_13 = identity_block(id_block_12, 3, [256,256,1024], stage=4, block="f")
        
        # Stage 5
        convblock_14 = conv_block(id_block_13,3,[512,512,2048], stage=5, block="a", ssize=1)
        id_block_15 = identity_block(convblock_14, 3, [512,512,2048], stage=5, block="b")
        id_block_16 = identity_block(id_block_15, 3, [512,512,2048], stage=5, block="c")
        
        num_nodes=1
        for i in range(1,4): num_nodes*=int(id_block_16.get_shape()[i])
        self.rsz = tf.reshape(id_block_16, [-1, num_nodes])
