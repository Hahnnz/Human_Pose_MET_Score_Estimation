import tensorflow as tf
from basic_layers import *

def identity_block(data, ksize, filters, stage, block, use_bias=True):
    suffix=str(stage)+block+"_branch"
    conv_name_base = "res"+suffix
    bn_name_base = "bn"+suffix
    
    filter1, filter2, filter3 = filters
    
    conv1 = conv(data,1,filter1,ssize=1,padding="SAME",conv_name=conv_name_base+"2a",
                  bn_name=bn_name_base+"2a",use_bias=use_bias,bn=True)
    conv2 = conv(conv1,ksize,filter2,ssize=1,padding="SAME",conv_name=conv_name_base+"2b",
                  bn_name=bn_name_base+"2b",use_bias=use_bias,bn=True)
    conv3 = conv(conv2,1,filter3,ssize=1,padding="SAME",conv_name=conv_name_base+"2c",
                  bn_name=bn_name_base+"2c",use_bias=use_bias,bn=True,act=False)
    addx_h = tf.add(conv3, data)
    
    return tf.nn.relu(addx_h, name="res"+str(stage)+block+"_out")

def conv_block(data, ksize, filters, stage, block, ssize=2, use_bias=True):
    suffix=str(stage)+block+"_branch"
    conv_name_base = "res"+suffix
    bn_name_base = "bn"+suffix
    
    filter1, filter2, filter3 = filters
    
    conv1 = conv(data,filter1, 1, ssize=ssize, padding="SAME",conv_name=conv_name_base+"2a",
                 bn_name=bn_name_base+"2a",use_bias=use_bias,bn=True,act=True)
    conv2 = conv(conv1,filter2, ksize=ksize, ssize=1, padding="SAME",conv_name=conv_name_base+"2b",
                 bn_name=bn_name_base+"2b",use_bias=use_bias,bn=True,act=True)
    conv3 = conv(conv2,filter3, ksize=ksize, ssize=1, padding="SAME",conv_name=conv_name_base+"2c",
                 bn_name=bn_name_base+"2c",use_bias=use_bias,bn=True,act=False)
    shortcut = conv(conv3,filter3, 1, ssize=ssize, padding="SAME", conv_name=conv_name_base+"1",use_bias=True,
                     bn_name=bn_name_base+"1",bn=True, act=False)
    addx_h = tf.add(shortcut, data)
    
    return tf.nn.relu(addx_h, name="res"+str(stage)+block+"_out")

def create_graph(dataset, stage5=False):
    # Stage 1
    padded = ZeroPadding2D(dataset,psize=[3,3])
    
    conv1 = conv(padded,filters=64,ksize=7,ssize=2,padding="SAME",use_bias=True,conv_name="conv1",
                 bn_name="bn_conv1",bn=True,act=True)
    S1 = pool1 = max_pooling(conv1)
    
    # Stage 2
    convblock_1 = conv_block(pool1,3,[64,64,256], stage=2, block="a", ssize=1)
    id_block_2 = identity_block(convblock1, 3, [64,64,256], stage=2, block="b")
    S2 = id_block_3 = identity_block(id_block_2, 3, [64,64,256], stage=2, block="c")

    # Stage 3
    convblock_4 = conv_block(id_block_3,3,[128,128,512], stage=3, block="a")
    id_block_5 = identity_block(convblock_4, 3, [128,128,512], stage=3, block="b")
    id_block_6 = identity_block(id_block_5, 3, [128,128,512], stage=3, block="c")
    S3 = id_block_7 = identity_block(id_block_6, 3, [128,128,512], stage=3, block="d")
    
    # Stage 4
    convblock_8 = convblock(id_block_7, 3, [256,256,1024], stage=4, block="a")
    if not stage5:
        for i in range(5): # Block count : 5 for resnet50
            id_block_9 = identity_block(convblock_8, 3, [256,256,1024], stage=4, block="a")
        S4 = id_block_9
        S5 = None
    elif stage5 :
        for i in range(22): # Block count : 22 for resnet101
            loop_block = identity_block(loop_block, 3, [256,256,1024], stage=4, block=chr(98+i))
        S4 = id_block_9 = loop_block
        
        # Stage 5
        conv_block_10 = conv_block(id_block_9, 3, [512,512,2048], stage=5, block="a")
        id_block_11 = identity_block(conv_block_10, 3, [512,512,2048], stage=5, block="b")
        S5 = id_block12 = identity_block(id_block_11, 3, [512,512,2048], stage=5, block="c")
    
    return [S1, S2, S3, S4, S5]