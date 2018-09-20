import tensorflow as tf
import numpy as np
from layers import *

class Unet:
    def __init__(self,input_shape,output_shape,num_classes,gpu_memory_fraction=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_classes = num_classes
        
        self.x = tf.placeholder(tf.float32, (None,) + self.input_shape, name='X')
        self.y = tf.placeholder(tf.int64, shape=(None,)+ self.output_shape, name='y')
        self.keep_prob = tf.placeholder(tf.float32)
        
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
            
        self.__create() 
        self.sess = tf.Session(config=config)
        self.graph = tf.get_default_graph()
    
    def __create(self):
        # Down Block 1
        self.conv1 = conv(self.x, ksize=3, filters=64, ssize=1, use_bias=True, padding='SAME',
                          conv_name='conv1', bn_name='bn1', bn=True)
        self.conv2 = conv(self.conv1, ksize=3, filters=64, ssize=1, use_bias=True, padding='SAME',
                          conv_name="conv2", bn_name='bn2', bn=True)
        self.pool1 = max_pooling(self.conv2, name="pool1")

        # Down Block 2
        self.conv3 = conv(self.pool1, ksize=3, filters=128, ssize=1, use_bias=True, padding='SAME',
                          conv_name="conv3", bn_name='bn3', bn=True)
        self.conv4 = conv(self.conv3, ksize=3, filters=128, ssize=1, use_bias=True, padding='SAME',
                          conv_name="conv4", bn_name='bn4', bn=True)
        self.pool2 = max_pooling(self.conv4, name="pool2")
        
        # Down Block 3
        self.conv5 = conv(self.pool2, ksize=3, filters=256, ssize=1, use_bias=True, padding='SAME',
                          conv_name="conv5", bn_name='bn5', bn=True)
        self.conv6 = conv(self.conv5, ksize=3, filters=256, ssize=1, use_bias=True, padding='SAME',
                          conv_name="conv6", bn_name='bn6', bn=True)
        self.pool3 = max_pooling(self.conv6, name="pool3")
        
        # Down Block 4
        self.conv7 = conv(self.pool3, ksize=3, filters=512, ssize=1, use_bias=True, padding='SAME',
                          conv_name="conv7", bn_name='bn7', bn=True)
        self.conv8 = conv(self.conv7, ksize=3, filters=512, ssize=1, use_bias=True, padding='SAME',
                          conv_name="conv8", bn_name='bn8', bn=True)
        self.drop4 = dropout(self.conv8, name='drop4', ratio=self.keep_prob)
        self.pool4 = max_pooling(self.drop4, name="pool4")

        # Down Block 5
        self.conv9 = conv(self.pool4, ksize=3, filters=1024, ssize=1, use_bias=True, padding='SAME',
                          conv_name="conv9", bn_name='bn9', bn=True)
        self.conv10 = conv(self.conv9, ksize=3, filters=1024, ssize=1, use_bias=True, padding='SAME',
                           conv_name="conv10", bn_name='bn10', bn=True)
        self.drop5 = dropout(self.conv10, name='drop5', ratio=self.keep_prob)
        
        # Up Block 4
        self.deconv1 = deconv(self.drop5, ksize=3, filters=512, ssize=2, use_bias=True, padding='SAME',
                              deconv_name="deconv1", bn_name='deconv_bn1', bn=True)
        self.concat1 = tf.concat((self.drop4,self.deconv1),axis=3, name='concat1')
        self.deconv2 = deconv(self.concat1, ksize=3, filters=512, ssize=1, use_bias=True, padding='SAME',
                              deconv_name="deconv2", bn_name='deconv_bn2', bn=True)
        self.deconv3 = deconv(self.deconv2, ksize=3, filters=512, ssize=1, use_bias=True, padding='SAME',
                              deconv_name="deconv3", bn_name='deconv_bn3', bn=True)

        # Up Block 3
        self.deconv4 = deconv(self.deconv3, ksize=3, filters=256, ssize=2, use_bias=True, padding='SAME',
                              deconv_name="deconv4", bn_name='deconv_bn4', bn=True)
        self.concat2 = tf.concat((self.conv6,self.deconv4),axis=3, name='concat2')
        self.deconv5 = deconv(self.concat2, ksize=3, filters=256, ssize=1, use_bias=True, padding='SAME',
                              deconv_name="deconv5", bn_name='deconv_bn5', bn=True)
        self.deconv6 = deconv(self.deconv5, ksize=3, filters=256, ssize=1, use_bias=True, padding='SAME',
                              deconv_name="deconv6", bn_name='deconv_bn6', bn=True)

        # Up Block 2
        self.deconv7 = deconv(self.deconv6, ksize=3, filters=128, ssize=2, use_bias=True, padding='SAME',
                              deconv_name="deconv7", bn_name='deconv_bn7', bn=True)
        self.concat3 = tf.concat((self.conv4,self.deconv7),axis=3, name='concat3')
        self.deconv8 = deconv(self.concat3, ksize=3, filters=128, ssize=1, use_bias=True, padding='SAME',
                              deconv_name="deconv8", bn_name='deconv_bn8', bn=True)
        self.deconv9 = deconv(self.deconv8, ksize=3, filters=128, ssize=1, use_bias=True, padding='SAME',
                              deconv_name="deconv9", bn_name='deconv_bn9', bn=True)

        # Up Block 1
        self.deconv10 = deconv(self.deconv9, ksize=3, filters=64, ssize=2, use_bias=True, padding='SAME',
                              deconv_name="deconv10", bn_name='deconv_bn10', bn=True)
        self.concat4 = tf.concat((self.conv2,self.deconv10),axis=3, name='concat9')
        self.deconv11 = deconv(self.concat4, ksize=3, filters=64, ssize=1, use_bias=True, padding='SAME',
                               deconv_name="deconv11", bn_name='deconv_bn11', bn=True)
        self.deconv12 = deconv(self.deconv11, ksize=3, filters=64, ssize=1, use_bias=True, padding='SAME',
                               deconv_name="deconv12", bn_name='deconv_bn12', bn=True)

        # Scoring
        self.score = conv(self.deconv12, ksize=1, filters=self.num_classes, ssize=1, use_bias=True, padding='SAME',
                          conv_name='score', bn=False, act=True)
