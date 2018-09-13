import tensorflow as tf
import numpy as np
from models.layers import *

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
        self.deconv6 = deconv(self.drop5, ksize=3, filters=512, ssize=2, use_bias=True, padding='SAME',
                              deconv_name="up6", bn_name='up6', bn=True)
        self.concat6 = tf.concat((self.drop4,self.deconv6),axis=3, name='concat6')
        self.conv11 = conv(self.concat6, ksize=3, filters=512, ssize=1, use_bias=True, padding='SAME',
                          conv_name="conv11", bn_name='bn11', bn=True)
        self.conv12 = conv(self.conv11, ksize=3, filters=512, ssize=1, use_bias=True, padding='SAME',
                           conv_name="conv12", bn_name='bn12', bn=True)

        # Up Block 3
        self.deconv7 = deconv(self.conv12, ksize=3, filters=256, ssize=2, use_bias=True, padding='SAME',
                              deconv_name="up7", bn_name='up7', bn=True)
        self.concat7 = tf.concat((self.conv6,self.deconv7),axis=3, name='concat7')
        self.conv13 = conv(self.concat7, ksize=3, filters=256, ssize=1, use_bias=True, padding='SAME',
                          conv_name="conv13", bn_name='bn13', bn=True)
        self.conv14 = conv(self.conv13, ksize=3, filters=256, ssize=1, use_bias=True, padding='SAME',
                           conv_name="conv14", bn_name='bn14', bn=True)

        # Up Block 2
        self.deconv8 = deconv(self.conv14, ksize=3, filters=128, ssize=2, use_bias=True, padding='SAME',
                              deconv_name="up8", bn_name='up8', bn=True)
        self.concat8 = tf.concat((self.conv4,self.deconv8),axis=3, name='concat8')
        self.conv15 = conv(self.concat8, ksize=3, filters=128, ssize=1, use_bias=True, padding='SAME',
                          conv_name="conv15", bn_name='bn15', bn=True)
        self.conv16 = conv(self.conv15, ksize=3, filters=128, ssize=1, use_bias=True, padding='SAME',
                           conv_name="conv16", bn_name='bn16', bn=True)

        # Up Block 1
        self.deconv9 = deconv(self.conv16, ksize=3, filters=64, ssize=2, use_bias=True, padding='SAME',
                              deconv_name="up9", bn_name='up9', bn=True)
        self.concat9 = tf.concat((self.conv2,self.deconv9),axis=3, name='concat9')
        self.conv17 = conv(self.concat9, ksize=3, filters=64, ssize=1, use_bias=True, padding='SAME',
                          conv_name="conv17", bn_name='bn17', bn=True)
        self.conv18 = conv(self.conv17, ksize=3, filters=64, ssize=1, use_bias=True, padding='SAME',
                           conv_name="conv18", bn_name='bn18', bn=True)

        # Scoring
        self.score = conv(self.conv18, ksize=1, filters=self.num_classes, ssize=1, use_bias=True, padding='SAME',
                          conv_name='score', bn=False, act=True)
