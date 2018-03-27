from models import alexnet
from models.basic_layers import *
import tensorflow as tf
import numpy as np

Classes=np.array(["Right ankle", "Right knee", "Right hip", "Left hip", "Left knee", "Left ankle", "Right wrist", "Right elbow", "Right shoulder", "Left shoulder", "Left elbow", "Left wrist", "Neck", "Head top"])

with tf.Graph().as_default():
    net = alexnet.alexnet(batch_size=100,input_shape=(400,400,3),output_shape=(100,28))
    drop7 = net.get_layers("drop7")
    net.fc_regression = fc(drop7, int(drop7.get_shape()[1]), Classes.shape[0]*2, name="fc_regression", relu=False)
    
    with tf.name_scope("PoseInput"):
        joints_gt = tf.placeholder(tf.float32, [None, Classes.shape[0], 2], name="joints_ground_truth")
        joints_is_valid = tf.placeholder(tf.int32, [None, Classes.shape[0], 2], name="joints_is_valid")
    
    joints_gt_flatted = tf.reshape(joints_gt, [-1, Classes.shape[0]*2])
    joints_is_valid_flatted = tf.cast(tf.reshape(joints_is_valid, shape=[-1, Classes.shape[0]*2]), tf.float32)
        
    diff = tf.subtract(joints_gt_flatted, net.fc_regression)
    diff_valid = tf.multiply(diff, joints_is_valid_flatted)
    
    num_valid_joints = tf.reduce_sum(joints_is_valid_flatted, axis=1) / tf.constant(2.0, dtype=tf.float32)
    
    pose_loss_op = tf.reduce_mean(tf.reduce_sum(tf.pow(diff_valid, 2), axis=1) / num_valid_joints, name="joint_euclidean_loss")
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    loss_with_decay_op = pose_loss_op + tf.constant(0.0005, name="weight_decay") * l2_loss
    
    with tf.variable_scope("lr"):
        conv_lr_pl = tf.placeholder(tf.float32, tuple(), name="conv_lr")
        fc_lr_pl = tf.placeholder(tf.float32, tuple(), name="fc_lr")
    
    #tf.summary.scalar('loss_with_decay', loss_with_decay_op)
    #tf.summary.scalar('loss', pose_loss_op)
    #tf.summary.scalar('conv_lr', conv_lr_pl)
    #tf.summary.scalar('fc_lr', fc_lr_pl)
    
    net.sess.run(tf.global_variables_initializer())
