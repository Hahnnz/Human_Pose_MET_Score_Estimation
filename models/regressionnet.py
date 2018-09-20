from models import alexnet, Convnet1, resnet
from models.layers import *
from scripts import tools
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np
import copy, math
from tqdm import tqdm

def create_regression_net(data_shape,
                          num_joints,
                          batch_size = None,
                          gpu_memory_fraction=None,
                          optimizer_type="adam",
                          net_type="alexnet"):
    
    # Create Regressionnet. - Alexnet, Resnet, Convnet1
    with tf.Graph().as_default():

        if net_type == "alexnet":
            net = alexnet.alexnet(batch_size=batch_size,
                                  input_shape=(data_shape),output_shape=(num_joints*2,),
                                 gpu_memory_fraction=gpu_memory_fraction)
            drop7 = net.get_layers("drop7")
            net.fc_regression = fc(drop7, int(drop7.get_shape()[1]),
                                   num_joints*2, name="fc_regression", relu=False)
        elif net_type == "convnet1":
            net = Convnet1.convNet(batch_size=batch_size,
                                  input_shape=(data_shape),output_shape=(num_joints*2,),
                                 gpu_memory_fraction=gpu_memory_fraction)
            #drop7 = net.get_layers("drop7")
            net.fc_regression = fc(net.drop7, int(net.drop7.get_shape()[1]),
                                   num_joints*2, name="fc_regression", relu=False)
        elif net_type == "resnet": 
            net = resnet.ResNet(batch_size=batch_size,
                                input_shape=(data_shape),output_shape=(num_joints*2,),
                                gpu_memory_fraction=gpu_memory_fraction)
            net.fc_regression = fc(net.rsz, int(net.rsz.get_shape()[1]),
                                   num_joints*2, name="fc_regression", relu=False)
        else : 
            raise ValueError("net type should be 'alexnet'. resnet will be updated soon")
            
        with tf.name_scope("PoseInput"):
            joints_gt = tf.placeholder(tf.float32, [batch_size, num_joints*2], name="joints_ground_truth")
            joints_is_valid = tf.placeholder(tf.float32, [batch_size, num_joints*2], name="joints_is_valid")
        
        diff = tf.subtract(joints_gt, net.fc_regression)
        diff_valid = tf.multiply(diff, joints_is_valid)

        num_valid_joints = tf.reduce_sum(joints_is_valid, axis=1) / tf.constant(2.0, dtype=tf.float32)

        pose_loss_op = tf.reduce_mean(tf.reduce_sum(
            tf.square(diff_valid), axis=1) / num_valid_joints, name="joint_euclidean_loss")
        
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        loss_with_decay_op = pose_loss_op + tf.constant(0.0005, name="weight_decay") * l2_loss

        with tf.variable_scope("lr"):
            conv_lr_pl = tf.placeholder(tf.float32, tuple(), name="conv_lr")
            fc_lr_pl = tf.placeholder(tf.float32, tuple(), name="fc_lr")

        tf.summary.scalar("loss_with_decay", loss_with_decay_op)
        tf.summary.scalar("loss", pose_loss_op)
        tf.summary.scalar("conv_lr", conv_lr_pl)
        tf.summary.scalar("fc_lr", fc_lr_pl)
            
        net.sess.run(tf.global_variables_initializer())
        train_op = set_op(net, pose_loss_op, fc_lr=fc_lr_pl, conv_lr=conv_lr_pl, optimizer_type=optimizer_type)

        uninit_vars = [v for v in tf.global_variables()
                      if not tf.is_variable_initialized(v).eval(session=net.sess)]
        net.sess.run(tf.variables_initializer(uninit_vars))
    
    return net, loss_with_decay_op, pose_loss_op, train_op

def set_op(net, loss_op, fc_lr, conv_lr, optimizer_type="adam"):
    with net.graph.as_default():
        if optimizer_type=="adam":
            conv_optimizer = tf.train.AdamOptimizer(conv_lr)
            fc_optimizer = tf.train.AdamOptimizer(fc_lr)
        elif optimizer_type == "adagrad":
            conv_optimizer = tf.train.AdagradOptimizer(conv_lr, initial_accumulator_value=0.0001)
            fc_optimizer = tf.train.AdagradOptimizer(fc_lr, initial_accumulator_value=0.0001)
        elif optimizer_type == "sgd":
            conv_optimizer = tf.train.GradientDescentOptimizer(conv_lr)
            fc_optimizer = tf.train.GradientDescentOptimizer(fc_lr)
        elif optimizer_type == "momentum":
            conv_optimizer = tf.train.MomentumOptimizer(conv_lr, momentum=0.9)
            fc_optimizer = tf.train.MomentumOptimizer(fc_lr, momentum=0.9)
        elif optimizer_type == "adadelta":
            conv_optimizer = tf.train.AdadeltaOptimizer(learning_rate=conv_lr,rho=0.95,epsilon=1e-09)
            fc_optimizer = tf.train.AdadeltaOptimizer(learning_rate=conv_lr,rho=0.95,epsilon=1e-09)
        else : raise ValueError("{} optimizer doesn't exist.".format(optimizer_type))

        conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "conv")
        fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "fc")

        grads = tf.gradients(loss_op, conv_vars + fc_vars)
        conv_grads = grads[:len(conv_vars)]
        fc_grads = grads[len(conv_vars):]

        with tf.name_scope("grad_norms"):
            for v, grad in zip(conv_vars + fc_vars, grads):
                if grad is not None :
                    grad_norm_op = tf.nn.l2_loss(grad, name=format(v.name[:-2]))
                    tf.add_to_collection("grads", grad_norm_op)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            conv_train_op = conv_optimizer.apply_gradients(zip(conv_grads, conv_vars), name="conv_train_op")
        fc_train_op = fc_optimizer.apply_gradients(zip(fc_grads, fc_vars), global_step=net.global_iter_counter, name="fc_train_op")

    return tf.group(conv_train_op, fc_train_op)
