from models import alexnet
from models import Convnet1
from models import Convnet2
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
            drop7 = net.get_layers("drop7")
            net.fc_regression = fc(drop7, int(drop7.get_shape()[1]),
                                   num_joints*2, name="fc_regression", relu=False)
        elif net_type == "convnet2":
            net = Convnet2.convNet(batch_size=batch_size,
                                  input_shape=(data_shape),output_shape=(num_joints*2,),
                                 gpu_memory_fraction=gpu_memory_fraction)
            drop7 = net.get_layers("drop7")
            net.fc_regression = fc(drop7, int(drop7.get_shape()[1]),
                                   num_joints*2, name="fc_regression", relu=False)
        elif net_type == "resnet": 
            raise ValueError("regressionnet for ResNet will be updated soon")
        else : 
            raise ValueError("net type should be 'alexnet'. resnet will be updated soon")
            
        with tf.name_scope("PoseInput"):
            joints_gt = tf.placeholder(tf.float32, [batch_size, num_joints*2], name="joints_ground_truth")
            joints_is_valid = tf.placeholder(tf.float32, [batch_size, num_joints*2], name="joints_is_valid")
        
        diff = tf.subtract(joints_gt, net.fc_regression)
        diff_valid = tf.multiply(diff, joints_is_valid)

        num_valid_joints = tf.reduce_sum(joints_is_valid, axis=1) / tf.constant(2.0, dtype=tf.float32)

        pose_loss_op = tf.reduce_mean(tf.reduce_sum(
            tf.pow(diff_valid, 2), axis=1) / num_valid_joints, name="joint_euclidean_loss")
        
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

def get_metric(joints_gt, predicted_joints, orig_bboxes, metric_name="PCP"):
    joint_gt=joints_gt.copy()
    predicted_joints=predicted_joints.copy()
    predicted_joints=np.clip(predicted_joints, -0.5, 0.5)
    
    for i in range(gt_joints.shape[0]):
        joints_gt[i, :] = tools.pose.project_joints(joints_gt[i], orig_bboxes[i])
        predicted_joints[i, :] = tools.pose.project_joints(predicted_joints[i], orig_bboxes[i])
    
    joints_gt = tools.pose.convert2canonical(joints_gt)
    predicted_joints = tools.pose.convert2canonical(predicted_joints)
    
    if metric_name == "RelaxedPCP":
        full_scores = eval_relaxed_pcp(joints_gt, predicted_joints)
    elif metric_name == "PCP":
        full_scores = eval_strict_pcp(joints_gt, predicted_joints)
    elif metric_name == "PCKh":
        full_scores = eval_pckh(joints_gt, predicted_joints)
    else : raise ValueError("Unknown metric {}. 'PCP','RelaxedPCP','PCKh' is available.".format(metric_name))
    return full_scores
    
def create_summary(tag, value):
    x = summary_pb2.Summary.Value(tag=tag, simple_value=value)
    return summary_pb2.Summary(value=[x])
    
def evaluate_pcp(net, pose_loss_op, test_iterator, summary_writer):
    test_iter = copy.copy(test_iterator)
    num_test_examples = test_iter.img_set.get_shape().as_list()[0]
    num_batches = int(math.ceil(num_test_examples/test_iter.batch_size))
    next_batch = test_iterator.iterator.get_next()
    num_joints = int(int(net.fc_regression.get_shape()[1])/2)
    
    joints_gt = list()
    joints_is_valid = list()
    predicted_joints = list()
    orig_bboxes = list()
    
    total_loss=0.
    
    with tqdm(total=num_batches) as pbar:
        for step in range(num_batches):
            img_batch, joints_batch = net.sess.run(test_iter.next_batch)
            feed_dict = {net.x : img_batch, net.y_gt: joints_batch, keep_prob: 1.0}
            pred_joint, batch_loss = net.sess.run([net.fc_regression, pose_loss_op], feed_dict=feed_dict)
            
            
    """
    for i, batch in tqdm(enumerate(test_iter), total=num_batches):
        feeds = batch2feeds(batch)
        feed_dict = fill_joint_feed_dict(net, feeds, conv_lr=0.0, fc_lr=0.0)
        pred_j, batch_loss_value = net.sess.run([net.fc_regression, pose_loss_op],feed_dict=feed_dict)
        total_loss += batch_loss_value * len(batch)
        joints_gt.append(feeds[1])
        joints_is_valid.append(feeds[2])
        predicted_joints.append(pred_j.reshape(-1, num_joints, 2))
        orig_bboxes.append(np.vstack([x["bbox"] for x in feeds[3]]))
    """
    avg_loss = total_loss/num_test_examples
    joints_gt=np.vstack(joints_gt)
    joints_is_valid=np.vstack(joints_is_valid)
    predicted_joints=np.vstack(predicted_joints)
    orig_bboxes = np.vstack(orig_bboxes)
    
    pcp_per_stick = get_metric(joints_gt, predicted_joints, orig_bboxes)
    
    return pcp_per_stick


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