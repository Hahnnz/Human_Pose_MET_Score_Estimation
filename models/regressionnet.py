from models import alexnet
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
                          optimizer_type="adam",
                          net_type="alexnet"):
    
    # Create Regressionnet. - Alexnet, Resnet
    with tf.Graph().as_default():

        if net_type == "alexnet":
            net = alexnet.alexnet(batch_size=batch_size,
                                  input_shape=(data_shape),output_shape=(num_joints*2,))
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
        train_op = tools.tftools.set_op(net, pose_loss_op, fc_lr=fc_lr_pl, conv_lr=conv_lr_pl, optimizer_type=optimizer_type)

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
