from scripts import dataset, tools
from models import alexnet, regressionnet
from tqdm import tqdm
import math, os
import tensorflow as tf
import numpy as np

# Set GPU and Network to use
tools.etc.set_GPU("3")
net_type = 'convnet1'

batch_size = 10
iter_num = 10000
snapshot_step = 100

# load Network
net, loss_op, pose_loss_op, train_op = regressionnet.create_regression_net(data_shape=(227,227,3),optimizer_type='adadelta',num_joints=14,net_type=net_type, gpu_memory_fraction=None)
    
with net.graph.as_default():
    saver = tf.train.Saver()

    train_it = dataset.met("./dataset/train.csv", Fliplr=True, Shuffle=True, 
                           batch_size=batch_size, dataset_root = "./dataset/")
    test_it = dataset.met("./dataset/test.csv", Fliplr=True, Shuffle=True, 
                          batch_size=batch_size, dataset_root="./dataset/")

    summary_writer = tf.summary.FileWriter("./out/", net.sess.graph)
    summary_op = tf.summary.merge_all()


with tf.device("/gpu:0"):
    with tqdm(total = iter_num) as pbar:
        for step in range(iter_num):
            #=============================#
            #       T R A I N I N G       #
            #=============================#
            tr_cost = 0.
            tr_cnt = 0
            tr_acc = []
            
            for n in range(train_it.num_batchs):
                summary_str, _, loss_value, predicted_joints = net.sess.run(
                    [summary_op, train_op, pose_loss_op, net.fc_regression],
                    feed_dict={net.x : train_it.batch_set['img'][n],
                               'PoseInput/joints_ground_truth:0' : train_it.batch_set['joints'][n],
                               'PoseInput/joints_is_valid:0': train_it.batch_set['valid'][n],
                               'lr/conv_lr:0': 0.001,
                               'lr/fc_lr:0': 0.001,
                               net.keep_prob:0.7})
                tr_cost+=loss_value
                tr_cnt+=1

                predicted_joints = predicted_joints.reshape(-1,14,2)
                pred_canonical = tools.pose.convert2canonical(predicted_joints)
                orig_canonical = tools.pose.convert2canonical(train_it.batch_set['joints'][n].reshape(-1,14,2))
                pcp_value=tools.pose.eval_strict_pcp(orig_canonical,pred_canonical)
                average_pcp = tools.pose.average_pcp_left_right_limbs(pcp_value)[0]
                tr_acc.append(average_pcp[-1])
            
            tr_acc = sum(tr_acc)/len(tr_acc)
            
            
            #=============================#
            #        T E S T I N G        #
            #=============================#
            te_cost = 0.
            te_cnt = 0
            te_acc = []
            
            for n in range(test_it.num_batchs):
                loss_value, predicted_joints = net.sess.run(
                    [pose_loss_op, net.fc_regression],
                    feed_dict={net.x : test_it.batch_set['img'][n],
                               'PoseInput/joints_ground_truth:0' : test_it.batch_set['joints'][n],
                               'PoseInput/joints_is_valid:0': test_it.batch_set['valid'][n],
                              net.keep_prob:1.0})
                te_cost+=loss_value
                te_cnt+=1
                predicted_joints = predicted_joints.reshape(-1,14,2)
                pred_canonical = tools.pose.convert2canonical(predicted_joints)
                orig_canonical = tools.pose.convert2canonical(test_it.batch_set['joints'][n].reshape(-1,14,2))
                pcp_value=tools.pose.eval_strict_pcp(orig_canonical,pred_canonical)
                average_pcp = tools.pose.average_pcp_left_right_limbs(pcp_value)[0]
                te_acc.append(average_pcp[-1])
            
            te_acc = sum(te_acc)/len(te_acc)
            
            if step%snapshot_step==0 and step !=0  and step > 3000:
                saver.save(net.sess, "./out/"+net_type+"_gpu3_"+str(step)+".ckpt")
            
            pbar.update(1)
            pbar.set_description("[ Step : "+str(step+1)+"]")
            pbar.set_postfix_str(" Train Acc : "+'%.6f' % (tr_acc)+" Train Loss : "+'%.4f' % (tr_cost/tr_cnt)+ \
                                 " Test Acc : "+'%.6f' % (te_acc)+" Test Loss : "+'%.4f' % (te_cost/te_cnt))