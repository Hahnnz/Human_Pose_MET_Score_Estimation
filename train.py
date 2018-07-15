from scripts import dataset, tools, iterator
from models import alexnet, regressionnet
from tensorflow.contrib.data import Iterator
from IPython.display import clear_output
from copy import copy
from tqdm import tqdm
import math, os
import tensorflow as tf
import numpy as np

# Set GPU and Network to use
tools.etc.set_GPU("3")
net_type = 'convnet1'

# load Network
net, loss_op, pose_loss_op, train_op = regressionnet.create_regression_net(data_shape=(227,227,3),optimizer_type='adadelta',num_joints=14,net_type=net_type, gpu_memory_fraction=None)
    
with net.graph.as_default():
    saver = tf.train.Saver()

    train_it = dataset.met("./dataset/train.csv", Fliplr=True, Shuffle=True, dataset_root = "./dataset/")
    test_it = dataset.met("./dataset/test.csv", Fliplr=True, Shuffle=True, dataset_root="./dataset/")

    summary_writer = tf.summary.FileWriter("./out/", net.sess.graph)
    summary_op = tf.summary.merge_all()

    
batch_size = 10
tr_batch_num = int(len(train_it.img_set)/batch_size)+1 if len(train_it.img_set) % batch_size !=0 else int(len(train_it.img_set)/batch_size)
te_batch_num = int(len(test_it.img_set)/batch_size)+1 if len(test_it.img_set) % batch_size !=0 else int(len(test_it.img_set)/batch_size)
iter_num = 100000
snapshot_step = 100
global_step = None
cur_train_op = train_op

with tf.device("/gpu:0"):
    with tqdm(total = iter_num) as pbar:
        for step in range(iter_num):
            #=============================#
            #       T R A I N I N G       #
            #=============================#
            tr_cost = 0.
            tr_cnt = 0
            tr_acc = []
            
            for n in range(tr_batch_num):
                x = train_it.img_set[n*batch_size:(n+1)*batch_size] if n != tr_batch_num-1 else train_it.img_set[n*batch_size:]
                y=train_it.coor_set.reshape(len(train_it.coor_set),-1)[n*batch_size:(n+1)*batch_size] if n != tr_batch_num-1 else train_it.coor_set.reshape(len(train_it.coor_set),-1)[n*batch_size:]
                v=train_it.joint_is_valid[n*batch_size:(n+1)*batch_size] if n != tr_batch_num-1 else train_it.joint_is_valid[n*batch_size:]
                global_step, summary_str, _, loss_value, predicted_joints = net.sess.run(
                    [net.global_iter_counter,summary_op,cur_train_op,pose_loss_op, net.fc_regression],
                    feed_dict={net.x : x,
                               'PoseInput/joints_ground_truth:0' : y,
                               'PoseInput/joints_is_valid:0': v,
                               'lr/conv_lr:0': 0.001,
                               'lr/fc_lr:0': 0.001,
                               net.keep_prob:0.5})
                tr_cost+=loss_value
                tr_cnt+=1

                predicted_joints = predicted_joints.reshape(-1,14,2)
                pred_canonical = tools.pose.convert2canonical(predicted_joints)
                orig_canonical = tools.pose.convert2canonical(y.reshape(-1,14,2))
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
            
            for n in range(te_batch_num):
                if len(test_it.img_set) % batch_size !=0:
                    
                    x = test_it.img_set[n*batch_size:(n+1)*batch_size] if n != te_batch_num-1 else test_it.img_set[n*batch_size:]
                    y=test_it.coor_set.reshape(len(test_it.coor_set),-1)[n*batch_size:(n+1)*batch_size] if n != te_batch_num-1 else test_it.coor_set.reshape(len(test_it.coor_set),-1)[n*batch_size:]
                    v=test_it.joint_is_valid[n*batch_size:(n+1)*batch_size] if n != te_batch_num-1 else test_it.joint_is_valid[n*batch_size:]
                elif len(test_it.img_set) % batch_size ==0:
                    x = test_it.img_set[n*batch_size:(n+1)*batch_size]
                    y = test_it.coor_set.reshape(len(test_it.coor_set),-1)[n*batch_size:(n+1)*batch_size]
                    v=test_it.joint_is_valid[n*batch_size:(n+1)*batch_size]
                    
                loss_value, predicted_joints = net.sess.run(
                    [pose_loss_op, net.fc_regression],
                    feed_dict={net.x : x,
                               'PoseInput/joints_ground_truth:0' : y,
                               'PoseInput/joints_is_valid:0': v,
                              net.keep_prob:1.0})
                te_cost+=loss_value
                te_cnt+=1
                predicted_joints = predicted_joints.reshape(-1,14,2)
                pred_canonical = tools.pose.convert2canonical(predicted_joints)
                orig_canonical = tools.pose.convert2canonical(y.reshape(-1,14,2))
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