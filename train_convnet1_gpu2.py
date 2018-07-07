from scripts import dataset, tools, iterator
from models import alexnet, regressionnet
from tensorflow.contrib.data import Iterator
from IPython.display import clear_output
from copy import copy
from tqdm import tqdm
import math, os
import tensorflow as tf
import numpy as np

tools.etc.set_GPU("2")
net_type = 'convnet1'

net, loss_op, pose_loss_op, train_op = regressionnet.create_regression_net(data_shape=(227,227,3),optimizer_type='adadelta',num_joints=14,net_type=net_type, gpu_memory_fraction=None)
    
with net.graph.as_default():
    saver = tf.train.Saver()

    train_it = dataset.met("/var/data/MET2/activity-met_n10_ub_train.csv", Fliplr=True, Shuffle=True)
    test_it = dataset.met("/var/data/MET2/activity-met_n10_ub_test.csv", Fliplr=True, Shuffle=True)

    summary_writer = tf.summary.FileWriter("./out/", net.sess.graph)
    summary_op = tf.summary.merge_all()

    
batch_size = 10
tr_batch_num = int(len(train_it.img_set)/batch_size)+1
iter_num = 100000
snapshot_step = 5000
global_step = None
cur_train_op = train_op

with tf.device("/gpu:0"):
    with tqdm(total = iter_num) as pbar:
        for step in range(iter_num):
                
            tr_cost = 0.
            tr_cnt = 0
            for n in range(tr_batch_num):
                x = train_it.img_set[n*batch_size:(n+1)*batch_size] if n != tr_batch_num-1 else train_it.img_set[n*batch_size:]
                y=train_it.coor_set.reshape(952,-1)[n*batch_size:(n+1)*batch_size] if n != tr_batch_num-1 else train_it.coor_set.reshape(952,-1)[n*batch_size:]
                v=train_it.joint_is_valid[n*batch_size:(n+1)*batch_size] if n != tr_batch_num-1 else train_it.joint_is_valid[n*batch_size:]
                global_step, summary_str, _, loss_value, score = net.sess.run(
                    [net.global_iter_counter,summary_op,cur_train_op,pose_loss_op, net.fc_regression],
                    feed_dict={net.x : x,
                               'PoseInput/joints_ground_truth:0' : y,
                               'PoseInput/joints_is_valid:0': v,
                               'lr/conv_lr:0': 1.,
                               'lr/fc_lr:0': 1.,
                               net.keep_prob:0.5})
                tr_cost+=loss_value
                tr_cnt+=1
                
            
            loss_value = net.sess.run(
                [pose_loss_op],
                feed_dict={net.x : test_it.img_set,
                           'PoseInput/joints_ground_truth:0' : test_it.coor_set.reshape(len(test_it.coor_set),-1),
                           'PoseInput/joints_is_valid:0': test_it.joint_is_valid,
                           'lr/conv_lr:0': 1.,
                           'lr/fc_lr:0': 1.,
                          net.keep_prob:0.5})
            
            if step%snapshot_step==0 and step !=0:
                saver.save(net.sess, "./out/"+net_type+"_"+str(step)+".ckpt")
            
            pbar.update(1)
            pbar.set_description("[ Step : "+str(step+1)+"]")
            pbar.set_postfix_str(" Train Loss : "+str(tr_cost/tr_cnt)+" Test Loss : "+str(loss_value[0]))
