from scripts import dataset, tools
from models import alexnet, regressionnet
from tqdm import tqdm
import math, os
import tensorflow as tf
import numpy as np
import scripts.preprocessing as pp

# Set GPU and Network to use
gpuNum="0"

tools.etc.set_GPU(gpuNum)
net_type = 'convnet1'

batch_size = 200
iter_num = 50000
bbox_scale = [1.0,1.1,1.2,1.3,1.4,
             1.5,1.6,1.7,1.8,1.9,2.0]
lowest_loss=None
highest_pcp=None

# load Network
net, loss_op, pose_loss_op, train_op = regressionnet.create_regression_net(data_shape=(227,227,3),optimizer_type='adadelta',num_joints=14,net_type=net_type, gpu_memory_fraction=None)
    
with net.graph.as_default():
    saver = tf.train.Saver()

    train_it = dataset.met("/var/data/MET3/activity-met_n10_ub_new_train.csv", batch_size=batch_size,
                           Rotate=True, Shuffle=True,
                           theta_set=[-15,-13,-10,-7,-5,-3,3,5,7,10,13,15],dataset_root="/var/data/MET3/")
    test_it = dataset.met("/var/data/MET3/activity-met_n10_ub_new_test.csv", batch_size=batch_size,
                          Rotate=True, Shuffle=True,
                          theta_set=[-15,-13,-10,-7,-5,-3,3,5,7,10,13,15],dataset_root="/var/data/MET3/")
    
    summary_writer = tf.summary.FileWriter("./out/gpu"+gpuNum, net.sess.graph)
    summary_op = tf.summary.merge_all()


with tf.device("/gpu:"+gpuNum):
    with tqdm(total = iter_num) as pbar:
        for step in range(iter_num):
            #=============================#
            #       T R A I N I N G       #
            #=============================#
            tr_cost = 0.
            tr_cnt = 0
            tr_acc = []
            
            for scale in bbox_scale:
                for n in range(train_it.num_batchs):
                    batch_data = []
                    batch_joints = []
                    for i in range(len(train_it.batch_set['img'][n])):
                        bbox_img, bbox_coord = pp.apply_bbox(train_it.batch_set['img'][n][i],
                                                             train_it.batch_set['joints'][n][i].reshape(14,2),
                                                             train_it.batch_set['valid'][n][i],
                                                             scale, random_shift=True)
                        batch_data.append(bbox_img)
                        batch_joints.append(bbox_coord)

                    summary_str, _, loss_value, predicted_joints = net.sess.run(
                        [summary_op, train_op, pose_loss_op, net.fc_regression],
                        feed_dict={net.x : np.array(batch_data),
                                   'PoseInput/joints_ground_truth:0' : np.array(batch_joints).reshape(-1,28),
                                   'PoseInput/joints_is_valid:0': train_it.batch_set['valid'][n],
                                   'lr/conv_lr:0': 1e-2,
                                   'lr/fc_lr:0': 1e-2,
                                   net.keep_prob:0.7})
                    tr_cost+=loss_value
                    tr_cnt+=1

                    predicted_joints = predicted_joints.reshape(-1,14,2)
                    pred_canonical = tools.pose.convert2canonical(predicted_joints)
                    orig_canonical = tools.pose.convert2canonical(np.array(batch_joints).reshape(-1,14,2))
                    pcp_value=tools.pose.eval_strict_pcp(orig_canonical,pred_canonical)
                    average_pcp = tools.pose.average_pcp_left_right_limbs(pcp_value)[0]
                    tr_acc.append(average_pcp[-1])
            
            tr_acc = sum(tr_acc)/len(tr_acc)
            summary_writer.add_summary(summary_str, step)
            
            #=============================#
            #        T E S T I N G        #
            #=============================#
            te_cost = 0.
            te_cnt = 0
            te_acc = []
            
            for scale in bbox_scale:
                for n in range(test_it.num_batchs):
                    batch_data = []
                    batch_joints = []
                    for i in range(len(test_it.batch_set['img'][n])):
                        bbox_img, bbox_coord = pp.apply_bbox(test_it.batch_set['img'][n][i],
                                                             test_it.batch_set['joints'][n][i].reshape(14,2),
                                                             test_it.batch_set['valid'][n][i],
                                                             scale, random_shift=True)
                        batch_data.append(bbox_img)
                        batch_joints.append(bbox_coord)

                    loss_value, predicted_joints = net.sess.run(
                        [pose_loss_op, net.fc_regression],
                        feed_dict={net.x : np.array(batch_data),
                                   'PoseInput/joints_ground_truth:0' : np.array(batch_joints).reshape(-1,28),
                                   'PoseInput/joints_is_valid:0': test_it.batch_set['valid'][n],
                                  net.keep_prob:1.0})
                    te_cost+=loss_value
                    te_cnt+=1
                    predicted_joints = predicted_joints.reshape(-1,14,2)
                    pred_canonical = tools.pose.convert2canonical(predicted_joints)
                    orig_canonical = tools.pose.convert2canonical(np.array(batch_joints).reshape(-1,14,2))
                    pcp_value=tools.pose.eval_strict_pcp(orig_canonical,pred_canonical)
                    average_pcp = tools.pose.average_pcp_left_right_limbs(pcp_value)[0]
                    te_acc.append(average_pcp[-1])
            
            te_acc = sum(te_acc)/len(te_acc)
            
            summ = tf.Summary()
            summ.value.add(tag='Test arverage PCP mean', simple_value=te_acc)
            summ.value.add(tag='Train arverage PCP mean', simple_value=tr_acc)
            
            summary_writer.add_summary(summ,step)
            
            if step > 100:
                if lowest_loss == None or lowest_loss > te_cost/te_cnt :
                    lowest_loss = te_cost/te_cnt
                    saver.save(net.sess, "./out/gpu"+gpuNum+"/"+net_type+"_lowest_loss.ckpt")
                if highest_pcp == None or highest_pcp < te_acc :
                    highest_pcp = te_acc
                    saver.save(net.sess, "./out/gpu"+gpuNum+"/"+net_type+"_highest_pcp.ckpt")
            
            pbar.update(1)
            pbar.set_description("[ Step : "+str(step+1)+"]")
            pbar.set_postfix_str(" Train Acc : "+'%.6f' % (tr_acc)+" Train Loss : "+'%.4f' % (tr_cost/tr_cnt)+ \
                                 " Test Acc : "+'%.6f' % (te_acc)+" Test Loss : "+'%.4f' % (te_cost/te_cnt))
