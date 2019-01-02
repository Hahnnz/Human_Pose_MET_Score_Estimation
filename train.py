from scripts import dataset, tools
from models.regressionnet_resnet import Regressionnet
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import scripts.preprocessing as pp

# Set GPU and Network to use
gpuNum="0"

tools.etc.set_GPU(gpuNum)

batch_size = 40
iter_num = 10000
bbox_scale = [x/10 for x in range(10,25)]
theta_set = list(range(-30,30,5))
theta_set.remove(0)
lowest_loss=None
highest_pcp=None

# load Network

net = Regressionnet(data_shape=(128,128,3),optimizer_type='adadelta',num_joints=14, gpu_memory_fraction=None)
    
with net.graph.as_default():
    saver = tf.train.Saver()

    train_it = dataset.met("./dataset/MET7/met7_train.csv", batch_size=batch_size,
                           Rotate=True, Shuffle=True, re_img_size=(128,128), Fliplr=True,
                           normalize=True, theta_set=theta_set,dataset_root="./dataset/MET7/")
    test_it = dataset.met("./dataset/MET7/met7_test.csv", batch_size=batch_size,
                          Rotate=True, Shuffle=True, re_img_size=(128,128), Fliplr=True,
                          normalize=True, theta_set=theta_set,dataset_root="./dataset/MET7/")
    
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
            tr_pcp = 0.
            tr_diff_dv = 0
            tr_l2_loss = 0
            
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
                    
                    summary_str, _, loss_value, predicted_joints, diff_dv, l2_loss = net.sess.run(
                        [summary_op, net.train_op, net.pose_loss_op, net.fc_regression, net.diff_dv, net.l2_loss],
                        feed_dict={net.x : np.array(batch_data),
                                   net.y : np.array(batch_joints).reshape(-1,28),
                                   net.valid : train_it.batch_set['valid'][n],
                                   net.lr: 1.,
                                   net.keep_prob:0.7,
                                   net.is_train:True,
                                   net.Lambda : 1.})
                    
                    tr_cost+=loss_value
                    tr_diff_dv+=diff_dv
                    tr_l2_loss+=l2_loss
                    tr_cnt+=1

                    predicted_joints = predicted_joints.reshape(-1,14,2)
                    pred_canonical = tools.pose.convert2canonical(predicted_joints)
                    orig_canonical = tools.pose.convert2canonical(np.array(batch_joints).reshape(-1,14,2))
                    pcp_value=tools.pose.eval_strict_pcp(orig_canonical,pred_canonical)
                    average_pcp = tools.pose.average_pcp_left_right_limbs(pcp_value)[0]
                    tr_pcp += average_pcp[-1]
            
            summary_writer.add_summary(summary_str, step)
            
            #=============================#
            #        T E S T I N G        #
            #=============================#
            te_cost = 0.
            te_cnt = 0
            te_pcp = 0
            te_diff_dv = 0
            te_l2_loss = 0
            
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

                    loss_value, predicted_joints, diff_dv, l2_loss = net.sess.run(
                        [net.pose_loss_op, net.fc_regression, net.diff_dv, net.l2_loss],
                        feed_dict={net.x : np.array(batch_data),
                                   net.y : np.array(batch_joints).reshape(-1,28),
                                   net.valid: test_it.batch_set['valid'][n],
                                   net.keep_prob:1.0,
                                   net.is_train:False,
                                   net.Lambda:1.})
                    
                    te_cost+=loss_value
                    te_diff_dv+=diff_dv
                    te_l2_loss+=l2_loss
                    te_cnt+=1
                    
                    predicted_joints = predicted_joints.reshape(-1,14,2)
                    pred_canonical = tools.pose.convert2canonical(predicted_joints)
                    orig_canonical = tools.pose.convert2canonical(np.array(batch_joints).reshape(-1,14,2))
                    pcp_value=tools.pose.eval_strict_pcp(orig_canonical,pred_canonical)
                    average_pcp = tools.pose.average_pcp_left_right_limbs(pcp_value)[0]
                    te_pcp+=average_pcp[-1]
            
            summ = tf.Summary()
            summ.value.add(tag='Test arverage PCP mean', simple_value=te_pcp/te_cnt)
            summ.value.add(tag='Test Cosine_distance', simple_value=te_diff_dv/te_cnt)
            summ.value.add(tag='Test l2 Loss', simple_value=te_l2_loss/te_cnt)
            
            summ.value.add(tag='Train arverage PCP mean', simple_value=tr_pcp/tr_cnt)
            summ.value.add(tag='Train Cosine_distance', simple_value=tr_diff_dv/tr_cnt)
            summ.value.add(tag='Train l2 Loss', simple_value=tr_l2_loss/tr_cnt)
            
            summary_writer.add_summary(summ,step)
            
            if step > 0:
                if lowest_loss == None or lowest_loss > te_cost/te_cnt :
                    lowest_loss = te_cost/te_cnt
                    saver.save(net.sess, "./out/gpu"+gpuNum+"/Regressionnet_lowest_loss.ckpt")
                if highest_pcp == None or highest_pcp < te_pcp :
                    highest_pcp = te_pcp
                    saver.save(net.sess, "./out/gpu"+gpuNum+"/Regressionnet_highest_pcp.ckpt")
                    
            pbar.update(1)
            pbar.set_description("[ Step : "+str(step+1)+"]")
            pbar.set_postfix_str(" Train Acc : "+'%.6f' % (tr_pcp)+" Train Loss : "+'%.4f' % (tr_cost/tr_cnt)+ \
                                 " Test Acc : "+'%.6f' % (te_pcp)+" Test Loss : "+'%.4f' % (te_cost/te_cnt))

