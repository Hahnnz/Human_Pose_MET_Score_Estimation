from scripts import dataset, tools
from models.regressionnet import Regressionnet
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import scripts.preprocessing as pp
import argparse

def training(img_size=(227,227), batch_size=800, iter_num=10000, lr=1e-2, dropout_ratio=0.7,
             start_save_step=100, save_path="./out/"):
    
    lowest_loss=None
    highest_pcp=None
    
    bbox_scale = [0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,
                  1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4]
    theta_set = [-15,-12.5,-10,-7.5,-5,-2.5,2.5,5,7.5,10,12.5,15]
    
    net = Regressionnet(data_shape=img_size+(3,),optimizer_type='adadelta',num_joints=14, gpu_memory_fraction=None)

    with net.graph.as_default():
        saver = tf.train.Saver()

        train_it = dataset.met("./dataset/MET6/met5_allcoor_train.csv", batch_size=batch_size,
                               Rotate=True, Shuffle=True, re_img_size=img_size,
                               theta_set=theta_set,dataset_root="./dataset/MET6/")
        test_it = dataset.met("./dataset/MET6/met5_allcoor_test.csv", batch_size=batch_size,
                              Rotate=True, Shuffle=True, re_img_size=img_size,
                              theta_set=theta_set,dataset_root="./dataset/MET6/")

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
                            [summary_op, net.train_op, net.pose_loss_op, net.fc_regression],
                            feed_dict={net.x : np.array(batch_data),
                                       net.y : np.array(batch_joints).reshape(-1,28),
                                       net.valid : train_it.batch_set['valid'][n],
                                       net.lr : lr,
                                       net.keep_prob : dropout_ratio,
                                       net.is_train : True})

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
                            [net.pose_loss_op, net.fc_regression],
                            feed_dict={net.x : np.array(batch_data),
                                       net.y : np.array(batch_joints).reshape(-1,28),
                                       net.valid: test_it.batch_set['valid'][n],
                                       net.keep_prob:1.0,
                                       net.is_train:False})

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

                if step > start_save:
                    if lowest_loss == None or lowest_loss > te_cost/te_cnt :
                        lowest_loss = te_cost/te_cnt
                        saver.save(net.sess, save_path+"Regressionnet_lowest_loss.ckpt")
                    if highest_pcp == None or highest_pcp < te_acc :
                        highest_pcp = te_acc
                        saver.save(net.sess, save_path+"Regressionnet_highest_pcp.ckpt")

                pbar.update(1)
                pbar.set_description("[ Step : "+str(step+1)+"]")
                pbar.set_postfix_str(" Train Acc : "+'%.6f' % (tr_acc)+" Train Loss : "+'%.4f' % (tr_cost/tr_cnt)+ \
                                     " Test Acc : "+'%.6f' % (te_acc)+" Test Loss : "+'%.4f' % (te_cost/te_cnt))

if __name__ == '__main__':
    #training()
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu_num", type=str, help = "set gpu devices number to use", default=0)
    parser.add_argument('img_size', type=str, help = 'set and resize image size to train', default=(227,227))
    parser.add_argument('batch_size', type=int,help = 'set batch size', default=100)
    parser.add_argument('iter_num', type=int, help = 'set max training step', default=10000)
    parser.add_argument('dropout_ratio', type=float, help='set dropout ratio', default=0.7)
    parser.add_argument('start_save_step', type=int, help='set start step to save', default=100)
    parser.add_argument('learning_rate', type=float, help='set learning rate', default=1e-2)
    args = parser.parse_args()
    
    tools.etc.set_GPU(args.gpu_num)
    save_path="./out/gpu"+args.gpu_num+"/"
    
    args.img_size = tuple(map(int, args.img_size.split(',')))
    training(img_size=args.img_size, batch_size=args.batch_size, iter_num=args.iter_num, lr=args.learning_rate,
             dropout_ratio=args.dropout_ratio, start_save_step=args.start_save_step, save_path=save_path)