import cv2
import os
import skimage
import tensorflow as tf
from skimage import draw

class etc:
    def markJoints(img, joints):  
        circSize=5
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(14):
            x = int(joints[i,0])
            y = int(joints[i,1])
            if x!=-1:
                rr, cc = skimage.draw.circle(y, x, circSize)
                skimage.draw.set_color(img, (rr, cc), (1,0,0))
                cv2.putText(img, str(i+1), (x,y), font, 0.5, (0.5,0.5,0.5), 2, cv2.LINE_AA)
        return img

    def set_GPU(device_num):
        if type(device_num) is str:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"]=device_num
        else : raise ValueError("devuce number should be specified in str type")
        
class tftools:
    def set_op(net, loss_op, fc_lr, conv_lr, optimizer_type="adam"):
        with net.graph.as_default():
            if optimizer_type=="adam":
                conv_optimizer = tf.train.AdamOptimizer(conv_lr)
                fc_optimizer = tf.train.AdamOptimizer(fc_lr)
            elif optimizer_type == "adagrad":
                conv_optimizer = tf.train.AdagradOptimizer(conv_lr, initial_accumulator_value=0.0001)
                fc_optimizer = tf.train.AdagradOptimizer(fc_lr, initial_accumulator_value=0.0001)
            elif optimizer_type == "sgd":
                conv_optimizer = tf.train.GrdientDescentOptimizer(conv_lr)
                fc_optimizer = tf.train.GrdientDescentOptimizer(fc_lr)
            elif optimizer_type == "momentum":
                conv_optimizer == tf.train.MomentumOptimizer(conv_lr, momentum=0.9)
                fc_optimizer == tf.train.MomentumOptimizer(fc_lr, momentum=0.9)
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