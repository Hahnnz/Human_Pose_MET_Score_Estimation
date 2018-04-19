import cv2
import os
import skimage
import numpy as np
import tensorflow as tf
from skimage import draw

def _joints2sticks(joints):
    # Input :
    # Head Top, Neck, Right shoulder, Right elbos, Right Wrist, Right hip, Right knee, Right ankle,
    # Left shoulder, Left elbow, Left wrist, Left hip, Left knee, Left ankle
        
    # Output :
    # Head, Torso, Right Upper Arm, Right Lower Arm, Right Upper Leg, Right Lower Leg,
    # Left Upper Arm, Left Lower Arm, Left Upper Leg, Left Lower Leg

    stick_n = 10 
    sticks = np.zeros((stick_n, 4), dtype=np.float32)
        
    # Head
    sticks[0, :] = np.hstack([joints[0, :], joints[1, :]])
    # Torso
    sticks[1, :] = np.hstack([(joints[2, :]+joints[8, :])/2.0, (joints[5, :]+joints[11, :])/2.0])
    # Left Upper Arm
    sticks[2, :] = np.hstack([joints[2, :],joints[3, :]])
    # Left Lower Arm
    sticks[3, :] = np.hstack([joints[3, :],joints[4, :]])
    # Left Upper Leg
    sticks[4, :] = np.hstack([joints[5, :],joints[6, :]])
    # Left Lower Leg
    sticks[5, :] = np.hstack([joints[6, :],joints[7, :]])
    # Right Upper Arm
    sticks[6, :] = np.hstack([joints[8, :],joints[9, :]])
    # Right Lower Arm
    sticks[7, :] = np.hstack([joints[9, :],joints[10, :]])
    # Right Upper Leg
    sticks[8, :] = np.hstack([joints[11, :],joints[12, :]])
    # Right Lower Leg
    sticks[9, :] = np.hstack([joints[12, :],joints[13, :]])
        
    return sticks

    def _pcp_err(joints_gt, predicted_joints):
        if len(joints_gt) != len(predicted_joints): 
            raise ValueError("Length of ground_truth(gt) must be equal to length of predicted")
        if len(joint_gt) ==0: raise ValueError("array is empty")
        num_sticks=joints_gt[0]["sticks"].shape[0]
        if num_sticks != 10:
            raise ValueError('PCP requires 10 sticks. Provided: {}'.format(num_sticks))

class pose:
    
    def convert2canonical(joints):
        assert joints.shape[1:] == (14,2), "joints must be 14"
        joint_order = [13,12,8,7,6,2,1,0,9,10,11,3,4,5]
        # order :
        # Head Top, Neck, Right shulder, Right elbos, Right Wrist, Right hip, Right knee, Right ankle
        # Left shoulder, Left elbow, Left wrist, Left hip, Left knee, Left ankle
        canonical = [dict() for _ in range(joints.shape[0])]
        for i in range(joints.shape[0]):
            canonical[i]["joints"] = joints[i, joint_order, :]
            canonical[i]["sticks"] = _joints2sticks(canonical[i]["joints"])
        return canonical
    
    def project_joints(joints, original_bbox):
        if joints.shape[1]!=2: raise ValueError("joints must be 2D array [num_joints x 2]")
        if joints.min() < -0.501 or joints.max() > 0.501:
            raise ValueError("'Joints\' coordinates must be normalized and be in [-0.5, 0.5], got[{}, {}]'.format(joints.min(), joints.max())")
        original_bbox = original_bbox.astype(int)
        x, y, w, h = original_bbox
        projected_joints = np.array(joints, dtype=np.float32)
        projected_joints += np.array([0.5, 0.5])
        projected_joints[:, 0] *= w
        projected_joints[:, 1] *= h
        projected_joints += np.array([x, y])
        return projected_joints

    def eval_relaxed_pcp(joints_gt, predicted_joints, thresh=0.5):
        _pcp_err(joints_gt, predicted_joints)
        is_matched = np.zeros((len(joints_gt), len(joints_gt[0]["sticks"].shape[0])), dtype=int)
        for i in range(len(joints_gt)):
            for stick_id in range(len(joints_gt[0]["sticks"].shape[0])):
                gt_stick_len = np.linalg.norm(joints_gt[i]['sticks'][stick_id, :2] -
                                              joints_gt[i]['sticks'][stick_id, 2:])
                delta_a = np.linalg.norm(predicted_joints[i]['sticks'][stick_id, :2] -
                                         joints_gt[i]['sticks'][stick_id, :2]) / gt_stick_len
                delta_b = np.linalg.norm(predicted_joints[i]['sticks'][stick_id, 2:] -
                                         joints_gt[i]['sticks'][stick_id, 2:]) / gt_stick_len
                delta = (delta_a + delta_b) / 2.0
            is_matched[i, stick_id] = delta <= thresh
        pcp_per_stick = np.mean(is_matched,0)
        return pcp_per_stick
    
    def eval_strict_pcp(joints_gt, predicted_joints, thresh=0.5):
        _pcp_err(joints_gt, predicted_joints)
        is_matched = np.zeros((len(joints_gt), len(joints_gt[0]["sticks"].shape[0])), dtype=int)
        
        for i in range(len(joints_gt)):
            for stick_id in range(len(joints_gt[0]["sticks"].shape[0])):
                gt_stick_len = np.linalg.norm(joints_gt[i]['sticks'][stick_id, :2] -
                                              joints_gt[i]['sticks'][stick_id, 2:])
                delta_a = np.linalg.norm(predicted_joints[i]['sticks'][stick_id, :2] -
                                         joints_gt[i]['sticks'][stick_id, :2]) / gt_stick_len
                delta_b = np.linalg.norm(predicted_joints[i]['sticks'][stick_id, 2:] -
                                         joints_gt[i]['sticks'][stick_id, 2:]) / gt_stick_len
            is_matched[i, stick_id] = (delta_a <= thresh and delta_b <= thresh)
        pcp_per_stick = np.mean(is_matched, 0)
        return pcp_per_stick
    
    def eval_pckh(joints_gt, predicted_joints, thresh=0.5):
        _pcp_err(joints_gt, predicted_joints)
        is_matched = np.zeros((len(joints_gt), len(joints_gt[0]["sticks"].shape[0])), dtype=int)
        
        num_joints=14
        num_examples=len(joints_gt)
        
        for i in range(num_examples):
            if gt_joints[i]['joints'].shape != (num_joints, 2):
                raise ValueError('lsp dataset PCKh requires 14 joints with 2D coordinates for each.'
                             ' Person {}: provided joints shape: {}'.format(i, joints_gt[0]['joints'].shape))
            head_id = 0
            gt_head_len = np.linalg.norm(joints_gt[i]['sticks'][head_id, :2] -
                                     joints_gt[i]['sticks'][head_id, 2:])
            for joint_id in range(num_joints):
                delta = np.linalg.norm(predicted_joints[i]['joints'][joint_id] - 
                                       joints_gt[i]['joints'][joint_id]) / gt_head_len

                is_matched[i, joint_id] = delta <= thresh
        pckh_per_joint = np.mean(is_matched, 0)
        
        return pcp_per_stick
    
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
    
    def drawSticks(img, sticks):
        Head=(255,0,0)
        Torso=(255,94,94)
        Right_Upper_Arm=(255,187,0)
        Right_Lower_Arm=(255,228,0)
        Right_Upper_Leg=(171,242,0)
        Right_Lower_Leg=(29,219,22)
        Left_Upper_Arm=(0,216,255)
        Left_Lower_Arm=(0,84,255)
        Left_Upper_Leg=(1,0,255)
        Left_Lower_Leg=(95,0,255)
        
        Stick_Color=np.array([Head, Torso, Right_Upper_Arm, Right_Lower_Arm, Right_Upper_Leg,Right_Lower_Leg, Left_Upper_Arm, Left_Lower_Arm, Left_Upper_Leg, Left_Lower_Leg])
        
        for i in range(10):
            scsc=sticks[i]
            rr,cc=skimage.draw.line(int(scsc[1]),int(scsc[0]),int(scsc[3]),int(scsc[2]))
            img[rr,cc]=Stick_Color[i]
        
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
