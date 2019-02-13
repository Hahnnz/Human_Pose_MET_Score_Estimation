import cv2, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from scripts.config import *

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
    
    def eval_strict_pcp(gt_joints, predicted_joints, thresh=0.5):

        is_matched = np.zeros((len(gt_joints),(len(gt_joints[0]['sticks']))), dtype=int)

        for n in range(len(gt_joints)):
            for i in range(len(gt_joints[n]['sticks'])):

                stick_len=np.linalg.norm(gt_joints[n]['sticks'][i,:2] - gt_joints[n]['sticks'][i,2:])
                if stick_len == 0 : stick_len = 1e-8

                delta_a = np.linalg.norm(predicted_joints[n]['sticks'][0, :2] -
                                         gt_joints[n]['sticks'][0, :2]) / stick_len 
                delta_b = np.linalg.norm(predicted_joints[n]['sticks'][0, 2:] -
                                         gt_joints[n]['sticks'][0, 2:]) / stick_len

                is_matched[n,i]=(delta_a <= thresh and delta_b <= thresh)

        return np.mean(is_matched, 0)
    
    def average_pcp_left_right_limbs(pcp_per_stick):
        part_names = ['Head', 'Torso', 'U Arm', 'L Arm', 'U Leg', 'L Leg', 'mean']
        pcp_per_part = pcp_per_stick[:2].tolist() + [(pcp_per_stick[i] + pcp_per_stick[i + 4]) / 2 for i in range(2, 6)]
        pcp_per_part.append(np.mean(pcp_per_part))
        return pcp_per_part, part_names
    
class etc:
    def pad_bbox_coor(img, bbox_coor, scale):
        # check a given scale value is between 0.0 to 5.0
        scale_range = np.array(list(map(str,np.linspace(0.,5.,51))))
        scale_range = list(map((lambda x : float(x[:3])),scale_range))
        if np.array(scale) in scale_range:
            scale = int(scale*10)
        else :
            raise ValueError('scale should be in range 0.0 to 5.0')

        # check img
        if len(img.shape) == 3 and img.shape[-1] not in [1,3]:
            raise ValueError('Single image is required')

        w,h = img.shape[:2]

        gaps = np.array([0,0,w,h]) - np.array(bbox_coor)
        padding_unit = gaps/50

        padded_coor = padding_unit*scale + np.array(bbox_coor)
        padded_coor = np.round(padded_coor)

        for i in range(len(padded_coor)):
            padded_coor[i] = 0 if padded_coor[i] < 0 and i in [0,1] else padded_coor[i]
            padded_coor[i] = 0 if padded_coor[i] > w and i == 2 else padded_coor[i]
            padded_coor[i] = 0 if padded_coor[i] > h and i == 3 else padded_coor[i]

        return np.array(padded_coor, np.int)
    
    def markJoints(img, joints, weight=0.01):
        img = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        dot_weight = sum(img.shape[0:2])/len(img.shape[0:2]) * weight
        dot_weight = int(dot_weight) if dot_weight > 1 else 1
        
        for i in range(len(joints)):
            x, y = map(int,joints[i])
            if x!=-1: 
                cv2.circle(img, (x, y), dot_weight, (0, 0, 255), thickness=-1)
                cv2.putText(img, str(i+1), (x,y), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
        return img
    
    def drawSticks(img, sticks, weight=0.005):
        Stick_Color = [(0,0,255),(39,157,47),(255,0,0),
                       (0,238,255),(221,0,255),(0,187,255),(255,0,0),
                       (0,238,255),(221,0,255),(0,187,255)]

        stick_weight = sum(img.shape[0:2])/len(img.shape[0:2]) * weight
        stick_weight = int(stick_weight) if stick_weight > 1 else 1
        
        for i in range(len(Stick_Color)):
            img=cv2.line(img.copy(), (int(sticks[i,0]),int(sticks[i,1])),
                         (int(sticks[i,2]),int(sticks[i,3])), Stick_Color[i], stick_weight)
        return img

    def set_GPU(device_num):
        if type(device_num) is str:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"]=device_num
        else : raise ValueError("devuce number should be specified in str type")
            
    def normalize_img(img):
        tmp_shape = img.shape
        img = img.astype(np.float32)
        img -= img.reshape(-1, 3).mean(axis=0)
        img /= img.reshape(-1, 3).std(axis=0) + 1e-5
        img = img.reshape(tmp_shape)
        return img
    
class analysis:
    def get_pcp_stick_result_table_per_activities(gt_labels, gt_canonical,pred_canonical):
        class_pred_result = [[] for i in range(10)]
        class_orig_result = [[] for i in range(10)]

        for i in range(len(pred_canonical)):
            class_pred_result[gt_labels.squeeze()[i]].append(pred_canonical[i])
            class_orig_result[gt_labels.squeeze()[i]].append(gt_canonical[i])

        class_pred_result = np.array(class_pred_result)
        class_orig_result = np.array(class_orig_result)

        pcp_result = []

        for i in range(len(class_pred_result)):
            pcp_value = pose.eval_strict_pcp(class_orig_result[i],class_pred_result[i])
            pcp_result.append(pose.average_pcp_left_right_limbs(pcp_value))

        pcp_value=pose.eval_strict_pcp(gt_canonical,pred_canonical)
        pcp_result.append(pose.average_pcp_left_right_limbs(pcp_value))

        pcp_result = np.array(pcp_result)[:,0]
        pcp_result = np.array([round(float(v),2) for v in pcp_result.reshape(-1)]).reshape(-1,7)

        pcp_table = pd.DataFrame(pcp_result.transpose(1,0))
        pcp_table.index = sticks
        pcp_table.columns = classes + ['Average']
        return pcp_table
    
    def show_pcp_result_plot_per_sticks(gt_labels, gt_canonical, pred_canonical, save=False, save_path=None):
        if save and save_path == None :
            raise ValueError('You have to specify the path you want to save. : save_path = "./path/to/".')
        if save and not os.path.exists(save_path) :
            os.makedirs(save_path)
        
        class_pred_result = [[] for i in range(10)]
        class_orig_result = [[] for i in range(10)]

        for i in range(len(pred_canonical)):
            class_pred_result[gt_labels.squeeze()[i]].append(pred_canonical[i])
            class_orig_result[gt_labels.squeeze()[i]].append(gt_canonical[i])

        class_pred_result = np.array(class_pred_result)
        class_orig_result = np.array(class_orig_result)
        
        pcp = [[] for _ in range(7)]
        pcp_result = []

        for thresh in [0.1,0.2,0.3,0.4,0.5]:
            pcp_value = pose.eval_strict_pcp(gt_canonical,pred_canonical,thresh)
            average_pcp = pose.average_pcp_left_right_limbs(pcp_value)
            for i in range(len(average_pcp[1])):
                pcp[i].append(average_pcp[0][i])

        fig, pcp_plot = plt.subplots(1,2)
        fig.set_size_inches(18,6)
        
        pcp_result.append(pose.average_pcp_left_right_limbs(pcp_value))

        pcp_result = np.array(pcp_result)[:,0]
        pcp_result = np.array([round(float(v),2) for v in pcp_result.reshape(-1)]).reshape(-1,7)

        for i, v in enumerate(pcp_result[-1]):
            pcp_plot[0].bar(x=sticks[i],height=v,color=sticks_color[i])
            pcp_plot[0].text(i,v,str(v),fontsize=15,color='gray')
        #pcp_plot[0].plot(pcp_result[-1], marker='o', c='aqua')
        pcp_plot[0].set_ylim(0.0,1.0)
        pcp_plot[0].grid(b=True, which='major',c='silver')
        pcp_plot[0].set_xlabel('Body parts')
        pcp_plot[0].set_ylabel('Percentage of Correct Parts')

        for i, c in enumerate(sticks_color):    
            pcp_plot[1].plot([0.1,0.2,0.3,0.4,0.5],pcp[i],c=c,marker='o')
        pcp_plot[1].set_xlabel('Threshold')
        pcp_plot[1].set_ylabel('Percentage of Correct Parts')
        pcp_plot[1].set_ylim(0.0,1.0)
        pcp_plot[1].legend(average_pcp[1], loc=2,)
        pcp_plot[1].grid(b=True, which='major',c='silver')
        
        if save:
            fig.savefig(save_path+"pcp_result_plot_per_sticks.pdf", bbox_inches='tight')
            
        plt.show()
        
    def visualize_Variances_per_joint(gt_labels, gt_joints, pred_joints, save=False, save_path=None):
        
        if save and save_path == None :
            raise ValueError('You have to specify the path you want to save. : save_path = "./path/to/".')
        if save and not os.path.exists(save_path) :
            os.makedirs(save_path)
        
        fig, points = plt.subplots(2,7)
        fig.set_size_inches(27,9)

        for idx in range(len(joints)):
            x_error = pred_joints[:,idx,0]-gt_joints[:,idx,0]
            y_error = pred_joints[:,idx,1]-gt_joints[:,idx,1]

            points[idx//7][idx%7].set_title(joints[idx]+' - Bias & Variance')
            points[idx//7][idx%7].plot(np.linspace(-80, 80), np.linspace(0, 0), linestyle='--', c='black')
            points[idx//7][idx%7].plot(np.linspace(0, 0), np.linspace(-80, 80), linestyle='--', c='black')
            points[idx//7][idx%7].add_artist(Ellipse((0,0),x_error.std(),y_error.std(),linewidth=2,linestyle='--', fill=False))

            for i, label in enumerate(gt_labels.squeeze()):
                points[idx//7][idx%7].scatter(pred_joints[i,idx,0]-gt_joints[i,idx,0],
                                              pred_joints[i,idx,1]-gt_joints[i,idx,1],c=class_color[label])
        if save:
            fig.savefig(save_path+"Variances_per_joint.pdf", bbox_inches='tight')
            
        plt.show()
    
    def hist_Variance_Bias_per_joint(gt_joints, pred_joints, save=False, save_path=None):
        
        if save and save_path == None :
            raise ValueError('You have to specify the path you want to save. : save_path = "./path/to/".')
        if save and not os.path.exists(save_path) :
            os.makedirs(save_path)
            
        errors = pred_joints-gt_joints.reshape(-1,14,2)
        joints_errors = [[] for _ in range(len(joints))]

        for j_error in errors:
            for i, coor_error in enumerate(j_error):
                joints_errors[i].append(coor_error)

        joints_errors = np.array(joints_errors)

        Variance = []
        for j in range(len(joints)):
            diff = joints_errors[j].reshape(-1)
            dist = diff - joints_errors[j].mean()
            Variance.append(sum(map(abs,dist))/len(diff))

        Bias = [abs(joints_errors[j].mean()) for j in range(len(joints))]

        fig, Var_Bias = plt.subplots(1,2)
        fig.set_size_inches(18,6)

        short_joint_name = [''.join(name[0] for name in j.split()) for j in joints[:-2]] + ['Ne', 'Hd']

        for i, v in enumerate(Variance):
            Var_Bias[0].bar(x=short_joint_name[i], height=v)
            Var_Bias[0].text(i,v,str(round(v,3)),fontsize=10,color='black')
        Var_Bias[0].set_xlabel('Joints')
        Var_Bias[0].set_ylabel('Variance')
        Var_Bias[0].set_title('Joint_Variance')

        for i, v in enumerate(Bias):
            Var_Bias[1].bar(x=short_joint_name[i], height=v)
            Var_Bias[1].text(i,v,str(round(v,3)),fontsize=12,color='black')
        Var_Bias[1].set_xlabel('Joints')
        Var_Bias[1].set_ylabel('Bias')
        Var_Bias[1].set_title('Joint_Bias')

        if save:
            fig.savefig(save_path+"Variance_Bias_per_joint_histogram.pdf", bbox_inches='tight')
        
        plt.show()
    
    def plot_total_pcp_result(gt_labels, gt_canonical,pred_canonical, save=False, save_path=None):
        
        if save and save_path == None :
            raise ValueError('You have to specify the path you want to save. : save_path = "./path/to/".')
        if save and not os.path.exists(save_path) :
            os.makedirs(save_path)
        
        class_pred_result = [[] for i in range(10)]
        class_orig_result = [[] for i in range(10)]

        for i in range(len(pred_canonical)):
            class_pred_result[gt_labels.squeeze()[i]].append(pred_canonical[i])
            class_orig_result[gt_labels.squeeze()[i]].append(gt_canonical[i])

        class_pred_result = np.array(class_pred_result)
        class_orig_result = np.array(class_orig_result)

        pcp_result = []

        for i in range(len(class_pred_result)):
            pcp_value = pose.eval_strict_pcp(class_orig_result[i],class_pred_result[i])
            pcp_result.append(pose.average_pcp_left_right_limbs(pcp_value))

        pcp_value=pose.eval_strict_pcp(gt_canonical,pred_canonical)
        pcp_result.append(pose.average_pcp_left_right_limbs(pcp_value))

        pcp_result = np.array(pcp_result)[:,0]
        pcp_result = np.array([round(float(v),2) for v in pcp_result.reshape(-1)]).reshape(-1,7)

        fig, points = plt.subplots(2,5)
        fig.set_size_inches(40, 10)

        for idx in range(len(classes)):
            for body_part in sticks[:-1]:
                points[idx//5][idx%5].bar(x=body_part, height=0)
            points[idx//5][idx%5].set_title(classes[idx]+' - PCP')
            points[idx//5][idx%5].plot(pcp_result[-1,:-1], marker='o',c='black',linestyle='--')
            points[idx//5][idx%5].set_ylim([0.1,1.1])
            points[idx//5][idx%5].plot(pcp_result[idx,:-1], marker='o',c=class_color[idx], linewidth=3)
            points[idx//5][idx%5].grid(b=True, which='major',c='silver')

        if save:
            fig.savefig(save_path+"total_pcp_result.pdf", bbox_inches='tight')
            
        plt.show()
    
    def demo_plot(img, gt_canonical ,pred_canonical, save=False, save_path=None):
        
        if save and save_path == None :
            raise ValueError('You have to specify the path you want to save. : save_path = "./path/to/".')
        if save and not os.path.exists(save_path) :
            os.makedirs(save_path)
        
        orig_img1 = img.copy()
        orig_img2 = img.copy()
        orig_img3 = img.copy()

        pred_img1 = img.copy()
        pred_img2 = img.copy()
        pred_img3 = img.copy()


        orig_img1=etc.markJoints(img=orig_img1, joints=gt_canonical['joints'])
        orig_img2=etc.drawSticks(img=orig_img2, sticks=gt_canonical['sticks'])

        pred_img1=etc.markJoints(img=pred_img1, joints=pred_canonical['joints'])
        pred_img2=etc.drawSticks(img=pred_img2, sticks=pred_canonical['sticks'])

        orig_img3=etc.markJoints(img=orig_img3, joints=gt_canonical['joints'])  
        orig_img3=etc.drawSticks(img=orig_img3, sticks=gt_canonical['sticks'])  

        pred_img3=etc.markJoints(img=pred_img3, joints=pred_canonical['joints'])
        pred_img3=etc.drawSticks(img=pred_img3, sticks=pred_canonical['sticks'])

        fig, ((p11,p12),(p21,p22),(p31,p32),) = plt.subplots(3,2)
        fig.set_size_inches(15, 15)

        p11.set_title("groundTruth joints")
        p11.imshow(orig_img1[:,:,[2,1,0]])
        p12.set_title("predicted joints")
        p12.imshow(pred_img1[:,:,[2,1,0]])

        p21.set_title("groundTruth sticks")
        p21.imshow(orig_img2[:,:,[2,1,0]])
        p22.set_title("predicted sticks")
        p22.imshow(pred_img2[:,:,[2,1,0]])

        p31.set_title("groundTruth joints with sticks")
        p31.imshow(orig_img3[:,:,[2,1,0]])
        p32.set_title("predicted joints with sticks")
        p32.imshow(pred_img3[:,:,[2,1,0]])
        
        if save:
            fig.savefig(save_path+"total_pcp_result.pdf", bbox_inches='tight')
        
        plt.show()
        
    def show_dataset(img_set, labels, save=False, save_path=None):
        if save and save_path == None :
            raise ValueError('You have to specify the path you want to save. : save_path = "./path/to/".')
        if save and not os.path.exists(save_path) :
            os.makedirs(save_path)

        class_imgs = [[] for _ in range(10)]

        for i, c in enumerate(labels):
            class_imgs[c].append(img_set[i])

        fig, data_explore = plt.subplots(10,5)
        fig.set_size_inches(15, 30)

        for i in range(5):
            rand_idx = np.random.choice(range(len(class_imgs[1])),10,replace=False)
            for c, clas in enumerate(rand_idx):
                if i == 0 : data_explore[c][0].set_ylabel(classes[c], fontsize=18)
                data_explore[c][i].imshow(class_imgs[c][clas][:,:,[2,1,0]])
                data_explore[c][i].set_yticklabels([])
                data_explore[c][i].set_xticklabels([])

        plt.subplots_adjust(wspace=0, hspace=0)
        if save:
            fig.savefig('dataset.pdf', bbox_inches='tight')
        plt.show()
        
    def show_estimated(img_set, labels, pred_canonical, save=False, save_path=None):
        if save and save_path == None :
            raise ValueError('You have to specify the path you want to save. : save_path = "./path/to/".')
        if save and not os.path.exists(save_path) :
            os.makedirs(save_path)

        class_imgs = [[] for _ in range(10)]
        pred_canonical_list = [[] for _ in range(10)]

        for i, c in enumerate(labels):
            class_imgs[c].append(img_set[i])
            pred_canonical_list[c].append(pred_canonical[i])

        fig, data_explore = plt.subplots(10,5)
        fig.set_size_inches(15, 30)

        for i in range(5):
            rand_idx = np.random.choice(range(len(class_imgs[1])),10,replace=False)
            for c, clas in enumerate(rand_idx):
                if i == 0 : data_explore[c][0].set_ylabel(classes[c], fontsize=18)
                orig_img = class_imgs[c][clas].copy()
                orig_img = etc.drawSticks(img=orig_img, sticks=pred_canonical_list[c][clas]['sticks'], weight=0.02)

                data_explore[c][i].imshow(orig_img[:,:,[2,1,0]])
                data_explore[c][i].set_yticklabels([])
                data_explore[c][i].set_xticklabels([])

        plt.subplots_adjust(wspace=0, hspace=0)
        if save:
            fig.savefig('estimated.pdf', bbox_inches='tight')
        plt.show()
        
    def plot_variance_bias_on_image(index, images, gt_joints, pred_joints, save=False, save_path=None):

        if save and save_path == None :
            raise ValueError('You have to specify the path you want to save. : save_path = "./path/to/".')
        if save and not os.path.exists(save_path) :
            os.makedirs(save_path)

        ground_truth = gt_joints[index]

        fig, result = plt.subplots(1,1)
        fig.set_size_inches(9,9)

        for joint_idx in range(len(ground_truth)):
            x_error = pred_joints[:,joint_idx,0]-gt_joints[:,joint_idx,0]
            y_error = pred_joints[:,joint_idx,1]-gt_joints[:,joint_idx,1]

            result.scatter(x_error + ground_truth[joint_idx,0], y_error + ground_truth[joint_idx,1])

        result.imshow(images[index][:,:,[2,1,0]])
        result.axis('off')

        if save:
            fig.savefig(save_path+"variances_on_image.pdf", bbox_inches='tight')

        plt.show()