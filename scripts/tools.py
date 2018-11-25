import cv2, os, glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
    def markJoints(img, joints):
        img = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(joints)):
            x, y = map(int,joints[i])
            if x!=-1: 
                cv2.circle(img, (x, y), 4, (0, 0, 255), thickness=-1)
                cv2.putText(img, str(i+1), (x,y), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
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
        
        Stick_Color = [Head, Torso, Right_Upper_Arm, Right_Lower_Arm,
                       Right_Upper_Leg, Right_Lower_Leg, Left_Upper_Arm,
                       Left_Lower_Arm, Left_Upper_Leg, Left_Lower_Leg]

        for i in range(len(Stick_Color)):
            img=cv2.line(img.copy(), (int(sticks[i,0]),int(sticks[i,1])),
                         (int(sticks[i,2]),int(sticks[i,3])), Stick_Color[i], 2)
        
        return img

    def set_GPU(device_num):
        if type(device_num) is str:
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"]=device_num
        else : raise ValueError("devuce number should be specified in str type")
            
    def explore_dir(dir,count=0,f_extensions=None):
        if count==0:
            global n_dir, n_file, filenames, filelocations
            n_dir=n_file=0
            filenames=list()
            filelocations=list()

        for img_path in sorted(glob.glob(os.path.join(dir,'*' if f_extensions is None else '*.'+f_extensions))):
            if os.path.isdir(img_path):
                n_dir +=1
                explore_dir(img_path,count+1)
            elif os.path.isfile(img_path):
                n_file += 1
                filelocations.append(img_path)
                filenames.append(img_path.split("/")[-1])
        return np.array((filenames,filelocations))
    
    def demo_plot(img, gt_canonical ,pred_canonical):
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
        
    def normalize_img(img):
        tmp_shape = img.shape
        img = img.astype(np.float32)
        img -= img.reshape(-1, 3).mean(axis=0)
        img /= img.reshape(-1, 3).std(axis=0) + 1e-5
        img = img.reshape(tmp_shape)
        return img