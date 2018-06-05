import numpy as np
import pandas as pd
import scipy, glob, os, copy, cv2
from scipy.io import loadmat
from tqdm import tqdm

import tensorflow as tf
from scripts import dataset
from tensorflow.contrib.data import Iterator
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

def one_hot_encoding(labels):
    return np.eye(np.max(labels) + 1)[labels].reshape(labels.shape[0],np.max(labels) + 1)

class met:
    def __init__(self, csv_file, re_img_size=(227,227), is_valid=False, 
                 Rotate=False, Fliplr=False, Shuffle=False, one_hot=False):
        joints=pd.read_csv(csv_file,header=None).as_matrix()
        
        self.re_img_size=re_img_size
        self.img_path=list(path for path in joints[:,0])
        self.joint_coors=np.array(list(coors for coors in joints[:,1:29]))
        self.joint_is_valid=np.array(list(is_valid for is_valid in joints[:,29:43]))
        self.labels=np.array(list(labels for labels in joints[:,43]))[:,np.newaxis]
        self.scores=np.array(list(scores for scores in joints[:,44]))[:,np.newaxis]
        
        self.img_set=np.zeros([len(self.img_path),re_img_size[0],re_img_size[1],3])
        self.coor_set=np.array(self.joint_coors).reshape(len(self.joint_coors),14,2)
        
        self.means = self._get_coor_means(csv_file,self.coor_set,max(self.labels)[0]+1)
        
        for i, coors in enumerate(self.coor_set):
            if list(coors.reshape(-1)).count(-1) > 0 :
                label = (joints[i][-2])
                for j in range(len(coors)):
                    if coors[j,0]==-1:
                        self.coor_set[i,j] = self.means[label,j]
        
        with tqdm(total=len(self.img_path)) as pbar_process:
            pbar_process.set_description("[Processing Images & Coordinates]")
            for i, path in enumerate(self.img_path):
                img=cv2.imread(path)
                self.img_set[i]=cv2.resize(img,(re_img_size[0],re_img_size[1]), interpolation=cv2.INTER_CUBIC)

                for j in range(len(self.coor_set[i])):
                    if is_valid and bool(self.joint_is_valid[i][j]): self.coor_set[i][j] = [-1,-1]

                    if self.coor_set[i][j][0] == -1: pass
                    else:
                            self.coor_set[i][j][0] = self.coor_set[i][j][0]*(re_img_size[0]/img.shape[1])
                            self.coor_set[i][j][1] = self.coor_set[i][j][1]*(re_img_size[1]/img.shape[0])
                pbar_process.update(1)
 
        if Rotate :
            rotated = self._rotation(copy.copy(self.img_set), copy.copy(self.coor_set), copy.copy(self.labels),
                                     copy.copy(self.joint_is_valid), copy.copy(self.scores))
            self.img_set = np.concatenate((self.img_set, rotated['images']), axis=0)
            self.coor_set = np.concatenate((self.coor_set, rotated['joints']), axis=0)
            self.joint_is_valid = np.concatenate((self.joint_is_valid, rotated['valid']), axis=0)
            self.labels = np.concatenate((self.labels, rotated['labels']),axis=0)
            self.scores = np.concatenate((self.scores, rotated['scores']),axis=0)
            
        if Fliplr :
            fliplred = self._mirroring(copy.copy(self.img_set), copy.copy(self.coor_set), copy.copy(self.labels),
                                       copy.copy(self.joint_is_valid), copy.copy(self.scores))
            self.img_set = np.concatenate((self.img_set, fliplred['images']), axis=0)
            self.coor_set = np.concatenate((self.coor_set, fliplred['joints']), axis=0)
            self.joint_is_valid = np.concatenate((self.joint_is_valid, fliplred['valid']), axis=0)
            self.labels = np.concatenate((self.labels, fliplred['labels']),axis=0)
            self.scores = np.concatenate((self.scores, fliplred['scores']),axis=0)
        
        if Shuffle :
            shuffled = self._shuffling(copy.copy(self.img_set), copy.copy(self.coor_set), copy.copy(self.labels), 
                                       copy.copy(self.joint_is_valid), copy.copy(self.scores))
            self.img_set = shuffled['images']
            self.coor_set = shuffled['joints']
            self.joint_is_valid = shuffled['valid']
            self.labels = shuffled['labels']
            self.scores = shuffled['scores']
            
        if one_hot :
            self.labels = one_hot_encoding(self.labels)
            
        self.rel_coor = np.array( list(self._rel_coor(self.coor_set[i]) for i in range(len(self.coor_set))))
    
    
    def _rotation(self, images, joints, labels, joint_is_valid, scores):
        thetas = np.deg2rad((-30,-20,-10,10,20,30))
        rotated_img = np.zeros([images.shape[0]*len(thetas), images.shape[1],images.shape[2],3])
        rotated_coor = np.zeros([joints.shape[0]*len(thetas), joints.shape[1],2])
        rotated_valid = joint_is_valid
        rotated_labels = np.zeros([labels.shape[0]*len(thetas),1])
        rotated_scores = np.zeros([scores.shape[0]*len(thetas),1])
        
        with tqdm(total=len(images)*len(thetas)) as pbar:
            pbar.set_description("[Rotating Images & Coordinates")
            for i, img in enumerate(images):
                for j, theta in enumerate(thetas):
                    img_rotated = scipy.ndimage.rotate(img, theta)
                    org_center = (np.array(img.shape[:2][::-1])-1)/2
                    rotated_center = (np.array(img_rotated.shape[:2][::-1])-1)/2

                    coor_list = list(np.array(coor)+org_center for coor in joints[i])
                    rotated_coor_list = list((lambda x : (x[0]*np.cos(theta) + x[1]*np.sin(theta) + rotated_center[0],
                                                          -x[0]*np.sin(theta) + x[1]*np.cos(theta) + rotated_center[1])
                                             )(coor) for coor in coor_list)
                    
                    rotated_orig= img_rotated.shape[:2]
                    img_rotated = cv2.resize(img_rotated, (self.re_img_size[0],self.re_img_size[1]), interpolation=cv2.INTER_CUBIC)

                    rotated_img[(i*len(thetas))+j]=img_rotated
                    rotated_coor[(i*len(thetas))+j]=np.array(rotated_coor_list)*(self.re_img_size[0]/rotated_orig[0])
                    rotated_labels[(i*len(thetas))+j] = labels[i]
                    rotated_scores[(i*len(thetas))+j] = scores[i]
                    pbar.update(1)
        
        for i in range(len(thetas)-1):
            rotated_valid = np.concatenate((rotated_valid, joint_is_valid), axis=0)
            
        return {'images':rotated_img,'joints':rotated_coor,'valid':rotated_valid,
                'labels':rotated_labels,'scores':rotated_scores}

    def _mirroring(self, images, joints, labels, joint_is_valid, scores):
        mirrored_img = np.zeros([images.shape[0], images.shape[1],images.shape[2],3])
        mirrored_coor = np.zeros([joints.shape[0], joints.shape[1],2])
        
        with tqdm(total=len(images)) as pbar:
            pbar.set_description("[Mirroring Images & Coordinates]")
            for i, img in enumerate(images):
                mirrored_img[i] = np.fliplr(img)
                for j, joint in enumerate(joints[i]):
                    mirrored_coor[i][j][1] = joint[1]
                    if joint[0] > (img.shape[0]/2):
                        mirrored_coor[i][j][0] = (lambda x : x-2*(x-(img.shape[0]/2)))(joint[0])
                    elif joint[0] > (img.shape[0]/2):
                        mirrored_coor[i][j][0] = (lambda x : x+2*((img.shape[0]/2)-x))(joint[0])
                    elif joint[0] == -1: pass
                pbar.update(1)
        return {'images':mirrored_img,'joints':mirrored_coor,'valid':copy.copy(joint_is_valid),
                'labels':copy.copy(labels),'scores':copy.copy(scores)}
    
    def _shuffling(self, images, joints, labels, joint_is_valid, scores):
        shuffled_img = np.zeros([images.shape[0], images.shape[1],images.shape[2],3])
        shuffled_coor = np.zeros([joints.shape[0], joints.shape[1],2])
        shuffled_valid = np.zeros([len(joint_is_valid),14])
        shuffled_labels = np.zeros([labels.shape[0],1])
        shuffled_scores = np.zeros([scores.shape[0],1])

        indices=np.random.permutation(len(shuffled_img))
        shuffled_img, shuffled_coor, shuffled_valid, shuffled_labels, shuffled_scores = images[indices], joints[indices], joint_is_valid[indices], labels[indices],  scores[indices]
        
        return {'images': shuffled_img,'joints':shuffled_coor,'valid':shuffled_valid,
                'labels':shuffled_labels,'scores':shuffled_scores}

    def _rel_coor(self,coors):
        coors=np.array((coors[:,0],coors[:,1]),dtype=np.float32)
        filt = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [-1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                         [0,-1,1,0,0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,1,-1,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,1,-1,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,-1,1,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,-1,1,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,1,-1,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0,1,-1,0,0],
                         [0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                         [0,0,-1,-1,0,0,0,0,-1,-1,0,0,-1,1],
                         [0,0,0,0,0,0,0,0,0,0,0,0,1,0],],dtype=np.float32)
        result= np.matmul(coors,filt)
        return np.array( list((lambda x: (result[0,i],result[1,i]))(i) for i in range(len(result[0]))) )
    
    def _get_coor_means(self, csv_file ,coor_set,num_classes):
        joints=pd.read_csv(csv_file,header=None).as_matrix()
        
        mean_set = np.zeros((num_classes,coor_set.shape[1],coor_set.shape[2]), dtype=np.float32)

        for cl in range(num_classes):
            count=0
            for i in range(len(joints)):
                if joints[i,-2] == cl:
                    mean_set[cl]=mean_set[cl]+coor_set[i]
                    count+=1
            mean_set[cl]=mean_set[cl]/count
        return mean_set
    
class iterator:
    def __init__(self, csv_file, batch_size, mode, Rotate=False, Fliplr=False, Shuffle=False):
        if not mode.lower() in {"classification", "regression", "all"}:
            raise ValueError("mode must be given 'classification', 'regression' or 'all'.")
        
        data = met(csv_file,Rotate=Rotate,Fliplr=Fliplr,Shuffle=Shuffle)
        
        # True : Classification mode
        # False : Regression mode
        self._mode = True if mode.lower()=="classification" else False
        self.batch_size = batch_size
        self.num_classes = max(data.labels)[0]+1
        
        self.img_set = convert_to_tensor(data.img_set, dtype=dtypes.float64)
        if self._mode:
            self.labels = convert_to_tensor(data.labels[:,0], dtype= dtypes.int32)
        elif not self._mode:
            self.coor_set = convert_to_tensor(data.coor_set.reshape(len(data.coor_set), -1), dtype = dtypes.float64) 

        data = tf.data.Dataset.from_tensor_slices((self.img_set, tf.one_hot(self.labels, self.num_classes) if self._mode else self.coor_set))
        data = data.batch(self.batch_size)

        self.iterator = Iterator.from_structure(data.output_types, data.output_shapes)
        self.init_op = self.iterator.make_initializer(data)