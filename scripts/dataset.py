import tensorflow as tf
from scripts import dataset
from tensorflow.contrib.data import Iterator
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

import numpy as np
import pandas as pd
import scipy, copy, cv2
from tqdm import tqdm

# one hot encoding
def one_hot_encoding(labels):
    return np.eye(np.max(labels) + 1)[labels].reshape(labels.shape[0],np.max(labels) + 1)

# process met dataset
class met:
    def __init__(self, csv_file, re_img_size=(227,227), is_valid=False, batch_size = None,
                 Rotate=False, Fliplr=False, Shuffle=False, one_hot=False, dataset_root=""):
        """
        Arguments
            csv_file : insert met csv file (No Default)
            re_img_size : insert image size you want to resize (Default : 227,227)
            is_valid : make unseen coordinates (but have values) get activated (Default : False)
            Rotate : rotate images and coordinates (Default : False)
            Fliplr : mirror images and coordinates (Default : False)
            Shuffle : Shuffle dataset and labels before return (Default : False)
            
            ex)
               met = dataset.met("csv_loc", Fliplr=True ,Shuffle=True) 
        Variables
            MM_norm_coord = relative coordinates normalized by Vector MinMaxScaling
            coor_set = coordinates shaped [..., 14, 2] (and augmented coordinates if augmentation arguments activated)
            head_norm = relative coordinates normalized by head relative coord.
            img_path = image paths
            img_set = loaded images (and augmented images if augmentation arguments activated)
            joint_coors = coordinates shaped [..., 28] (Originals from csv)
            joint_is_valid = 0 if a joint is valid else 1
            means = mean values for each joints of each classes
            rel_coor = relative coordinates
            
            ex)
                met.head_norm
        """
        joints=pd.read_csv(csv_file,header=None).as_matrix()
        
        # Parsing csv
        self.re_img_size=re_img_size
        self.img_path=list(path for path in joints[:,0])
        self.joint_coors=np.array(list(coors for coors in joints[:,1:29]))
        self.joint_is_valid=np.array(list(valid for valid in joints[:,29:43])) 
        self.scores=np.array(list(scores for scores in joints[:,43]))[:,np.newaxis]  # MET Score
        self.labels=np.array(list(labels for labels in joints[:,44]))[:,np.newaxis]  # Activities
        
        # make space for image instances
        self.img_set=np.zeros([len(self.img_path),re_img_size[0],re_img_size[1],3])
        # reshape coordinates
        self.coor_set=np.array(self.joint_coors).reshape(len(self.joint_coors),14,2)
        
        # Get each joints coordinates mean value
        self.means = self._get_coor_means(csv_file,self.coor_set,np.max(self.labels)+1)
        
        # Load images & coords with Augmentations
        with tqdm(total=len(self.img_path)) as pbar_process:
            pbar_process.set_description("[Processing Images & Coordinates]")
            for i, path in enumerate(self.img_path):
                img=cv2.imread(dataset_root+path)
                self.img_set[i]=cv2.resize(img,(re_img_size[0],re_img_size[1]), interpolation=cv2.INTER_LINEAR)

                for j in range(len(self.coor_set[i])):
                    if is_valid and bool(self.joint_is_valid[i][j]): self.coor_set[i][j] = [-1,-1]

                    if self.coor_set[i][j][0] == -1: pass
                    else:
                            self.coor_set[i][j][0] = self.coor_set[i][j][0]*(re_img_size[0]/img.shape[1])
                            self.coor_set[i][j][1] = self.coor_set[i][j][1]*(re_img_size[1]/img.shape[0])
                pbar_process.update(1)
                
        valid_expanded = np.concatenate((self.joint_is_valid[:,:,np.newaxis], self.joint_is_valid[:,:,np.newaxis]),
                                        axis=2).reshape(len(self.joint_is_valid),-1)
        def reverseNum(num):
            return 0 if num==1 else 1
        for i in range(len(valid_expanded)):
            valid_expanded[i]=np.array(list((lambda x : map(reverseNum,x))(valid_expanded[i])))
        self.joint_is_valid = valid_expanded
                
        # fill missing or -1 value to each joints mean values
        for i, coors in enumerate(self.coor_set):
            if list(coors.reshape(-1)).count(-1) > 0 :
                label = (joints[i][-1])
                for j in range(len(coors)):
                    if coors[j,0]==-1:
                        self.coor_set[i,j] = self.means[label,j]
        
        # Rotate images and coords if 'Rotate' is True
        if Rotate :
            rotated = self._rotation(copy.copy(self.img_set), copy.copy(self.coor_set), copy.copy(self.labels),
                                     copy.copy(self.joint_is_valid), copy.copy(self.scores))
            self.img_set = np.concatenate((self.img_set, rotated['images']), axis=0)
            self.coor_set = np.concatenate((self.coor_set, rotated['joints']), axis=0)
            self.joint_is_valid = np.concatenate((self.joint_is_valid, rotated['valid']), axis=0)
            self.labels = np.concatenate((self.labels, rotated['labels']),axis=0)
            self.scores = np.concatenate((self.scores, rotated['scores']),axis=0)
            
        # mirror images and coords if 'Fliplr' is True
        if Fliplr :
            fliplred = self._mirroring(copy.copy(self.img_set), copy.copy(self.coor_set), copy.copy(self.labels),
                                       copy.copy(self.joint_is_valid), copy.copy(self.scores))
            self.img_set = np.concatenate((self.img_set, fliplred['images']), axis=0)
            self.coor_set = np.concatenate((self.coor_set, fliplred['joints']), axis=0)
            self.joint_is_valid = np.concatenate((self.joint_is_valid, fliplred['valid']), axis=0)
            self.labels = np.concatenate((self.labels, fliplred['labels']),axis=0)
            self.scores = np.concatenate((self.scores, fliplred['scores']),axis=0)
        
        # Shuffle images and labels if 'Shuffle' is True
        if Shuffle :
            shuffled = self._shuffling(copy.copy(self.img_set), copy.copy(self.coor_set), copy.copy(self.labels), 
                                       copy.copy(self.joint_is_valid), copy.copy(self.scores))
            self.img_set = shuffled['images']
            self.coor_set = shuffled['joints']
            self.joint_is_valid = shuffled['valid']
            self.labels = shuffled['labels']
            self.scores = shuffled['scores']
        
        # make labels one-hot vectors if 'one_hot' is True
        if one_hot :
            self.labels = one_hot_encoding(self.labels)
            
        # get relatives coordinates
        self.rel_coor = np.array( list(self._rel_coor(self.coor_set[i]) for i in range(len(self.coor_set))))
        # MinMaxScaler following Vector quantities
        self.MM_norm_coord = np.array(list(map(self._Vec_MinMaxScaler,self.rel_coor)))
        # Normalize by head vector
        self.head_norm_coord = np.array(list(map(self._head_basis,self.rel_coor)))
    
    
    
        # make batch_set
        if batch_size == None : pass
        else :
            # get available number of batchs
            self.num_batchs = int(len(self.img_set)/batch_size)+1 if len(self.img_set) % batch_size !=0 else int(len(self.img_set)/batch_size)
            
            img_data = []
            joint_set = []
            valid_set = []
            
            # make dataset batchs
            for n in range(self.num_batchs):
                img_data.append(self.img_set[n*batch_size:(n+1)*batch_size] if n != self.num_batchs-1 else self.img_set[n*batch_size:])
                joint_set.append(self.coor_set.reshape(len(self.coor_set),-1)[n*batch_size:(n+1)*batch_size] if n != self.num_batchs-1 else self.coor_set.reshape(len(self.coor_set),-1)[n*batch_size:])
                valid_set.append(self.joint_is_valid[n*batch_size:(n+1)*batch_size] if n != self.num_batchs-1 else self.joint_is_valid[n*batch_size:])
            
            self.batch_set = {'img':np.array(img_data),'joints':np.array(joint_set),'valid':np.array(joint_set)}
    
    
    
    #----------------------------
    #  function definitions
    #----------------------------
    
    #  Augmentation functions
    
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
                    x, y = img.shape[:2]
                    rotation_matrix = cv2.getRotationMatrix2D((x/2,y/2),theta,1.0)
                    img_rotated = cv2.warpAffine(img, rotation_matrix, (x,y))
                    
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
        mirrored_valid = joint_is_valid.reshape(-1,14,2)
        
        with tqdm(total=len(images)) as pbar:
            pbar.set_description("[Mirroring Images & Coordinates]")
            for i, img in enumerate(images):
                mirrored_img[i] = np.fliplr(img)
                for j, joint in enumerate(joints[i]):
                    mirrored_coor[i][j][1] = joint[1]
                    if joint[0] > (img.shape[0]/2):
                        mirrored_coor[i][j][0] = (lambda x : x-2*(x-(img.shape[0]/2)))(joint[0])
                    elif joint[0] < (img.shape[0]/2):
                        mirrored_coor[i][j][0] = (lambda x : x+2*((img.shape[0]/2)-x))(joint[0])
                    elif joint[0] == -1: pass
                pbar.update(1)
                mirrored_coor[i] = mirrored_coor[i][[5,4,3,2,1,0,11,10,9,8,7,6,12,13]]
                mirrored_valid[i] = mirrored_valid[i][[5,4,3,2,1,0,11,10,9,8,7,6,12,13]]
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
        
        mean_set = np.zeros((num_classes,coor_set.shape[1],coor_set.shape[2]))

        for cl in range(num_classes):
            count=0
            for i in range(len(joints)):
                if joints[i,-1] == cl:
                    mean_set[cl]=mean_set[cl]+coor_set[i]
                    count+=1
            mean_set[cl]=mean_set[cl]/count
        return mean_set
            
    #  Normalization functions
    
    def _Vec_MinMaxScaler(self, coord):
        r_coord = coord[:13]**2
        coord_list = list(r_coord[:,0]+r_coord[:,1])

        minval = coord[coord_list.index(min(coord_list))]
        maxval = coord[coord_list.index(max(coord_list))]
        numerator = r_coord - minval
        denominator = maxval - minval
        return numerator/ (denominator + 1e-8)
    
    def _head_basis(self, coord):
        return coord[:14]/(coord[13] + 1e-8)