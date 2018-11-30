import tensorflow as tf
import numpy as np
import pandas as pd
import scipy, cv2
import scripts.preprocessing as pp
from multiprocessing import Pool as pool
from tqdm import tqdm

# one hot encoding
def one_hot_encoding(labels):
    return np.eye(np.max(labels) + 1)[labels].reshape(labels.shape[0],np.max(labels) + 1)

# process met dataset
class met:
    def __init__(self, csv_file, re_img_size = (128,128), is_valid = False, batch_size = None,
                 Rotate = False, Fliplr = False, Shuffle = False, Bbox = False, Shift = False, normalize = False,
                 one_hot = False, theta_set = None, scale_set = None, Bbox_mode = "", random_time = None,
                 dataset_root = ""):
        
        joints = np.array(pd.read_csv(csv_file,header = None))
        
        # Parsing csv
        self.re_img_size = re_img_size
        self.img_path = list(path for path in joints[:,0])
        self.joint_coors = np.array(list(coors for coors in joints[:,1:29])).reshape(-1,14,2)
        self.joint_is_valid = np.array(list(valid for valid in joints[:,29:43])) 
        self.scores = np.array(list(scores for scores in joints[:,43]))[:,np.newaxis]  # MET Score
        self.labels = np.array(list(labels for labels in joints[:,44]))[:,np.newaxis]  # Activities
        
        # init list
        self.img_set = []
        self.coor_set = []
        valid = []
        scores = []
        labels = []
        
        # Load images & coords with Augmentations
        with tqdm(total=len(self.img_path)) as pbar_process:
            description = "[Processing Images & Coordinates]" if not Rotate else "[Processing Images & Coordinates With Rotation]"
            pbar_process.set_description(description)
            
            for i, path in enumerate(self.img_path):
                img = cv2.imread(dataset_root+path)
                
                if normalize :
                    tmp_shape = img.shape
                    img = img.astype(np.float32)
                    img -= img.reshape(-1, 3).mean(axis=0)
                    img /= img.reshape(-1, 3).std(axis=0) + 1e-5
                    img = img.reshape(tmp_shape)

                self.img_set.append(cv2.resize(img,re_img_size))
                joints = self.joint_coors[i].copy()
                for j in range(len(joints)):
                    joints[j] = ([joints[j][0]*(re_img_size[0]/img.shape[1]),
                                  joints[j][1]*(re_img_size[1]/img.shape[0])])
                self.coor_set.append(joints)
                valid.append(self.joint_is_valid[i])
                scores.append(self.scores[i])
                labels.append(self.labels[i])
                
                # Rotate images and Coords if arg-'Rotate' is True
                if Rotate :
                        
                    if theta_set is None :
                        raise ValueError("theta_set is empty.")
                    for theta in theta_set:
                        h, w = img.shape[:2]
                        center = (w//2, h//2)

                        rotation_matrix = cv2.getRotationMatrix2D(center,-theta,1.0)
                        img_rotated = cv2.warpAffine(img, rotation_matrix, (w,h))

                        n=len(joints)
                        joints_rotated = np.array(np.c_[self.joint_coors[i].copy(), np.ones(n)] * np.mat(rotation_matrix).transpose())
                        
                        for j in range(len(joints_rotated)):
                            joints_rotated[j] = ([joints_rotated[j][0]*(re_img_size[0]/img.shape[1]),
                                                  joints_rotated[j][1]*(re_img_size[1]/img.shape[0])])
                        
                        self.img_set.append(cv2.resize(img_rotated,re_img_size))
                        self.coor_set.append(joints_rotated)
                        valid.append(self.joint_is_valid[i])
                        scores.append(self.scores[i])
                        labels.append(self.labels[i])
                pbar_process.update(1)
                
        # convert list to numpy array
        self.img_set = np.array(self.img_set)
        self.coor_set = np.array(self.coor_set)
        self.joint_is_valid = np.array(valid)
        self.scores = np.array(scores)
        self.labels = np.array(labels)
                
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
        # mirror images and coords if 'Fliplr' is True
        if Fliplr :
            fliplred = self._mirroring(self.img_set.copy(), self.coor_set.copy(), self.labels.copy(), 
                                       self.joint_is_valid.copy(), self.scores.copy())
            self.img_set = np.concatenate((self.img_set, fliplred['images']), axis=0)
            self.coor_set = np.concatenate((self.coor_set, fliplred['joints']), axis=0)
            self.joint_is_valid = np.concatenate((self.joint_is_valid, fliplred['valid']), axis=0)
            self.labels = np.concatenate((self.labels, fliplred['labels']),axis=0)
            self.scores = np.concatenate((self.scores, fliplred['scores']),axis=0)
            
        if Bbox:
            if scale_set is None :
                raise ValueError("scale_set is empty.")
            if Bbox_mode.lower() not in ['augment', 'apply', 'random_shift'] :
                raise ValueError("Bbox mode must be defined as 'augment' or 'apply' or 'random_shift'.")
            if Bbox_mode.lower() == 'random_shift' and random_time == None:
                raise ValueError("must insert the random time how many times you want to make a random bbox.")
            if random_time is not None and type(random_time) is not int:
                raise TypeError("You must insert random_time as int type.")
            
            bbox_img_set = []
            bbox_coor_set = []
            bbox_valid_set = []
            bbox_label_set = self.labels.copy()
            bbox_score_set = self.scores.copy()
            
            with tqdm(total=len(self.img_set)*len(scale_set)) as pbar:
                pbar.set_description("[ {{BBOX}} "+Bbox_mode.title()+"ing Images & Coordinates]")
                for scale in scale_set:
                    for i in range(len(self.img_set)):
                        if Bbox_mode.lower() == 'random_shift':
                            for _ in range(random_time):
                                bbox_img, bbox_coor = pp.apply_bbox(self.img_set[i], self.coor_set[i], 
                                                                    self.joint_is_valid[i], scale, 
                                                                    random_shift=True)
                                bbox_img_set.append(bbox_img)
                                bbox_coor_set.append(bbox_coor)
                                bbox_valid_set.append(self.joint_is_valid[i])
                            
                            """
                            if len(scale_set)>1:
                                bbox_label_set = np.concatenate((bbox_label_set, self.labels.copy()),axis=0)
                                bbox_score_set = np.concatenate((bbox_score_set, self.scores.copy()),axis=0)
                            """
                        else :
                            bbox_img, bbox_coor = pp.apply_bbox(self.img_set[i], self.coor_set[i], self.joint_is_valid[i], scale)
                            bbox_img_set.append(bbox_img)
                            bbox_coor_set.append(bbox_coor)
                            bbox_valid_set.append(self.joint_is_valid[i])
                        pbar.update(1)
                    if len(scale_set)>1 and Bbox_mode.lower() != 'random_shift':
                        bbox_label_set = np.concatenate((bbox_label_set, self.labels.copy()),axis=0)
                        bbox_score_set = np.concatenate((bbox_score_set, self.scores.copy()),axis=0)
                        
            bbox_img_set = np.array(bbox_img_set)
            bbox_coor_set = np.array(bbox_coor_set)
            bbox_valid_set = np.array(bbox_valid_set)
            
            if Bbox_mode.lower() == 'augment':
                self.img_set = np.concatenate((self.img_set, bbox_img_set), axis=0)
                self.coor_set = np.concatenate((self.coor_set, bbox_coor_set), axis=0)
                self.joint_is_valid = np.concatenate((self.joint_is_valid, bbox_valid_set), axis=0)
                self.labels = np.concatenate((self.labels, bbox_label_set),axis=0)
                self.scores = np.concatenate((self.scores, bbox_score_set),axis=0)
                
            elif Bbox_mode.lower() == 'apply':
                self.img_set = bbox_img_set
                self.coor_set = bbox_coor_set
                self.joint_is_valid = bbox_valid_set
                self.labels = bbox_label_set
                self.scores = bbox_score_set
                
        # Shuffle images and labels if 'Shuffle' is True
        if Shuffle :
            shuffled = self._shuffling(self.img_set.copy(), self.coor_set.copy(), self.labels.copy(), 
                                       self.joint_is_valid.copy(), self.scores.copy())
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
            label_set = []
            score_set = []
            
            # make dataset batchs
            for n in range(self.num_batchs):
                if len(self.img_set) % batch_size !=0:
                    img_data.append(self.img_set[n*batch_size:(n+1)*batch_size] if n != self.num_batchs-1 else self.img_set[n*batch_size:])
                    joint_set.append(self.coor_set.reshape(len(self.coor_set),-1)[n*batch_size:(n+1)*batch_size] if n != self.num_batchs-1 else self.coor_set.reshape(len(self.coor_set),-1)[n*batch_size:])
                    valid_set.append(self.joint_is_valid[n*batch_size:(n+1)*batch_size] if n != self.num_batchs-1 else self.joint_is_valid[n*batch_size:])
                    score_set.append(self.scores[n*batch_size:(n+1)*batch_size] if n != self.num_batchs-1 else self.scores[n*batch_size:])
                    label_set.append(self.labels[n*batch_size:(n+1)*batch_size] if n != self.num_batchs-1 else self.labels[n*batch_size:])
                elif len(self.img_set) % batch_size ==0:
                    img_data.append(self.img_set[n*batch_size:(n+1)*batch_size])
                    joint_set.append(self.coor_set.reshape(len(self.coor_set),-1)[n*batch_size:(n+1)*batch_size])
                    valid_set.append(self.joint_is_valid[n*batch_size:(n+1)*batch_size])
                    label_set.append(self.labels[n*batch_size:(n+1)*batch_size])
                    score_set.append(self.scores[n*batch_size:(n+1)*batch_size])

            self.batch_set = {'img':img_data, 'joints':joint_set, 'valid':joint_set, 'labels':label_set, 'scores':score_set}
    
    #----------------------------
    #  function definitions
    #----------------------------
    
    #  Augmentation functions
    
    def _mirroring(self, images, joints, labels, joint_is_valid, scores):
        mirrored_img = np.zeros([images.shape[0], images.shape[1],images.shape[2],3])
        mirrored_coor = np.zeros([joints.shape[0], joints.shape[1],2])
        mirrored_valid = joint_is_valid.reshape(-1,14,2)
        
        with tqdm(total=len(images)) as pbar:
            pbar.set_description("[Mirroring Images & Coordinates]")
            for i, img in enumerate(images):
                mirrored_img[i] = cv2.flip(img, 1)
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
        return {'images':mirrored_img,'joints':mirrored_coor,'valid': joint_is_valid.copy(),
                'labels':labels.copy(),'scores':scores.copy()}
    
    def _shuffling(self, images, joints, labels, joint_is_valid, scores):
        indices=np.random.permutation(len(images))
        return {'images': images[indices],'joints':joints[indices],'valid':joint_is_valid[indices],
                'labels':labels[indices],'scores':scores[indices]}

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
        joints=np.array(pd.read_csv(csv_file,header=None))
        
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
