import numpy as np
import pandas as pd
import skimage, scipy, glob, os, copy
from skimage import io, transform
from scipy.io import loadmat

class lsp:
    def __init__(self):
        self.trainset, self.testset = __create(DATASET_ROOT="./dataset/", reshape_size=[227,227])
    
    def __create(DATASET_ROOT="./dataset/", reshape_size=[400,400],active_joint_absence=False, divide_train_test=True):
        joints = loadmat(DATASET_ROOT+"joints.mat")

        if active_joint_absence :
            joints = joints["joints"].transpose(2,1,0)[:, :, :]
            for i in range(joints.shape[0]):
                for j in range(joints.shape[1]):
                    if joints[i][j][2] == 1: joints[i][j][:1]=[-1, -1]
            joints_set = joints[:,:,:2]
        else :
            joints_set = joints["joints"].transpose(2,1,0)[:, :, :2]

        img_set = np.zeros([len(joints_set),reshape_size[0],reshape_size[1],3])
        for i, img_path in enumerate(sorted(glob.glob(os.path.join(DATASET_ROOT+"images/", '*.jpg')))):
            img = skimage.io.imread(img_path)
            img_set[i]=skimage.transform.resize(img,reshape_size)
            for j in range(14):
                    if active_joint_absence and joints_set[i][j][0]==-1: joints_set[i][j][0]=-1
                    else: joints_set[i][j][0]=joints_set[i][j][0]*(reshape_size[0]/img.shape[1])
                    if active_joint_absence and joints_set[i][j][1]==-1: joints_set[i][j][1]=-1
                    else: joints_set[i][j][1]=joints_set[i][j][1]*(reshape_size[1]/img.shape[0])

        if divide_train_test:
            return {'images':img_set[:1800], 'joints':joints_set[:1800]}, {'images':img_set[1800:], 'joints':joints_set[1800:]} 
        elif not divide_train_test:
            return {'images':img_set, 'joints':joints_set}

    def feed_batch(self, batch_size, num):
        return copy.copy(self["images"][num*batch_size:(num+1)*batch_size]), copy.copy(self["joints"][num*batch_size:(num+1)*batch_size])

class met:
    def __init__(self, csv_file, re_img_size=(227,227),Rotate=False, Fliplr=False, Shift=False, Crop=False):
        joints=pd.read_csv(csv_file,header=None).as_matrix()
        
        self.img_path=list(path for path in joints[:,0])
        self.joint_coors=list(coors for coors in joints[:,1:29])
        self.joint_is_valid=list(is_valid for is_valid in joints[:,29:])
        
        self.img_set=np.zeros([len(self.img_path),re_img_size[0],re_img_size[1],3],dtype=np.float)
        self.coor_set=np.array(self.joint_coors).reshape(len(self.joint_coors),14,2)
        
        for i, path in enumerate(self.img_path):
            img=skimage.io.imread(path)
            self.img_set[i]=skimage.transform.resize(img,(re_img_size[0],re_img_size[1],3))

            for j in range(len(self.joint_coors)):
                if self.is_valid and bool(self.joint_is_valid[i][j]): self.coor_set[i][j] = [-1,-1]

                if self.coor_set[i][j][0]==-1: pass
                else: self.coor_set[i][j][0]=self.coor_set[i][j][0]*(re_img_size[0]/img.shape[1])

                if self.coor_set[i][j][1]==-1: pass
                else: self.coor_set[i][j][1]=self.coor_set[i][j][1]*(re_img_size[1]/img.shape[0])
        
        if Rotate :
            rotated = _rotation(copy.copy(self.img_set), copy.copy(self.coor_set))
            self.img_set = np.concatenate((self.img_set, rotated['images']), axis=0)
            self.coor_set = np.concatenate((self.coor_set, rotated['joints']), axis=0)
            
        if Fliplr :
            fliplred = _mirroring(copy.copy(self.img_set), copy.copy(self.coor_set))
            self.img_set = np.concatenate((self.img_set, fliplred['images']), axis=0)
            self.coor_set = np.concatenate((self.coor_set, fliplred['joints']), axis=0)
            
        if Shift :
            shifted = _shifting(copy.copy(self.img_set), copy.copy(self.coor_set))
            self.img_set = np.concatenate((self.img_set, shifted['images']), axis=0)
            self.coor_set = np.concatenate((self.coor_set, shifted['joints']), axis=0)
            
        if Crop :
            cropped = _croppping(copy.copy(self.img_set), copy.copy(self.coor_set))
            self.img_set = np.concatenate((self.img_set, cropped['images']), axis=0)
            self.coor_set = np.concatenate((self.coor_set, cropped['joints']), axis=0)
            
    def _rotation(images, joints):
        # Will be updated
        return {'images':rotated_img,'joints':rotated_coor}
    
    def _mirroring(img, joints):
        # Will be updated
        return {'images':mirrored_img,'joints':mirrored_coor}
    
    def _shifting(img, joints):
        # Will be updated
        return {'images':shifted_img,'joints':shifted_coor}
    
    def _shuffling(img, joints):
        # Will be updated
        return {'images':,'joints':}