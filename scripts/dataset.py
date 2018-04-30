import numpy as np
import pandas as pd
import skimage, scipy, glob, os, copy
from skimage import io, transform
from scipy.io import loadmat
from tqdm import tqdm

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
    def __init__(self, csv_file, re_img_size=(227,227), is_valid=False, Rotate=False, Fliplr=False, Shuffle=False):
        joints=pd.read_csv(csv_file,header=None).as_matrix()
        
        self.img_path=list(path for path in joints[:,0])
        self.joint_coors=list(coors for coors in joints[:,1:29])
        self.joint_is_valid=list(is_valid for is_valid in joints[:,29:])
        
        self.img_set=np.zeros([len(self.img_path),re_img_size[0],re_img_size[1],3],dtype=np.float)
        self.coor_set=np.array(self.joint_coors).reshape(len(self.joint_coors),14,2)
        
        with tqdm(total=len(self.img_path)) as pbar_process:
            pbar_process.set_description("[Processing Images & Coordinates]")
            for i, path in enumerate(self.img_path):
                img=skimage.io.imread(path)
                self.img_set[i]=skimage.transform.resize(img,(re_img_size[0],re_img_size[1],3))

                for j in range(len(self.coor_set[i])):
                    if is_valid and bool(self.joint_is_valid[i][j]): self.coor_set[i][j] = [-1,-1]

                    if self.coor_set[i][j][0] == -1: pass
                    else:
                            self.coor_set[i][j][0] = self.coor_set[i][j][0]*(re_img_size[0]/img.shape[1])
                            self.coor_set[i][j][1]=self.coor_set[i][j][1]*(re_img_size[1]/img.shape[0])
                pbar_process.update(1)
 
        if Rotate :
            rotated = _rotation(copy.copy(self.img_set), copy.copy(self.coor_set))
            self.img_set = np.concatenate((self.img_set, rotated['images']), axis=0)
            self.coor_set = np.concatenate((self.coor_set, rotated['joints']), axis=0)
            self.joint_is_valid = np.concatenate((self.joint_is_valid, rotated['valid']), axis=0)
            
        if Fliplr :
            fliplred = _mirroring(copy.copy(self.img_set), copy.copy(self.coor_set))
            self.img_set = np.concatenate((self.img_set, fliplred['images']), axis=0)
            self.coor_set = np.concatenate((self.coor_set, fliplred['joints']), axis=0)
            self.joint_is_valid = np.concatenate((self.joint_is_valid, fliplred['valid']), axis=0)
        
        if Shuffle :
            shuffled = _shuffling(copy.copy(self.img_set), copy.copy(self.coor_set), copy.copy(self.joint_is_valid))
            self.img_set = shuffled['images']
            self.coor_set = shuffled['joints']
            self.joint_is_valid = shuffled['valid']
            
    def _rotation(images, joints, joint_is_valid):
        thetas = np.deg2rad((-30,-20,-10,10,20,30))
        rotated_img = np.zeros([images.shape[0]*len(thetas), images.shape[1],images.shape[2],3])
        rotated_coor = np.zeros([joints.shape[0]*len(thetas), joints.shape[1],2])
        rotated_valid = joint_is_valid
        
        with tqdm(total=len(images)*len(thetas)) as pbar:
            pbar.set_description("[Rotating Images & Coordinates]")
            for i, img in enumerate(images):
                for j, theta in enumerate(thetas):
                    img_rotated = scipy.ndimage.rotate(img, theta)
                    org_center = (np.array(img.shape[:2][::-1])-1)/2
                    rotated_center = (np.array(img_rotated.shape[:2][::-1])-1)/2

                    coor_list = list(np.array(coor)+org_center for coor in joints[i])
                    rotated_coor_list = list((lambda x : (x[0]*np.cos(theta) + x[1]*np.sin(theta) + rotated_center[0],
                                                          -x[0]*np.sin(theta) + x[1]*np.cos(theta) + rotated_center[1])
                                             )(coor) for coor in coor_list)

                    img_rotated = skimage.transform.resize(img_rotated, (re_img_size[0],re_img_size[1],3))

                    rotated_img[(i*len(thetas))+j]=img_rotated
                    rotated_coor[(i*len(thetas))+j]=np.array(rotated_coor_list)*(re_img_size[0]/img_rotated.shape[0])
                    pbar.update(1)
        
        for i in range(len(thetas)-1):
            rotated_valid = np.concatenate((rotated_valid, joint_is_valid), axis=0)
            
        return {'images':rotated_img,'joints':rotated_coor,'valid':rotated_valid}

    def _mirroring(images, joints, joint_is_valid):
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
                    else:
                        mirrored_coor[i][j][0] = (lambda x : x+2*((img.shape[0]/2)-x))(joint[0])
                pbar.update(1)
        return {'images':mirrored_img,'joints':mirrored_coor,'valid':joint_is_valid}
    
    def _shuffling(images, joints, joint_is_valid):
        shuffled_img = np.zeros([images.shape[0], images.shape[1],images.shape[2],3])
        shuffled_coor = np.zeros([joints.shape[0], joints.shape[1],2])
        shuffled_valid = np.zeros([len(joint_is_valid),14])
        
        indices=list(range(len(picpic)))
        np.random.shuffle(indices)
        
        for i, idx in enumerate(indices):
            shuffled_img[i] = images[idx]
            shuffled_coor[i] = joints[idx]
            shuffled_valid[i] = joint_is_valid[idx]
            
        return {'images': shuffled_img,'joints':shuffled_coor,'valid':shuffled_valid}
