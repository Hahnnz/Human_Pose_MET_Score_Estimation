from scipy.io import loadmat
import numpy as np
import skimage
from skimage import io, transform
import glob, os

# For lsp dataset...
def create(DATASET_ROOT="./dataset/", reshape_size=[400,400],active_joint_absence=False):
    joints = loadmat(DATASET_ROOT+"joints.mat")
    
    if active_joint_absence :
        joints = joints["joints"].transpose(2,1,0)[:, :, :]
        for i in range(joints.shape[0]):
            for j in range(joints.shape[1]):
                if joints[i][j][2] == 1: joints[i][j][:1]=[-1, -1]
        joints_set = joints[:,:,:2]
    else :
        joints_set = joints["joints"].transpose(2,1,0)[:, :, :2]
    
    i=0
    img_set = np.zeros([len(joints_set),reshape_size[0],reshape_size[1],3])
    for img_path in sorted(glob.glob(os.path.join(DATASET_ROOT+"images/", '*.jpg'))):
        img = skimage.io.imread(img_path)
        img_set[i]=skimage.transform.resize(img,reshape_size)
        for j in range(14):
                if active_joint_absence and joints_set[i][j][0]==-1: joints_set[i][j][0]=-1
                else: joints_set[i][j][0]=joints_set[i][j][0]*(reshape_size[0]/img.shape[1])
                if active_joint_absence and joints_set[i][j][1]==-1: joints_set[i][j][1]=-1
                else: joints_set[i][j][1]=joints_set[i][j][1]*(reshape_size[1]/img.shape[0])
        i+=1
    return {'images':img_set, 'joints':joints_set}