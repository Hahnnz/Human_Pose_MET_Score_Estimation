import cv2
import os
import skimage
from skimage import draw

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
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
