import cv2
import os

def markJoints(img, joints):
  circleSize=10
  font=cv2.FONT_HERSHEY_SIMPLEX

  for i in range(len(joints)):
  	x,y = joints[i]
  	cv2.circle(img, (x,y), 4, (255,0, 0), thickness=-1)
  	cv2.putText(img, str(i+1), (x,y), font, 0.5, (100,100,100), 1, cv2.LINE_AA)

  return img 

def set_GPU(device_num):
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
