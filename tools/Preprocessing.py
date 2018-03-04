import numpy as np
import pandas as pd
import cv2

dataset_root="/var/data"
csv_file = dataset_root+"/MET/dataset.csv"

def config(csv):
  return pd.read_csv(csv,header=None).as_matrix()

class prepare_dataset:

  def check_sizes(dataset):
    allsum_x = allsum_y = max_x = max_y = 0

    for i in range(len(dataset)):
      img=cv2.imread(dataset[i][0])
      if max_x<img.shape[0]: max_x=img.shape[0]
      if max_y<img.shape[1]: max_y=img.shape[1]  
      allsum_x+=img.shape[0]
      allsum_y+=img.shape[1]

    avg_x=allsum_x/len(dataset)
    avg_y=allsum_y/len(dataset)

    return "Arverage X : "+avg_x+"\nArverage Y : "+avg_y

  # if coos is "[00, 00]",
  def convert_str_to_int(dataset):
    coors=np.zeros([len(dataset),14,2],dtype=int)
    for i in range(len(dataset)):
      for j in range(14):
        a=0
        while True:
          if dataset[i][j+1][a]==",": break
          else : a+=1
          coors[i][j]=int(dataset[i][j+1][1:a]),int(dataset[i][j+1][a:1:-1]) 
    return coors
  
  def prepare(dataset):
  	img_set = np.zeros([len(dataset),256,256,3])
  	coor_set = convert_str_to_int(dataset)

  	for i in range(len(dataset)):
  	  img=cv2.imread(dataset[i][0])

  	  img_set[i]=cv2.resize(img,(256,256))

  	  for j in range(14):
  	    if coor_set[i][j][0]==-1 : coor_set[i][j][0]=-1
  	    else : coor_set[i][j][0]=coor_set[i][j][0]*(256/img.shape[1])
  	    if coor_set[i][j][1]==-1 : coor_set[i][j][1]=-1
  	    else : coor_set[i][j][1]=coor_set[i][j][1]*(256/img.shape[0])
  	return img_set, coor_set