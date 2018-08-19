import numpy as np
import cv2

def get_bbox_coor(img, coords, valid):
    # copy variables
    img = img.copy()
    coords = coords.copy()
    valid = valid[::2].copy()
    
    # check whether joint is valid or not
    valid_joint = []
    for i in range(len(valid)):
        if valid[i]>0 :
            if coords[i,0]>0 and coords[i,1]>0:
                valid_joint.append(coords[i])
    valid_joint = np.array(valid_joint)
    
    # get coords
    x_axis, y_axis = valid_joint.transpose(1,0)
    
    # get tight bbox coordinates
    tight_bbox_coord = np.array([min(y_axis),max(y_axis), min(x_axis),max(x_axis)])
    
    return list(map(int,tight_bbox_coord))

def apply_bbox(img, coords, valid, scale=0., resize=True, random_shift=False):
    
    # check a given scale value is between 0.0 to 5.0
    scale_range = np.array(list(map(str,np.linspace(0.,5.,51))))
    scale_range = list(map((lambda x : float(x[:3])),scale_range))
    if np.array(scale) in scale_range:
        scale = int(scale*10)
    else :
        raise ValueError('scale should be in range 0.0 to 5.0')
    
    # copy variables
    img = img.copy()
    coords = coords.copy()
    valid = valid[::2].copy()
    
    # check whether joint is valid or not
    valid_joint = []
    for i in range(len(valid)):
        if valid[i]>0 :
            if coords[i,0]>0 and coords[i,1]>0:
                valid_joint.append(coords[i])
    valid_joint = np.array(valid_joint)
    
    # get coords
    x_axis, y_axis = valid_joint.transpose(1,0)
    
    # get tight bbox coordinates
    tight_bbox_coord = np.array([min(y_axis),max(y_axis), min(x_axis),max(x_axis)])
    
    # set scale bias values and scale the size
    gaps = np.array([0,img.shape[1],0,img.shape[0]]) - tight_bbox_coord
    gap_units = gaps*scale/50
    scaled_bbox_coord = list(map(int,(tight_bbox_coord + gap_units)))

    if random_shift :
        area = 0 if gap_units[1]*gap_units[3] >= gap_units[0]*gap_units[2] else 1
        if area == 0 and len(range(scaled_bbox_coord[0], int(tight_bbox_coord[0]))) !=0 and len(range(scaled_bbox_coord[2], int(tight_bbox_coord[2]))) !=0:
            chosen_x = np.random.choice(range(scaled_bbox_coord[0],
                                              int(tight_bbox_coord[0])),1)
            chosen_y = np.random.choice(range(int(scaled_bbox_coord[2]),
                                              int(tight_bbox_coord[2])),1)
            
            gap_by_shift = (abs(tight_bbox_coord[0]-chosen_x),abs(tight_bbox_coord[2]-chosen_y))
            
            scaled_bbox_coord = list(map(int,([chosen_x, scaled_bbox_coord[1]-gap_by_shift[0],
                                               chosen_y, scaled_bbox_coord[3]-gap_by_shift[1]])))
        elif area == 1 and len(range(scaled_bbox_coord[1], int(tight_bbox_coord[1]))) !=0 and len(range(scaled_bbox_coord[3], int(tight_bbox_coord[3]))) !=0:
            chosen_x = np.random.choice(range(int(tight_bbox_coord[1]),scaled_bbox_coord[1]), 1)
            chosen_y = np.random.choice(range(int(tight_bbox_coord[3]),scaled_bbox_coord[3]), 1)
            
            gap_by_shift = (abs(tight_bbox_coord[1]-chosen_x),abs(tight_bbox_coord[3]-chosen_y))
            
            scaled_bbox_coord = list(map(int,([scaled_bbox_coord[0]+gap_by_shift[0], chosen_x,
                                               scaled_bbox_coord[2]+gap_by_shift[1], chosen_y])))

    # Crop the bbox
    bbox_img = img[scaled_bbox_coord[0]:scaled_bbox_coord[1],
                  scaled_bbox_coord[2]:scaled_bbox_coord[3]]
    
    # Compute bbox joints coordinates
    for i in range(len(coords)):
        if valid[i]==0 :
            if coords[i,0]<=0 and coords[i,1]<=0:
                coords[i]=[-1,-1]
        else:
            coords[i]-=np.array([scaled_bbox_coord[2],scaled_bbox_coord[0]])
    
    # resize bbox image and joints if arg 'resize' is True
    if resize :
        for j in range(len(coords)):
            if valid[j]==0 : pass
            else :
                coords[j,0] = coords[j,0] * (img.shape[0]/bbox_img.shape[1])
                coords[j,1] = coords[j,1] * (img.shape[1]/bbox_img.shape[0])
        bbox_img = cv2.resize(bbox_img, img.shape[:2])
    
    return bbox_img, coords