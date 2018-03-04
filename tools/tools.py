import cv2


def markJoints(img, joints):
  circleSize=10
  font=cv2.FONT_HERSHEY_SIMPLEX

  for i in range(len(joints)):
  	x,y = joints[i]
  	cv2.circle(img, (x,y), 4, (255,0, 0), thickness=-1)
  	cv2.putText(img, str(i+1), (x,y), font, 0.5, (100,100,100), 1, cv2.LINE_AA)

  return img 