# Human Movement Estimation ( HME ) `WORK IN PROGRESS`
Human Movement Estimation using MaskRCNN with KeyPointing

## Todo
- [X] Preparing needed
- [ ] Combine **[Mask-RCNN](https://github.com/matterport/Mask_RCNN)** & **[DeepPose](https://github.com/ys7yoo/deeppose)** Algorithms
  - [X] Test Mask RCNN   
  - [X] Test DeepPose      
  - [ ] Combine them
    - [X] Change DeepPose Training Model (Alexnet to ResNet) 
    - [ ] Combine DeepPose-Resnet with Mask-RCNN
  - [ ] improve Combined Algorithm
  - [ ] Predict MET Score Based on Calculating how mach it affect to humans' pose

## Activity Categories
We have total 16 activities to be estimated.

* **Office Activities<br/>**
  `Walking about`, `Writing, Reading.seated`, `Typing`, `Lifting.packing - lifting`, `Lifting.packing - packing`, `Filling.seated`, `Filling.stand`<br />
* **Miscellaneous Occupational Activity<br />**
`Cooking`, `House cleaning`, `Machine work.light`, `Machine work.sawing`<br />
* **Resting<br />**
`Reclining` ,`Seated.quiet`, `Sleeping`<br />

Activity | MET Score | Label Number
---------|-----------|-------------
Walking about | 1.7 | 0
Writing | 1.0 | 1
Reading.seated | 1.0 | 2
Typing | 1.1 | 3
Lifting.packing - lifting | 2.1 | 4
Lifting.packing - packing | 2.1 | 5
Filling.seated | 1.2 | 6
Filling.stand | 1.4 | 7
Cooking | 1.6 | 8
House cleaning | 2.0 | 9
Machine work.light | 2.0 | 10
Machine work.sawing | 1.8 | 11
Reclining | 0.8 | 12
Seated.quiet | 1.0 | 13
Sleeping | 0.7 | 14
Standing.relaxed | 1.2 | 15
