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

 index | Activity | MET Score | Label Number
 ------|----------|-----------|-------------
 **01** | Sleeping | 0.7 | 0
 **02** |Reclining | 0.8 | 1
 **03** |Writing | 1.0 | 2
 **04** |Reading.seated | 1.0 | 3
 **05** |Seated.quiet | 1.0 | 4
 **06** |Typing | 1.1 | 5
 **07** |Standing.Relaxed | 1.2 | 6
 **08** |Filling.Seated | 1.2 | 7
 **09** |Fiiling.Stand | 1.4 | 8
 **10** |Cooking | 1.6 | 9
 **11** |Walking about | 1.7 | 10
 **12** |Machine Work.Sawing | 1.8 | 11
 **13** |House Cleaning | 2.0 | 12
 **14** |Machine Work.light | 2.0 | 13
 **15** |Lifting | 2.1 | 14
 **16** |Packing | 2.1 | 15
