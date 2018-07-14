# Human Action Estimation ( HAE ) `WORK IN PROGRESS`
Human Action Estimation using MaskRCNN with KeyPointing

## Usage
> Will be updated soon

## Activity Categories
We have total 16 activities to be estimated.

* **Office Activities<br/>**
  `Walking about`, `Writing`, `Reading.seated`, `Typing`,`Filing.seated`, `Filing.stand`<br />
* **Resting<br />**
`Reclining` ,`Seated.quiet`, `Sleeping`, `Standing.relaxed`<br />

 index | Activity | MET Score | Label Number
 ------|----------|-----------|-------------
 **01** | Sleeping | 0.7 | 0
 **02** | Reclining | 0.8 | 1
 **03** | Writing | 1.0 | 2
 **04** | Reading.seated | 1.0 | 3
 **05** | Seated.quiet | 1.0 | 4
 **06** | Typing | 1.1 | 5
 **07** | Standing.Relaxed | 1.2 | 6
 **08** | Filling.Seated | 1.2 | 7
 **09** | Fiiling.Stand | 1.4 | 8
 **10** | Cooking | 1.6 | 9
 **11** | Walking about | 1.7 | 10

## Requirements
- **Python 3**
- **Tensorflow ≥ 1.5.0**
- **Tqdm ≥ 4.19.9**
- **Numpy ≥ 1.14.3**
- **Pandas ≥ 0.22.0**

## References
- [DeepPose implementation on TensorFlow](https://github.com/asanakoy/deeppose_tf)
