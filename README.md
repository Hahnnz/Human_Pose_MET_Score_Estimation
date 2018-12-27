# Human Activities & Pose Estimation `WORK IN PROGRESS`
Human Activities Estimation

## Usage
> Will be updated soon


## Activity Categories
We have total 10 activities to be estimated.

* **Office Activities<br/>**
  `Walking about`, `Writing`, `Reading.seated`, `Typing`,`Filing.seated`, `Filing.stand`<br />
* **Resting<br />**
`Reclining` ,`Seated.quiet`, `Sleeping`, `Standing.relaxed`<br />

 index | Activity | MET Score | Label Number
 ------|----------|-----------|-------------
 **01** | Sleeping | 0.7 | 0
 **02** | Reclining | 0.8 | 1
 **03** | Seated.Quiet | 1.0 | 2
 **04** | Standing.Relexed | 1.2 | 3
 **05** | Reading.Seated | 1.0 | 4
 **06** | Writing | 1.0 | 5
 **07** | Typing | 1.1 | 6
 **08** | Filing.Seated | 1.2 | 7
 **09** | Filing.Stand | 1.4 | 8
 **10** | Walking About | 1.7 | 9

## Current Test mPCP@0.5
 **Body Parts** | **mPCP@0.5** | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 
 :--------: | :------: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
 Head | 0.430952 |  |  |  |  |  |  |  |  |  |  |
 Torso | 0.795238 |  |  |  |  |  |  |  |  |  |  |
 U Arm | 0.483333 |  |  |  |  |  |  |  |  |  |  |
 L Arm | 0.398810 |  |  |  |  |  |  |  |  |  |  |
 U Leg | 0.492857 |  |  |  |  |  |  |  |  |  |  |
 L Leg | 0.536905 |  |  |  |  |  |  |  |  |  |  |
 **MEAN** | **0.523016** |  |  |  |  |  |  |  |  |  |  |

## Requirements
- **Python 3**
- **Tensorflow ≥ 1.5.0**
- **Tqdm ≥ 4.19.9**
- **Numpy ≥ 1.14.3**
- **Pandas ≥ 0.22.0**

## Acknowledgments

This research was supported by a grant (code 18CTAP-C129762-02) from Infrastructure and Transportation Technology Promotion Research Program funded by Ministry of Land, Infrastructure and Transport of Korean government.

## References
- [DeepPose implementation on TensorFlow](https://github.com/asanakoy/deeppose_tf)
