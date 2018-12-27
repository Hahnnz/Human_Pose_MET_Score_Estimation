import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

joints = ['Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip', 'Left Knee', 'Left Ankle',
          'Right Wrist', 'Right Elbow', 'Right Shoulder', 'Left Shoulder', 'Left Elbow', 'Left Wrist',
          'Neck', 'Head']

classes = ['Sleeping','Reclining','Seated.quiet','Standing.Relaxed','Reading.seated',
              'Writing', 'Typing', 'Filing.Seated','Filing.Stand','Walking about']
class_color = ['red','orange','green','blue','purple',
               'aqua','brown','navy','magenta','lightseagreen']

sticks = ['Head','Torso','U Arm','L Arm','U Leg','L Leg','Mean']
sticks_color = ['red','green','blue','yellow','pink','orange','black']