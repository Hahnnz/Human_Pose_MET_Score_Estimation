import pandas as pd
import numpy as np
import tensorflow as tf
from copy import copy
from tensorflow.contrib.data import Iterator
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

class create:
    def __init__(self, csv_file, mode, batch_size=None, re_img_size=(227,227),
                 is_valid=False, Rotate=False, Fliplr=False, Shuffle=False):
        if not mode.lower() in {"classification", "regression", "all"}:
            raise ValueError("mode must be given 'classification', 'regression' or 'all'.")

        joints = pd.read_csv(csv_file,header=None).as_matrix()

        self.img_path = np.array(list(path for path in joints[:,0]))
        self.joint_coors = np.array(list(coors for coors in joints[:,1:29]))
        self.joint_is_valid = np.array(list(is_valid for is_valid in joints[:,29:43]))
        self.scores = np.array(list(scores for scores in joints[:,43]))[:,np.newaxis]
        self.labels = np.array(list(labels for labels in joints[:,44]))[:,np.newaxis]

        self.re_img_size = re_img_size # ex [227,227]
        self.batch_size = batch_size
        self.num_classes = np.max(self.labels)+1

        if Shuffle : self._shuffling()

        img_path_tf = convert_to_tensor(self.img_path, dtype=dtypes.string)
        
        self._mode = True if mode.lower()=="classification" else False
        if self._mode:
            self.labels = convert_to_tensor(self.labels[:,0], dtype= dtypes.int32)
            self.data = tf.data.Dataset.from_tensor_slices((img_path_tf, self.labels))
            
        elif not self._mode:
            self.coor_set = convert_to_tensor(self.joint_coors, dtype = dtypes.float32)
            valid_expanded = np.concatenate((self.joint_is_valid[:,:,np.newaxis], copy(self.joint_is_valid[:,:,np.newaxis])),
                                             axis=2).reshape(len(self.joint_is_valid),-1)
            def reverseNum(num):
                return 0 if num==1 else 1
            
            for i in range(len(valid_expanded)):
                valid_expanded[i]=np.array(list((lambda x : map(reverseNum,x))(valid_expanded[i])))
            
            self.valid = convert_to_tensor(valid_expanded, dtype = dtypes.int32)
            self.data = tf.data.Dataset.from_tensor_slices((img_path_tf, self.coor_set, self.valid))

        self.data = self.data.map(self._process)
        self.data = self.data.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        self.iterator = Iterator.from_structure(self.data.output_types, self.data.output_shapes)
        self.next = self.iterator.get_next()

    def _mirroring(self):
        tf.image.flip_left_right(self.data)

    def _shuffling(self):
        indices=np.random.permutation(len(self.img_path))
        self.img_path = self.img_path[indices]
        self.joint_coors = self.joint_coors[indices]
        self.joint_is_valid = self.joint_is_valid[indices]
        self.labels = self.labels[indices]
        self.scores = self.scores[indices]

    def _process(self, filename, target, is_valid=None):
        img = tf.read_file("/var/data/MET2/"+filename)
        img_decoded = tf.image.decode_image(img, channels=3)
        img_resized = tf.image.resize_images(img_decoded, self.re_img_size)
        if self._mode and is_valid==None:
            target = tf.one_hot(target, self.num_classes)
            return img_resized, target
        if is_valid != None:
            return img_resized, target, is_valid
