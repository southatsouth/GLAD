#-*- encoding: utf-8 -*_

import sys
import caffe
import numpy as np
import os
import os.path as osp
import random
import cv2

class id2_data_layer(caffe.Layer):
    """
    这个python的data layer用于动态的构造训练deepID2的数据
    每次forward会产生多对数据，每对数据可能是相同的label或者不同的label
    """
    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # 读取输入的参数
        params = eval(self.param_str)
        print "init data layer"
        print params

        self.batch_size = params['batch_size']  # batch_size
        self.ratio = float(params['ratio'])
        self.scale = float(params['scale'])
        assert self.batch_size > 0 and self.batch_size % 2 == 0, "batch size must be 2X"
        assert self.ratio > 0 and self.ratio < 1, "ratio must be in (0, 1)"
        self.image_root_dir = params['image_root_dir']
        self.mean_file = params['mean_file']       
        self.source = params['source']
        self.crop_size = params['crop_size']

        top[0].reshape(self.batch_size, 3, params['crop_size'], params['crop_size'])
        top[1].reshape(self.batch_size, 1)
        self.batch_loader = BatchLoader(self.image_root_dir, self.mean_file, self.scale, self.source, self.batch_size, self.ratio)

    def forward(self, bottom, top):
        blob, label_list = self.batch_loader.get_mini_batch()
        top[0].data[...] = blob
        top[1].data[...] = label_list

    def backward(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        pass

class BatchLoader(object):

    def __init__(self, root_dir, mean_file, scale, image_list_path, batch_size, ratio):
        print "init batch loader"
        self.batch_size = batch_size
        self.ratio = ratio # true pair / false pair
        self.image2label = {}   # key:image_name    value:label
        self.label2images = {}  # key:label         value: image_name array
        self.images = []        # store all image_name
        self.mean = np.load(mean_file)
        self.scale = scale
        self.root_dir = root_dir
        with open(image_list_path) as fp:
            for line in fp:
                data = line.strip().split()
                image_name = data[0]
                label = data[-1]
                self.images.append(image_name)
                self.image2label[image_name] = label
                if label not in self.label2images:
                    self.label2images[label] = []
                self.label2images[label].append(image_name)
        self.labels = self.label2images.keys()
        self.label_num = len(self.labels)
        self.image_num = len(self.image2label)
        print "init batch loader over"

    def get_mini_batch(self):
        image_list, label_list = self._get_batch(self.batch_size / 2)
        cv_image_list = map(lambda image_name: (self.scale * (cv2.imread(os.path.join(self.root_dir, image_name)).astype(np.float32, copy=False).transpose((2, 0, 1)) - self.mean)), image_list)
        blob = np.require(cv_image_list)
        label_blob = np.require(label_list, dtype=np.float32).reshape((self.batch_size, 1))
        return blob, label_blob
    
    def _get_batch(self, pair_num):
        image_list = []
        label_list = []
        for pair_idx in xrange(pair_num):
            if random.random() < self.ratio: # true pair
                while True:
                    label_idx = random.randint(0, self.label_num - 1)
                    label = self.labels[label_idx]
                    if len(self.label2images[label]) > 5:
                        break
                first_idx = random.randint(0, len(self.label2images[label]) - 1)
                second_id = random.randint(0, len(self.label2images[label]) - 2)
                if second_id >= first_idx:
                    second_id += 1
                image_list.append(self.label2images[label][first_idx])
                image_list.append(self.label2images[label][second_id])
                label_list.append(int(label))
                label_list.append(int(label))
            else:                   # false pair
                for i in xrange(2):
                    image_id = random.randint(0, self.image_num - 1)
                    image_name = self.images[image_id]
                    label = self.image2label[image_name]
                    image_list.append(image_name)
                    label_list.append(int(label))
        return image_list, label_list