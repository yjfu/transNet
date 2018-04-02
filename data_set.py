from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import struct
import tensorflow as tf


print tf.__version__

class DataSet:
    def __init__(self, train_path_file, batch_size):
        self.batch_size = batch_size
        self.train_file = train_path_file
        # read train file list
        file = open(train_path_file, "r")
        train_paths = file.readlines()
        self.train_rgb_paths = []
        self.train_segment_paths = []
        self.train_flow_paths = []
        self.train_label_paths = []
        for line in train_paths:
            _path = line.split()
            self.train_rgb_paths.append(_path[0])
            self.train_segment_paths.append(_path[1])
            self.train_flow_paths.append(_path[2])
            self.train_label_paths.append(_path[3])

        # read train_data, which is a array shaped (height, width, 6)
        self.train_datas = []
        for i in range(self.train_rgb_paths.__len__()):
            # we can also use the function named imread from matplotlib.image
            # which will automatically change the img from 0~255 to 0~1
            _rgb = Image.open(self.train_rgb_paths[i])
            _rgb = np.array(_rgb, dtype=np.float32)/255
            _segment = Image.open(self.train_segment_paths[i])
            _segment = np.array(_segment.split()[0], dtype=np.uint8)/255
            _segment = np.expand_dims(_segment, axis=2)
            _flow = self.parse_flo_file(self.train_flow_paths[i])
            _data = np.concatenate((_rgb, _segment, _flow), axis=2)
            self.train_datas.append(_data)

        # read train_label
        self.train_labels = []
        for label_path in self.train_label_paths:
            _label = Image.open(label_path)
            _label = _label.split()[0]
            _label = np.array(_label, dtype=np.uint8)/255
            _label = np.expand_dims(_label, axis=2)
            self.train_labels.append(_label)
        # init parameters
        self.train_ptr = 0                                  # point to next index of the data to train
        self.train_data_num = len(self.train_datas)          # number of data for train
        self.train_index = np.arange(self.train_data_num)   # index of data in self.train_data
        np.random.shuffle(self.train_index)                 # we access data by this order


    def next_batch(self):
        data = []
        label = []
        if self.train_data_num > self.train_ptr+self.batch_size:
            index = self.train_index[self.train_ptr:self.train_ptr+self.batch_size]
            self.train_ptr = self.train_ptr+self.batch_size
            data = [self.train_datas[i] for i in index]
            label = [self.train_labels[i] for i in index]
            print index
        else:
            print "change"
            index = self.train_index[self.train_ptr:]
            print "index1", index
            self.train_ptr = (self.train_ptr+self.batch_size)-self.train_data_num
            data = [self.train_datas[i] for i in index]
            label = [self.train_labels[i] for i in index]
            np.random.shuffle(self.train_index)
            index = self.train_index[:self.train_ptr]
            print "index2", index
            data += [self.train_datas[i] for i in index]
            label += [self.train_labels[i] for i in index]
            print
        return data, label


    @staticmethod
    def parse_fflo_file(file_path):
        fflo_file = open(file_path, "rb")
        rows_num, cols_num = struct.unpack("2i", fflo_file.read(8))

        pix_num = rows_num*cols_num
        horizontal_channel = struct.unpack("%df" % pix_num, fflo_file.read(pix_num*4))
        vertical_channel = struct.unpack("%df" % pix_num, fflo_file.read(pix_num * 4))
        fflo_data = np.array([horizontal_channel, vertical_channel], np.float32)

        return np.reshape(fflo_data, (2, rows_num, cols_num))

    @staticmethod
    def parse_flo_file(file_path):
        flo_file = open(file_path, "rb")
        _, cols_num, rows_num = struct.unpack("4s2i", flo_file.read(12))

        pix_num = rows_num * cols_num
        data = struct.unpack("%df" % pix_num*2, flo_file.read(pix_num * 4 * 2))
        data = np.array(data, np.float32)

        return np.reshape(data, (rows_num, cols_num, 2))


d = DataSet("data_path.txt", 16)
for i in range(10):
    data, label = d.next_batch()




