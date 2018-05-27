import numpy as np
import os
from src.utils import util

CIFAR_DATASET_DIR = 'D:/faks/diplomski/lip-reading/data/datasets/cifar'

class CifarReader():

    def __init__(self):

        self.name = 'çifar'
        self.__init_datasets__()

    def __init_datasets__(self):

        width = 32
        height = 32
        channel = 3

        self.train_x = np.ndarray((0, width * height * channel), dtype=np.float32)
        self.train_y = np.ndarray((0, ))

        subset = util.unpickle(os.path.join(CIFAR_DATASET_DIR, 'train'))
        self.train_x = np.vstack((self.train_x, subset['data']))
        self.train_y = np.concatenate((self.train_y, subset['fine_labels']), axis=0)

        self.train_x = self.train_x.reshape((-1, channel, height, width)).transpose(0, 2, 3, 1)
        self.train_y = np.array(self.train_y, dtype=np.uint8)

        subset = util.unpickle(os.path.join(CIFAR_DATASET_DIR, 'test'))
        self.test_x = subset['data'].reshape((-1, channel, height, width)).transpose(0, 2, 3, 1).astype(np.uint8)
        self.test_y = np.array(subset['fine_labels'], dtype=np.uint8)

        valid_size = 5000
        self.train_x, self.train_y = util.shuffle_data(self.train_x, self.train_y)

        self.valid_x = self.train_x[:valid_size, ...]
        self.valid_y = self.train_y[:valid_size, ...]

        self.train_x = self.train_x[valid_size:, ...]
        self.train_y = self.train_y[valid_size:, ...]

        self.num_train_examples = len(self.train_x)
        self.num_val_examples = len(self.valid_x)
        self.num_test_examples = len(self.test_x)

    def getNumOfExamples(self, datasetType):

        if datasetType == 'train':
            return self.num_train_examples
        elif datasetType == 'test':
            return self.num_test_examples
        elif datasetType == 'val':
            return self.num_val_examples
        else:
            return 0

    def getData(self, datasetType):

        if datasetType == 'train':
            return self.train_x, self.train_y
        elif datasetType == 'test':
            return self.test_x, self.test_y
        elif datasetType == 'val':
            return self.valid_x, self.valid_y
        else:
            return ([], [])