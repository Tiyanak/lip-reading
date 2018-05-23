from utils import config
from tensorflow.examples.tutorials.mnist import input_data

DATASET_DIR = 'D:/faks/diplomski/lip-reading/data/datasets/mnist'
INPUT_SHAPE = [28, 28, 1]

class MnistDataset:

    def __init__(self):

        self.initConfig()
        self.read_mnist()

    def initConfig(self):

        self.name = 'mnist'
        self.batch_size = config.config['batch_size']
        self.frames = 1
        self.h = INPUT_SHAPE[0]
        self.w = INPUT_SHAPE[1]
        self.c = INPUT_SHAPE[2]
        self.num_classes = 10
        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

    def read_mnist(self):

        dataset = input_data.read_data_sets(DATASET_DIR, one_hot=True)

        train_x = dataset.train.images
        valid_x = dataset.validation.images
        test_x = dataset.test.images

        self.train_images = train_x.reshape([-1, self.h, self.w, self.c])
        self.valid_images = valid_x.reshape([-1, self.h, self.w, self.c])
        self.test_images = test_x.reshape([-1,self.h, self.w, self.c])

        self.train_labels = dataset.train.labels
        self.valid_labels = dataset.validation.labels
        self.test_labels = dataset.test.labels

        self.num_train_examples = dataset.train.num_examples
        self.num_valid_examples = dataset.validation.num_examples
        self.num_test_examples = dataset.test.num_examples

        self.num_batches_train = self.num_train_examples // self.batch_size
        self.num_batches_valid = self.num_valid_examples // self.batch_size
        self.num_batches_test = self.num_test_examples // self.batch_size

    def get_batch(self, datasettype='train'):

        if datasettype == 'val':
            return self.get_valid_batch()
        elif datasettype == 'test':
            return self.get_test_batch()
        else:
            return self.get_train_batch()

    def get_train_batch(self):
        batch_x = self.train_images[self.train_index:self.train_index + self.batch_size]
        batch_y = self.train_labels[self.train_index:self.train_index + self.batch_size]
        self.train_index += self.batch_size
        if self.train_index >= self.num_train_examples:
            self.train_index = 0
        return (batch_x, batch_y)

    def get_valid_batch(self):
        batch_x = self.valid_images[self.val_index:self.val_index + self.batch_size]
        batch_y = self.valid_labels[self.val_index:self.val_index + self.batch_size]
        self.val_index += self.batch_size
        if self.val_index >= self.num_valid_examples:
            self.val_index = 0
        return (batch_x, batch_y)

    def get_test_batch(self):
        batch_x = self.test_images[self.test_index:self.test_index + self.batch_size]
        batch_y = self.test_labels[self.test_index:self.test_index + self.batch_size]
        self.test_index += self.batch_size
        if self.test_index >= self.num_test_examples:
            self.test_index = 0
        return (batch_x, batch_y)