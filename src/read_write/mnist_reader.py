from tensorflow.examples.tutorials.mnist import input_data

DATASET_DIR = 'D:/faks/diplomski/lip-reading/data/datasets/mnist'
INPUT_SHAPE = [28, 28, 1]

class MnistReader():

    def __init__(self):

        self.initConfig()
        self.read_mnist()

    def initConfig(self):

        self.name = 'mnist'
        self.frames = 1
        self.h = INPUT_SHAPE[0]
        self.w = INPUT_SHAPE[1]
        self.c = INPUT_SHAPE[2]
        self.num_classes = 10

    def read_mnist(self):

        dataset = input_data.read_data_sets(DATASET_DIR, one_hot=True)

        train_x = dataset.train.images
        valid_x = dataset.validation.images
        test_x = dataset.test.images

        self.train_images = train_x.reshape([-1, self.h, self.w, self.c])
        self.valid_images = valid_x.reshape([-1, self.h, self.w, self.c])
        self.test_images = test_x.reshape([-1 ,self.h, self.w, self.c])

        self.train_labels = dataset.train.labels
        self.valid_labels = dataset.validation.labels
        self.test_labels = dataset.test.labels

        self.num_train_examples = dataset.train.num_examples
        self.num_valid_examples = dataset.validation.num_examples
        self.num_test_examples = dataset.test.num_examples

    def getNumOfExamples(self, datasetType):

        if datasetType == 'train':
            return self.num_train_examples
        if datasetType == 'test':
            return self.num_test_examples
        if datasetType == 'val':
            return self.num_valid_examples

    def getData(self, datasetType):

        if datasetType == 'train':
            return self.train_images, self.train_labels
        if datasetType == 'test':
            return self.test_images, self.test_labels
        if datasetType == 'val':
            return self.valid_images, self.valid_labels