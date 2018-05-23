import os
import math
import tensorflow as tf
import numpy as np
from utils import config, util

DATASET_DIR = 'D:/faks/diplomski/lip-reading/data/tfrecords/lrw_tfrecords'
INPUT_SHAPE = [29, 112, 112, 3]

class LrwDataset:

    def __init__(self, is_training=True):

        self._initConfig()

        shapes = [INPUT_SHAPE, []]

        self.train_tfrecords = self.getTfRecordFiles('train')
        self.valid_tfrecords = self.getTfRecordFiles('val')
        self.test_tfrecords = self.getTfRecordFiles('test')

        self.num_train_examples = self.numberOfExamples('train')
        self.num_valid_examples = self.numberOfExamples('val')
        self.num_test_examples = self.numberOfExamples('test')

        self.num_batches_train = self.num_train_examples // self.batch_size
        self.num_batches_valid = self.num_valid_examples // self.batch_size
        self.num_batches_test = self.num_test_examples // self.batch_size

        train_file_queue = tf.train.string_input_producer(self.train_tfrecords, capacity=len(self.train_tfrecords))
        valid_file_queue = tf.train.string_input_producer(self.valid_tfrecords, capacity=len(self.valid_tfrecords))
        test_file_queue = tf.train.string_input_producer(self.test_tfrecords, capacity=len(self.test_tfrecords))

        train_images, train_labels = self.input_decoder(train_file_queue, 'train')
        valid_images, valid_labels = self.input_decoder(valid_file_queue, 'val')
        test_images, test_labels = self.input_decoder(test_file_queue, 'test')

        if is_training:
            self.train_images, self.train_labels = tf.train.shuffle_batch(
                [train_images, train_labels], batch_size=self.batch_size, shapes=shapes, capacity=100,
                min_after_dequeue=50)
        else:
            self.train_images, self.train_labels = tf.train.batch(
                [train_images, train_labels], batch_size=self.batch_size, shapes=shapes)

        self.valid_images, self.valid_labels = tf.train.batch(
            [valid_images, valid_labels], batch_size=self.batch_size, shapes=shapes)

        self.test_images, self.test_labels = tf.train.batch(
            [test_images, test_labels], batch_size=self.batch_size, shapes=shapes)

    def _initConfig(self):

        self.name = 'lrw'
        util.create_dir(DATASET_DIR)
        self.batch_size = config.config['batch_size']
        self.frames = INPUT_SHAPE[0]
        self.h = INPUT_SHAPE[1]
        self.w = INPUT_SHAPE[2]
        self.c = INPUT_SHAPE[3]
        self.num_classes = 500

    def getTfRecordFiles(self, datasettype):

        dataDir = os.path.join(DATASET_DIR, datasettype)
        tfRecordsList = []

        if not os.path.exists(dataDir):
            return tfRecordsList

        for file in os.listdir(dataDir):
            filename = os.fsdecode(file)
            if filename.endswith(".tfrecords"):
                tfRecordsList.append(os.path.join(dataDir, filename))

        return tfRecordsList

    def parse_sequence_example(self, record_string, dataset_type):

        features_dict = {dataset_type + '/video': tf.FixedLenFeature([], tf.string),
                   dataset_type + '/label': tf.FixedLenFeature([], tf.int64)}

        features = tf.parse_single_example(record_string, features_dict)

        images = tf.decode_raw(features[dataset_type + '/video'], tf.uint8)
        images = tf.cast(images, dtype=tf.float32)
        label = tf.cast(features[dataset_type + '/label'], tf.int32)

        images = tf.reshape(images, INPUT_SHAPE)

        return images, label

    def input_decoder(self, filename_queue, dataset_type):
        reader = tf.TFRecordReader(
            options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        )
        key, record_string = reader.read(filename_queue)

        return self.parse_sequence_example(record_string, dataset_type)

    def mean_image_normalization(self, sess):

        num_batches = int(math.ceil(self.num_train_examples / self.batch_size))
        mean_channels = np.zeros((3))

        for i in range(num_batches):
            print('Normalization step {}/{}'.format(i + 1, num_batches))
            image_vals = sess.run(self.train_images)
            mean_image_vals = np.mean(image_vals, axis=0)
            mean_image_channels = np.mean(mean_image_vals, axis=(0, 1))
            np.add(mean_channels, mean_image_channels, mean_channels)

        np.divide(mean_channels, float(num_batches), mean_channels)
        self.train_images = util.vgg_normalization(self.train_images, mean_channels)
        self.valid_images = util.vgg_normalization(self.valid_images, mean_channels)
        self.test_images = util.vgg_normalization(self.test_images, mean_channels)
        print('Done with mean channel image dataset normalization...')

        return mean_channels

    def numberOfExamples(self, datasetType):

        dataDir = os.path.join(DATASET_DIR, datasetType)
        if not os.path.exists(dataDir):
            return 0

        totalDataNumber = 0
        for file in os.listdir(dataDir):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                totalDataNumber += len(open(os.path.join(dataDir, filename), 'r').readlines())

        return totalDataNumber