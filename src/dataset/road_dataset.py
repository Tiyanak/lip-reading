import os
import math
import tensorflow as tf
import numpy as np
from src.utils import util

DATASET_DIR = 'E:/workspace/relic-diplomski/data/he_ftts_irap_intersection_lines_350_140_tagged'
INPUT_SHAPE = [25, 140, 350, 3]
ADD_GEOLOCATIONS = False

class RoadDataset():

    def __init__(self, is_training=True, batch_size=20):

        self.batch_size = batch_size
        self._initConfig()

        if ADD_GEOLOCATIONS:
            shapes = [INPUT_SHAPE, [], [2]]
        else:
            shapes = [INPUT_SHAPE, []]

        self.contains_geolocations = ADD_GEOLOCATIONS

        self.train_dir = os.path.join(DATASET_DIR, 'train')
        self.valid_dir = os.path.join(DATASET_DIR, 'validate')
        self.test_dir = os.path.join(DATASET_DIR, 'test')

        self.train_tfrecords_dirs = [tfrecords_dir for tfrecords_dir in os.listdir(self.train_dir)]
        self.train_tfrecords = [
            os.path.join(self.train_dir, train_tfrecords_dir, train_tfrecords_dir + '_sequential.tfrecords')
            for train_tfrecords_dir in self.train_tfrecords_dirs]

        self.valid_tfrecords_dirs = [tfrecords_dir for tfrecords_dir in os.listdir(self.valid_dir)]
        self.valid_tfrecords = [
            os.path.join(self.valid_dir, valid_tfrecords_dir, valid_tfrecords_dir + '_sequential.tfrecords')
            for valid_tfrecords_dir in self.valid_tfrecords_dirs]

        self.test_tfrecords_dirs = [tfrecords_dir for tfrecords_dir in os.listdir(self.test_dir)]
        self.test_tfrecords = [
            os.path.join(self.test_dir, test_tfrecords_dir, test_tfrecords_dir + '_sequential.tfrecords')
            for test_tfrecords_dir in self.test_tfrecords_dirs]

        self.num_train_examples = self.number_of_examples(self.train_tfrecords_dirs, self.train_dir)
        self.num_valid_examples = self.number_of_examples(self.valid_tfrecords_dirs, self.valid_dir)
        self.num_test_examples = self.number_of_examples(self.test_tfrecords_dirs, self.test_dir)

        self.num_batches_train = self.num_train_examples // self.batch_size
        self.num_batches_valid = self.num_valid_examples // self.batch_size
        self.num_batches_test = self.num_test_examples // self.batch_size

        train_file_queue = tf.train.string_input_producer(self.train_tfrecords, capacity=len(self.train_tfrecords))
        valid_file_queue = tf.train.string_input_producer(self.valid_tfrecords, capacity=len(self.valid_tfrecords))
        test_file_queue = tf.train.string_input_producer(self.test_tfrecords, capacity=len(self.test_tfrecords))

        if ADD_GEOLOCATIONS:

            train_images, train_labels, train_geolocations = self.input_decoder(train_file_queue)
            test_images, test_labels, test_geolocations = self.input_decoder(test_file_queue)
            valid_images, valid_labels, valid_geolocations = self.input_decoder(valid_file_queue)

            if is_training:
                self.train_images, self.train_labels, self.train_geolocations = tf.train.shuffle_batch(
                    [train_images, train_labels, train_geolocations], batch_size=self.batch_size, shapes=shapes,
                    capacity=100, min_after_dequeue=50)
            else:
                self.train_images, self.train_labels, self.train_geolocations = tf.train.batch(
                    [train_images, train_labels, train_geolocations], batch_size=self.batch_size, shapes=shapes)

            self.valid_images, self.valid_labels, self.valid_geolocations = tf.train.batch(
                [valid_images, valid_labels, valid_geolocations], batch_size=self.batch_size, shapes=shapes)

            self.test_images, self.test_labels, self.test_geolocations = tf.train.batch(
                [test_images, test_labels, test_geolocations], batch_size=self.batch_size, shapes=shapes)


        else:

            train_images, train_labels = self.input_decoder(train_file_queue)
            valid_images, valid_labels = self.input_decoder(valid_file_queue)
            test_images, test_labels = self.input_decoder(test_file_queue)

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

        self.name = 'road'
        util.create_dir(DATASET_DIR)
        self.frames = INPUT_SHAPE[0]
        self.h = INPUT_SHAPE[1]
        self.w = INPUT_SHAPE[2]
        self.c = INPUT_SHAPE[3]
        self.num_classes = 2

    def parse_sequence_example(self, record_string):

        features_dict = {
            'images_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'sequence_length': tf.FixedLenFeature([], tf.int64)
        }

        if ADD_GEOLOCATIONS:
            features_dict['geo'] = tf.FixedLenFeature([], tf.string)

        features = tf.parse_single_example(record_string, features_dict)
        images = tf.decode_raw(features['images_raw'], tf.float32)
        width = tf.cast(features['width'], tf.int32)
        height = tf.cast(features['height'], tf.int32)
        depth = tf.cast(features['depth'], tf.int32)
        label = tf.cast(features['label'], tf.int32)
        sequence_length = tf.cast(features['sequence_length'], tf.int32)
        images = tf.reshape(images, [sequence_length, height, width, depth])

        if ADD_GEOLOCATIONS:
            geo = tf.decode_raw(features['geo'], tf.float32)
            geo = tf.reshape(geo, [2, ])
            return images, label, geo
        else:
            return images, label

    def number_of_examples(self, tfrecord_dirs, path_prefix='', examples_file_name='examples.txt'):
        examples = 0
        for tfrecord_dir in tfrecord_dirs:
            with open(os.path.join(path_prefix, tfrecord_dir, examples_file_name)) as examples_file:
                examples += int(examples_file.read().strip())

        return examples

    def input_decoder(self, filename_queue):
        reader = tf.TFRecordReader(
            options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        )
        key, record_string = reader.read(filename_queue)

        return self.parse_sequence_example(record_string)

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