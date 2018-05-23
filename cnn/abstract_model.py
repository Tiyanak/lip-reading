import abc
from cnn import layers
from utils import config
import tensorflow as tf

class AbstractModel():

    @abc.abstractmethod
    def __init__(self):

        self.learning_rate = config.config['learning_rate']
        self.activation_fn = config.config['activation_fn']
        self.dataset_name = config.config['dataset']

        self.w = config.config['input_w']
        self.h = config.config['input_h']
        self.c = config.config['input_c']
        self.frames = config.config['frames']
        self.num_classes = config.config['num_classes']

        self.initializer = config.config['initializer']
        self.regularizer = config.config['regularizer']

        self.pool_size = config.config['pool_size']
        self.stride = config.config['stride']

    @abc.abstractmethod
    def build(self, input, is_training): pass

    def build_conv_block5(self, logits, is_training=True, initializer=None, regularizer=None):

        with tf.variable_scope(tf.get_variable_scope(), 'vgg_conv5_fc8'):

            logits = layers.conv2d(logits, 512, [3, 3], activation_fn=self.activation_fn, name='conv5',
                                   kernel_initializer=initializer, kernel_regularizer=regularizer,
                                   bias_regularizer=regularizer)
            logits = layers.max_pool2d(logits, pool_size=[3, 3], stride=2, name='max_pool5')

            logits = layers.flatten(logits, name='flatten')

            logits = layers.fc(logits, output_shape=4096, activation_fn=self.activation_fn, name='fc6',
                               kernel_initializer=initializer, kernel_regularizer=regularizer,
                               bias_regularizer=regularizer)
            logits = layers.fc(logits, output_shape=4096, activation_fn=self.activation_fn, name='fc7',
                               kernel_initializer=initializer, kernel_regularizer=regularizer,
                               bias_regularizer=regularizer)
            logits = layers.fc(logits, output_shape=self.num_classes, activation_fn=None, name='fc8',
                               kernel_initializer=initializer, kernel_regularizer=regularizer,
                               bias_regularizer=regularizer)

        return logits

    def build_conv_block3(self, logits, is_training=True, initializer=None, regularizer=None):

        with tf.variable_scope(tf.get_variable_scope(), 'vgg_conv3_fc8'):

            logits = layers.conv2d(logits, 512, [3, 3], activation_fn=self.activation_fn, name='conv3',
                                   kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)

            logits = layers.conv2d(logits, 512, [3, 3], activation_fn=self.activation_fn, name='conv4',
                                   kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)

            logits = layers.conv2d(logits, 512, [3, 3], activation_fn=self.activation_fn, name='conv5',
                                   kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
            logits = layers.max_pool2d(logits, pool_size=[3, 3], stride=2, name='max_pool5')

            logits = layers.flatten(logits, name='flatten')

            logits = layers.fc(logits, output_shape=4096, activation_fn=self.activation_fn, name='fc6',
                               kernel_initializer=initializer, kernel_regularizer=regularizer,
                               bias_regularizer=regularizer)
            logits = layers.fc(logits, output_shape=4096, activation_fn=self.activation_fn, name='fc7',
                               kernel_initializer=initializer, kernel_regularizer=regularizer,
                               bias_regularizer=regularizer)
            logits = layers.fc(logits, output_shape=self.num_classes, activation_fn=None, name='fc8',
                               kernel_initializer=initializer, kernel_regularizer=regularizer,
                               bias_regularizer=regularizer)

        return logits

