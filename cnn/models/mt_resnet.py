from cnn import layers
from cnn.abstract_model import AbstractModel
import tensorflow as tf
from utils import config

class MT_ResNet(AbstractModel):

    def __init__(self):
        super().__init__()
        self.name = 'mt_resnet'

    def build(self, logits, is_training=True, initializer=None, regularizer=None):

        logits = self.build_local_model(logits, is_training, initializer, regularizer)
        logits = self.build_conv_block3(logits, is_training, initializer, regularizer)

        return logits

    def build_local_model(self, logits, is_training=True, initializer=None, regularizer=None):

        with tf.variable_scope(tf.get_variable_scope(), 'resnet_18_mt'):

            logits = layers.reshape(logits, [-1, self.frames, 22, 22, 512]) # OUTPUT FROM RESNET-18

            logits = layers.transpose(logits, [0, 2, 3, 1, 4])
            logits = layers.reshape(logits, [-1, logits.shape[1], logits.shape[2], logits.shape[3] * logits.shape[4]])

            logits = layers.conv2d(logits, filters=96, kernel_size=[1, 1], name='conv1d',
                                   kernel_initializer=initializer, kernel_regularizer=regularizer,
                                   bias_regularizer=regularizer)

            logits = layers.conv2d(logits, stride=2, filters=256, name='conv2', activation_fn=None,
                                   kernel_initializer=initializer, kernel_regularizer=regularizer, bias_regularizer=regularizer)
            # logits = layers.batchNormalization(logits, is_training=is_training)
            logits = layers.relu(logits)
            logits = layers.max_pool2d(logits, pool_size=[3, 3], stride=2, name='max_pool2')

        return logits