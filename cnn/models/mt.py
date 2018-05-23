from cnn import layers
from cnn.abstract_model import AbstractModel
import tensorflow as tf
from utils import config

class MT(AbstractModel):

    def __init__(self):
        super().__init__()
        self.name = 'mt'

    def build(self, logits, is_training=True):

        logits = self.build_local_model(logits, is_training)
        logits = self.build_conv_block3(logits, is_training)

        return logits

    def build_local_model(self, logits, is_training=True):

        with tf.variable_scope(tf.get_variable_scope(), 'mt'):

            logits = layers.reshape(logits, [-1, self.frames, self.w, self.h, self.c])
            logits = layers.transpose(logits, [1, 0, 2, 3, 4])
            tmpLogits = []

            reuse = None
            for i in range(self.frames):
                tower = layers.conv2d(logits[i], stride=2, padding='valid', filters=48, name='conv1', activation_fn=None, reuse=reuse,
                                      kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)
                # tower = layers.batchNormalization(tower, is_training=is_training, reuse=reuse)
                tower = layers.relu(tower)
                tower = layers.max_pool2d(tower, pool_size=[3, 3], stride=2, name='max_pool1')
                tmpLogits.append(tower)
                reuse = True

            logits = layers.stack(tmpLogits)
            del tmpLogits[:]

            logits = layers.transpose(logits, [1, 2, 3, 0, 4])
            logits = layers.reshape(logits, [-1, logits.shape[1], logits.shape[2], logits.shape[3] * logits.shape[4]])

            logits = layers.conv2d(logits, filters=96, kernel_size=[1, 1], name='conv1d',
                                   kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer)

            logits = layers.conv2d(logits, stride=2, filters=256, name='conv2', activation_fn=None,
                                   kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer)
            # logits = layers.batchNormalization(logits, is_training=is_training)
            logits = layers.relu(logits)
            logits = layers.max_pool2d(logits, pool_size=[3, 3], stride=2, name='max_pool2')

        return logits