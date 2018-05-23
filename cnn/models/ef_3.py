from cnn import layers
from cnn.abstract_model import AbstractModel
from utils import config
import tensorflow as tf

class EF_3(AbstractModel):

    def __init__(self):
        super().__init__()
        self.name = 'ef3'

    def build(self, logits, is_training=True):

        logits = self.build_local_model(logits, is_training)
        logits = self.build_conv_block5(logits, is_training)

        return logits

    def build_local_model(self, logits, is_training=True):

        with tf.variable_scope(tf.get_variable_scope(), 'ef_3'):

            logits = layers.reshape(logits, [-1, self.frames, self.w, self.h, self.c])

            logits = layers.conv3d(logits, stride=2, padding='valid', filters=48, name='conv1', activation_fn=None,
                                   kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer)
            # logits = layers.batchNormalization(logits, is_training=is_training)
            logits = layers.relu(logits)
            logits = layers.max_pool3d(logits, pool_size=[3, 3, 3], stride=2, name='max_pool1')

            logits = layers.conv3d(logits, stride=2, padding='same', filters=256, name='conv2', activation_fn=None,
                                   kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer)
            # logits = layers.batchNormalization(logits, is_training=is_training)
            logits = layers.relu(logits)
            logits = layers.max_pool3d(logits, pool_size=[3, 3, 3], stride=2, name='max_pool2')

            logits = layers.transpose(logits, [0, 2, 3, 1, 4])
            logits = layers.reshape(logits, [-1, logits.shape[1], logits.shape[2], logits.shape[3] * logits.shape[4]])

            logits = layers.conv2d(logits, filters=512, name='conv3',
                                   kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer)

            logits = layers.conv2d(logits, filters=512, name='conv4',
                                   kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer)

        return logits