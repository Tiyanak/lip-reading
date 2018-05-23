from cnn import layers
from cnn.abstract_model import AbstractModel
import tensorflow as tf
from utils import config

class MT_3_ResNet(AbstractModel):

    def __init__(self):
        super().__init__()
        self.name = 'MT_ResNet_3'
        self.frames = config.config['frames']

    def build(self, logits, is_training=True):

        logits = self.build_local_model(logits, is_training)
        logits = self.build_conv_block5(logits, is_training)

        return logits


    def build_local_model(self, logits, is_training=True):

        with tf.variable_scope(tf.get_variable_scope(), 'resnet_18_mt_3'):

            logits = layers.transpose(logits, [1, 0, 2, 3, 4])
            tmpLogits = []

            reuse = None
            for i in range(self.frames):
                tower = self.build_resnet_18(logits[i], is_training=is_training, initializer=self.initializer, regularizer=self.regularizer)
                # logits = layers.batchNormalization(logits, is_training=is_training, reuse=reuse)
                tower = layers.relu(tower)
                tower = layers.max_pool2d(tower, pool_size=[3, 3], stride=2, name='max_pool1' + str(i))
                tmpLogits.append(tower)
                reuse = True

            logits = layers.stack(tmpLogits)
            del tmpLogits[:]

            logits = layers.reshape(layers.transpose(logits, [1, 2, 3, 0, 4]), [-1, 27, 27, 1200])

            logits = layers.conv3d(logits, stride=2, filters=256, name='conv2', activation_fn=None,
                                   kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer)
            # logits = layers.batchNormalization(logits, is_training=is_training)
            logits = layers.relu(logits)
            logits = layers.max_pool3d(logits, pool_size=[3, 3, 3], stride=2, name='max_pool2')

            logits = layers.conv3d(logits, filters=512, name='conv3',
                                   kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer)

            logits = layers.conv3d(logits, filters=512, name='conv4',
                                   kernel_initializer=self.initializer, kernel_regularizer=self.regularizer,
                                   bias_regularizer=self.regularizer)

        return logits