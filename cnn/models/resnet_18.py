from cnn.abstract_model import AbstractModel
import tensorflow as tf
from cnn import layers

class ResNet18(AbstractModel):

    def __init__(self):
        super().__init__()
        self.name = 'ResNet18'

        self.X = tf.placeholder(name='image', dtype=tf.float32, shape=[None, self.w, self.h, self.c])
        self.is_training = tf.placeholder_with_default(bool(True), [], name='is_training')

        self.build(self.X, self.is_training)

    def build(self, logits, is_training=True):

        self.logits = self.build_resnet_18(logits, is_training)

    def build_resnet_18(self, logits, is_training=True):

        with tf.variable_scope(tf.get_variable_scope(), 'resnet_18'):

            logits = layers.conv2d(logits, 64, [7, 7], stride=2, activation_fn=None, name='resnet_conv1',
                               kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)
            # logits = layers.batchNormalization(logits, is_training, reuse=reuse)
            logits = layers.relu(logits)
            logits = layers.max_pool2d(logits, pool_size=[3, 3], stride=2, padding='same')

            logits = self.res_block(logits, 64, [3, 3], is_training=is_training, name='resnet_conv2a')
            logits = self.res_block(logits, 64, [3, 3], is_training=is_training, name='resnet_conv2b')

            logits = self.res_block_first(logits, 128, [3, 3], is_training=is_training, name='resnet_conv3a')
            logits = self.res_block(logits, 128, [3, 3], is_training=is_training, name='resnet_conv3b')

            logits = self.res_block_first(logits, 256, [3, 3], is_training=is_training, name='resnet_conv4a')
            logits = self.res_block(logits, 256, [3, 3], is_training=is_training, name='resnet_conv4b')

            logits = self.res_block_first(logits, 512, [3, 3], is_training=is_training, name='resnet_conv5a')
            logits = self.res_block(logits, 512, [3, 3], is_training=is_training, name='resnet_conv5b')

            logits = layers.avg_pool2d(logits, pool_size=[7, 7])

        return logits

    def res_block(self, logits, filters=64, kernel=[3, 3], is_training=True, name='resnet_convX', reuse=None):

        with tf.variable_scope(tf.get_variable_scope(), 'resnet_block'):

            tmp_logits = logits
            logits = layers.conv2d(logits, filters, kernel, padding='same', activation_fn=None, name=name + '_1', reuse=reuse,
                                   kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)
            # logits = layers.batchNormalization(logits, is_training, name='bn_' + name + '_1', reuse=reuse)
            logits = layers.relu(logits)
            logits = layers.conv2d(logits, filters, kernel, padding='same', activation_fn=None, name=name + '_2', reuse=reuse,
                                   kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)
            # logits = layers.batchNormalization(logits, is_training, name='bn_' + name + '_2', reuse=reuse)
            logits = layers.add(logits, tmp_logits)
            logits = layers.relu(logits)

        return logits

    def res_block_first(self, logits, filters=64, kernel=[3, 3], stride=1, is_training=True, name='resnet_convX'):

        with tf.variable_scope(tf.get_variable_scope(), 'resnet_block_first'):

            tmp_logits = layers.conv2d(logits, filters, kernel, stride=stride, padding='same', activation_fn=None, name=name + '_shortcut',
                                   kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)

            logits = layers.conv2d(logits, filters, kernel, stride=stride, padding='same', activation_fn=None, name=name + '_1',
                                   kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)
            # logits = layers.batchNormalization(logits, is_training, name='bn_' + name + '_1', reuse=reuse)
            logits = layers.relu(logits)
            logits = layers.conv2d(logits, filters, kernel, stride=stride, padding='same', activation_fn=None, name=name + '_2',
                                   kernel_initializer=self.initializer, kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)
            # logits = layers.batchNormalization(logits, is_training, name='bn_' + name + '_2',  reuse=reuse)
            logits = layers.add(logits, tmp_logits)
            logits = layers.relu(logits)

        return logits
