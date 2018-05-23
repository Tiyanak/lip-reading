import numpy as np
import tensorflow as tf

class ImageResizer():

    def start_resizing(self, images, width, height):

        return np.array(self.resize(images, width, height))

    def resize(self, images, width, height):

        with tf.variable_scope('image_resizer') as scope:

            tf_batch = tf.placeholder(dtype=tf.uint8, shape=[len(images), 256, 256, 3])
            resized_images = tf.image.resize_images(tf_batch, [height, width], method=tf.image.ResizeMethod.BICUBIC)
            tf_data = tf.cast(resized_images, tf.uint8)

        session = tf.Session()
        session.run(tf.global_variables_initializer())

        return session.run(tf_data, feed_dict={tf_batch: images})

