import numpy as np
import tensorflow as tf

class ImageResizer():

    def start_resizing(self, images, width, height, currentWidth, currentHeight, currentChannel, dtype):

        self.current_w = currentWidth
        self.current_h = currentHeight
        self.current_c = currentChannel
        self.dtype = dtype
        return np.array(self.resize(images, width, height))

    def resize(self, images, width, height):

        with tf.variable_scope('image_resizer') as scope:

            tf_batch = tf.placeholder(dtype=tf.float32, shape=[len(images), self.current_w, self.current_h, self.current_c])
            resized_images = tf.image.resize_images(tf_batch, [height, width], method=tf.image.ResizeMethod.BICUBIC)
            tf_data = tf.cast(resized_images, self.dtype)

        session = tf.Session()
        session.run(tf.global_variables_initializer())

        return session.run(tf_data, feed_dict={tf_batch: images})

