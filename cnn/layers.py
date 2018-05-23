import tensorflow as tf
import tensorflow.contrib.layers as contrib_layers

from utils import config

### AKTIVACIJE ###
def selu(x):
    alpha = 1.6733
    scale = 1.0507
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha, 'selu')

def relu(x):
    return tf.nn.relu(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)

def softmax(x):
    return tf.nn.softmax(x)

### GUBITAK ###
def softmax_cross_entropy(labels, logits):
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)

def decayLearningRate(learning_rate, global_step, decay_steps, decay_rate):
    return tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name='learning_rate_decay')

### POMOCNE FUNKCIJE ###
def concat(input, axis=0):
    return tf.concat(input, axis=axis)

def transpose(net, shape):
    return tf.transpose(net, shape)

def reshape(net, shape):
    return tf.reshape(net, shape)

def rgb_to_grayscale(images):
    return tf.image.rgb_to_grayscale(images)

def onehot_to_class(Yoh):
    return tf.argmax(Yoh, 1)

def stack(itemList, axis=0):
    return tf.stack(itemList, axis=axis)

def flatten(input, name='flatten'):
    return contrib_layers.flatten(input, scope=name)

def add(input1, input2):
    return tf.add(input1, input2)

def toOneHot(input, num_classes):
    return contrib_layers.one_hot_encoding(input, num_classes)

def reduce_mean(input):
    return tf.reduce_mean(input)

### OPTIMIZATORI ###
def adam(learningRate):
    return tf.train.AdamOptimizer(learningRate)

def sgd(learningRate):
    return tf.train.GradientDescentOptimizer(learningRate)

def adagrad(learningRate):
    return tf.train.AdagradOptimizer(learningRate)

def adadelta(learningRate):
    return tf.train.AdadeltaOptimizer(learningRate)

def adagradDA(learningRate):
    return tf.train.AdagradDAOptimizer(learningRate)

### INITIALIZATORI ###
def xavier_initializer():
    return contrib_layers.xavier_initializer()

### REGULARIZATORI ###
def l1_regularizer(scale=1.0):
    return contrib_layers.l1_regularizer(scale=scale)

def l2_regularizer(scale=1.0):
    return contrib_layers.l2_regularizer(scale=scale)

### METRIKA ###
def accuracy(labels, preds):
    return tf.metrics.accuracy(labels, preds)

def meanPerClassAccuracy(labels, preds, num_classes):
    return tf.metrics.mean_per_class_accuracy(labels, preds, num_classes)

def precision(labels, preds):
    return tf.metrics.precision(labels, preds)

def recall(labels, preds):
    return tf.metrics.recall(labels, preds)

def precisionAtTopK(labels, preds, k):
    return tf.metrics.precision_at_top_k(labels, preds, k)

def recallAtTopK(labels, preds, k):
    return tf.metrics.recall_at_top_k(labels, preds, k)

### KONVOLUCIJSKI SLOJEVI ###
def conv2d(input, filters=16, kernel_size=[3, 3], padding="same", stride=1, activation_fn=relu, reuse=None, name="conv2d",
           weights_initializer=xavier_initializer(), biases_initializer=tf.zeros_initializer(), weights_regularizer=None, biases_regularizer=None):

   return contrib_layers.conv2d(input, num_outputs=filters, kernel_size=kernel_size, padding=padding, stride=stride, scope=name,
                        activation_fn=activation_fn, reuse=reuse, weights_initializer=weights_initializer, biases_initializer=biases_initializer,
                        weights_regularizer=weights_regularizer, biases_regularizer=biases_regularizer)

def conv3d(input, filters=16, kernel_size=[3, 3, 3], padding="same", stride=1, activation_fn=relu, reuse=None, name="conv3d",
           weights_initializer=None, biases_initializer=tf.zeros_initializer(), weights_regularizer=None, biases_regularizer=None):

    return contrib_layers.conv2d(input, num_outputs=filters, kernel_size=kernel_size, padding=padding, stride=stride, scope=name,
                                 activation_fn=activation_fn, reuse=reuse, weights_initializer=weights_initializer,
                                 biases_initializer=biases_initializer, weights_regularizer=weights_regularizer, biases_regularizer=biases_regularizer)

### SLOJEVI SAZIMANJA ###
def max_pool2d(input, kernel_size=[3, 3], stride=2, padding='VALID', name="max_pool2d"):
    return contrib_layers.max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding, scope=name)

def max_pool3d(input, kernel_size=[3, 3, 3], stride=2, padding='VALID', name='max_pool3d'):
    return contrib_layers.max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding, scope=name)

def avg_pool2d(input, kernel_size=[3, 3], stride=2, padding='VALID', name="max_pool2d"):
    return contrib_layers.avg_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding, scope=name)

def avg_pool3d(input, kernel_size=[3, 3, 3], stride=2, padding='VALID', name="max_pool3d"):
    return contrib_layers.avg_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding, scope=name)

### POTPUNO POVEZANI SLOJ ###
def fc(inputs, num_outputs, activation_fn=relu, weights_initializer=xavier_initializer(), weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(), biases_regularizer=None, reuse=None, name=None):

    return contrib_layers.fully_connected(inputs, num_outputs=num_outputs, activation_fn=activation_fn, scope=name, reuse=reuse,
                        weights_initializer=weights_initializer, biases_initializer=biases_initializer,
                        weights_regularizer=weights_regularizer, biases_regularizer=biases_regularizer)

### NORMALIZACIJA ###
def batchNormalization(input, is_training=True, reuse=None, name='bn'):
    return contrib_layers.batch_norm(inputs=input, is_training=is_training, reuse=reuse, scope=name)

def lrn (input):
    return tf.nn.local_response_normalization(input)

# SQUEEZE AND EXCITE
def squeeze_and_excite(input, filters=16, scope='se', name='se'):

    filters2 = (int) (filters / config.config['se_r'])
    filters1 = filters

    se = avg_pool2d(input, name='se_avg_pool')
    se = fc(se, filters1, activation_fn=relu, name='se_fc_' + name + '_1')
    se = fc(se, filters2, activation_fn=sigmoid, name='se_fc_' + name + '_2')

    return se