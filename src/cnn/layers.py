import tensorflow as tf
import tensorflow.contrib.layers as contrib_layers
import tensorflow.contrib as contrib
from tensorflow.contrib.framework.python.framework import checkpoint_utils
slim = tf.contrib.slim

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
    return tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)

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

def flatten(input, name=None):
    return contrib_layers.flatten(input, scope=name)

def add(input1, input2):
    return tf.add(input1, input2)

def toOneHot(input, num_classes):
    return contrib_layers.one_hot_encoding(input, num_classes)

def reduce_mean(input):
    return tf.reduce_mean(input)

def convertImageType(image):
    return tf.image.convert_image_dtype(image, dtype=tf.float32)

def imageStandardization(image):
    return tf.image.per_image_standardization(image)

def imagesStandardization(images):
    return tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)

def scan_checkpoint_for_vars(checkpoint_path, vars_to_check):
    check_var_list = checkpoint_utils.list_variables(checkpoint_path)
    check_var_list = [x[0] for x in check_var_list]
    check_var_set = set(check_var_list)
    vars_in_checkpoint = [x for x in vars_to_check if x.name[:x.name.index(":")] in check_var_set]
    vars_not_in_checkpoint = [x for x in vars_to_check if x.name[:x.name.index(":")] not in check_var_set]
    return vars_in_checkpoint, vars_not_in_checkpoint

def remove_vars(mainVars, varsToRemove):
    varNamesToRemove = set([x[0] for x in varsToRemove])
    return [x for x in mainVars if x.name[:x.name.index(":")] not in varNamesToRemove]

def replaceVarInListsByName(listToRemoveFrom, listToPutIn, varName):
    listToPutIn += [x for x in listToRemoveFrom if varName in x.name]
    listToRemoveFrom = [x for x in listToRemoveFrom if varName not in x.name]
    return listToRemoveFrom, listToPutIn

def latest_checkpoint(ckpt_dir, latestFilename=None):
    return tf.train.latest_checkpoint(ckpt_dir, latest_filename=latestFilename)

### OPTIMIZATORI ###
def adam(learningRate):
    return tf.train.AdamOptimizer(learningRate)

def sgd(learningRate):
    return tf.train.GradientDescentOptimizer(learningRate)

def adagrad(learningRate):
    return tf.train.AdagradOptimizer(learningRate)

def adadelta(learningRate):
    return tf.train.AdadeltaOptimizer(learningRate)

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
def conv2d(input, filters=16, kernel_size=3, padding="same", stride=1, activation_fn=relu, reuse=None, name=None, data_format='NHWC',
           weights_initializer=xavier_initializer(), biases_initializer=tf.zeros_initializer(), weights_regularizer=None, biases_regularizer=None,
           normalizer_fn=None, normalizer_params=None):

   return contrib_layers.conv2d(input, num_outputs=filters, kernel_size=kernel_size, padding=padding, stride=stride, scope=name, data_format=data_format,
                        activation_fn=activation_fn, reuse=reuse, weights_initializer=weights_initializer, biases_initializer=biases_initializer,
                        weights_regularizer=weights_regularizer, biases_regularizer=biases_regularizer, normalizer_fn=normalizer_fn, normalizer_params=normalizer_params)

def conv3d(input, filters=16, kernel_size=3, padding="same", stride=1, activation_fn=relu, reuse=None, name=None, data_format='NDHWC',
           weights_initializer=None, biases_initializer=tf.zeros_initializer(), weights_regularizer=None, biases_regularizer=None,
           normalizer_fn=None, normalizer_params=None):

    return contrib_layers.conv3d(input, num_outputs=filters, kernel_size=kernel_size, padding=padding, stride=stride, scope=name,
                                 activation_fn=activation_fn, reuse=reuse, weights_initializer=weights_initializer, data_format=data_format,
                                 biases_initializer=biases_initializer, weights_regularizer=weights_regularizer, biases_regularizer=biases_regularizer,
                                 normalizer_fn=normalizer_fn, normalizer_params = normalizer_params)

### SLOJEVI SAZIMANJA ###
def max_pool2d(input, kernel_size=3, stride=2, padding='VALID', name=None, data_format='NHWC'):
    return contrib_layers.max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding, scope=name, data_format=data_format)

def max_pool3d(input, kernel_size=3, stride=2, padding='VALID', name=None, data_format='NDHWC'):
    return contrib_layers.max_pool3d(input, kernel_size=kernel_size, stride=stride, padding=padding, scope=name, data_format=data_format)

def avg_pool2d(input, kernel_size=3, stride=2, padding='VALID', name=None, data_format='NHWC'):
    return contrib_layers.avg_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding, scope=name, data_format=data_format)

def avg_pool3d(input, kernel_size=3, stride=2, padding='VALID', name=None, data_format='NDHWC'):
    return contrib_layers.avg_pool3d(input, kernel_size=kernel_size, stride=stride, padding=padding, scope=name, data_format=data_format)

### POTPUNO POVEZANI SLOJ ###
def fc(inputs, num_outputs, activation_fn=relu, weights_initializer=xavier_initializer(), weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(), biases_regularizer=None, reuse=None, name=None, normalizer_fn=None,
                                                normalizer_params=None):

    return contrib_layers.fully_connected(inputs, num_outputs=num_outputs, activation_fn=activation_fn, scope=name, reuse=reuse,
                        weights_initializer=weights_initializer, biases_initializer=biases_initializer,
                        weights_regularizer=weights_regularizer, biases_regularizer=biases_regularizer,
                        normalizer_fn=normalizer_fn, normalizer_params=normalizer_params)

### NORMALIZACIJA ###
def batchNormalization(input, is_training=True, reuse=None, name=None, decay=0.999, center=True, scale=True,
            epsilon=0.001, updates_collections=None):
    return contrib_layers.batch_norm(inputs=input, is_training=is_training, reuse=reuse, scope=name, decay=decay, center=center, scale=scale, epsilon=epsilon, updates_collections=updates_collections)

def lrn (input):
    return tf.nn.local_response_normalization(input)

# SQUEEZE AND EXCITE
def squeeze_and_excite2d(input, indexHeight, indexWidth, name, filters=16, reuse=False):

    filters1 = (int) (filters / 16)
    filters2 = filters

    se = avg_pool2d(input, kernel_size=[input.shape[indexHeight], input.shape[indexWidth]], name=name + 'avgpool')
    se = fc(se, filters1, activation_fn=relu, name=name + 'fc1', reuse=reuse)
    se = fc(se, filters2, activation_fn=sigmoid, name=name + 'fc2', reuse=reuse)
    se = tf.multiply(input, se)

    return se

def squeeze_and_excite3d(input, indexHeight, indexWidth, indexSeq, name, filters=16, reuse=False):

    filters1 = (int) (filters / 16)
    filters2 = filters

    se = avg_pool3d(input, kernel_size=[input.shape[indexHeight], input.shape[indexWidth], input.shape[indexSeq]], name=name + 'avgpool')
    se = fc(se, filters1, activation_fn=relu, name=name + 'fc1', reuse=reuse)
    se = fc(se, filters2, activation_fn=sigmoid, name=name + 'fc2', reuse=reuse)
    se = tf.multiply(input, se)

    return se

def HashTable(keys, values):
    return contrib.lookup.HashTable(contrib.lookup.KeyValueTensorInitializer(keys, values), -1)