import tensorflow as tf

from cnn import layers
from utils import config
class Model():

    def __init__(self, is_training=True):

        self.learning_rate = config.config['learning_rate']
        self.input_w = config.config['input_w']
        self.input_h = config.config['input_h']
        self.input_c = config.config['input_c']
        self.num_classes = config.config['num_classes']
        self.frames = config.config['frames']
        self.opt = config.config['optimizer']
        self.decay_steps = config.config['decay_steps']
        self.decay_rate = config.config['decay_rate']
        self.dataset_name = config.config['dataset']

        self.X = tf.placeholder(name='image', dtype=tf.float32, shape=None)
        self.Yoh = tf.placeholder(name='label', dtype=tf.int64, shape=[None, self.num_classes])

        self.global_step = tf.Variable(0, trainable=False)
        self.is_training = tf.placeholder_with_default(bool(is_training), [], name='is_training')

        self.build_model()

    def build_model(self):

        self.logits = self.create_network()
        self.prediction = self.predict()

        self.loss = self.calcLoss()
        self.learning_rate = self.calcLearningRate()
        self.train_op = self.calcOpt()

        self.accuracy, self.precison, self.recall, self.precAtTop1, self.precAtTop10, self.recallAtTop1, \
        self.recallAtTop10 = self.calcMetrics()

        self.loss, self.learning_rate, self.accuracy, self.precison, self.recall, self.precAtTop1, \
        self.precAtTop10, self.recallAtTop1, self.recallAtTop10 = self.createSummaries()

    def calcMetrics(self):

        labels = layers.onehot_to_class(self.Yoh)
        preds = layers.onehot_to_class(self.prediction)

        return (self.calcAccuracy(labels, preds), self.calcPrecision(labels, preds), self.calcRecall(labels, preds), self.calcPrecisionAtTopK(1),
                self.calcPrecisionAtTopK(10), self.calcRecallAtTopK(1), self.calcRecallAtTopK(10))

    def createSummaries(self):

        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy[1])
        tf.summary.scalar('precision', self.precison[1])
        tf.summary.scalar('recall', self.recall[1])
        tf.summary.scalar('precision_top_1', self.precAtTop1[1])
        tf.summary.scalar('precision_top_10', self.precAtTop10[1])
        tf.summary.scalar('recall_top_1', self.recallAtTop1[1])
        tf.summary.scalar('recall_top_10', self.recallAtTop10[1])

        return self.loss, self.learning_rate, self.accuracy, self.precison, self.recall, self.precAtTop1, self.precAtTop10, self.recallAtTop1, self.recallAtTop10

    def create_network(self):
        return config.config['model'].build(self.X, self.is_training)

    def calcLoss(self):
        return tf.reduce_mean(layers.softmax_cross_entropy(self.logits, self.Yoh))

    def calcLearningRate(self):
        return tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=False, name='learning_rate_decay')

    def calcOpt(self):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

    def predict(self):
        return layers.softmax(self.logits)

    def calcAccuracy(self, labels, preds):
        return tf.metrics.accuracy(labels, preds)

    def calcMean_per_class_accuracy(self):
        return tf.metrics.mean_per_class_accuracy(self.Yoh, self.prediction, self.num_classes)

    def calcPrecision(self, labels, preds):
        return tf.metrics.precision(labels, preds)

    def calcRecall(self, labels, preds):
        return tf.metrics.recall(labels, preds)

    def calcPrecisionAtTopK(self, k):
        return tf.metrics.precision_at_top_k(self.Yoh, self.prediction, k)

    def calcRecallAtTopK(self, k):
        return tf.metrics.recall_at_top_k(self.Yoh, self.prediction, k)
