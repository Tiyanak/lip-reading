import time
import tensorflow as tf
import numpy as np
import random
import os

from utils import util
from read_write.tf_records_reader import TFRecordsReader
from cnn.model import Model
from utils import config
from read_write.read_mnist import ReadMnist
from read_write.logger import Logger
from cnn.models.resnet_18 import ResNet18

class CNN:

    def __init__(self, is_testing=False):

        self.initConfig()
        self.initLogger()
        self.init_plot_data()
        self.initModel(is_testing)
        self.initSession()
        self.initTfSummories(is_testing)
        self.initTfRecordsReaders(is_testing)
        self.initSessionVariables()
        self.createRestorer()

    def train(self):

        print("Training is starting right now!")

        for epoch_num in range(1, self.max_epochs + 1):

            epoch_start_time = time.time()

            for step in range(self.num_batches_train):

                start_time = time.time()

                batch_x, batch_y = self.getCurrentBatch(self.tfTrainTrainReader, 'train')
                batch_x = self.runResNet(batch_x)

                feed_dict = {self.model.X: batch_x, self.model.Yoh: batch_y, self.model.is_training: True}
                eval_tensors = [self.model.train_op, self.model.loss]

                if (step + 1) * self.batch_size % self.log_every == 0:
                    eval_tensors += [self.merged_summary_op, self.model.accuracy, self.model.precison, self.model.recall, self.model.precAtTop1,
                                self.model.precAtTop10, self.model.recallAtTop1, self.model.recallAtTop10]

                eval_ret = self.sess.run(eval_tensors, feed_dict=feed_dict)
                eval_ret = dict(zip(eval_tensors, eval_ret))

                loss_val = eval_ret[self.model.loss]

                if self.merged_summary_op in eval_tensors:
                    self.summary_train_train_writer.add_summary(eval_ret[self.merged_summary_op], self.model.global_step.eval(session=self.sess))

                duration = time.time() - start_time
                self.log_step(epoch_num, step, duration, loss_val)

                if (step % self.num_batches_train) > (0.05 * self.num_batches_train):
                    self.saver.save(self.sess, os.path.join(self.saved_session_dir, config.config['model_name'] + '.ckpt'))

            epoch_time = time.time() - epoch_start_time
            print("Total epoch time training: {}".format(epoch_time))

            self.startValidation(epoch_num, epoch_time)

        self.finishTraining()

    def startValidation(self, epoch_num, epoch_time):

        print("EPOCH VALIDATION : ")

        train_loss, train_acc, train_pr, train_rec = self.validate(self.num_examples_train, epoch_num, self.tfValidTrainReader, "train")
        valid_loss, valid_acc, valid_pr, valid_rec = self.validate(self.num_examples_val, epoch_num, self.tfValidValidReader, "val")
        lr = self.sess.run([self.model.learning_rate])

        self.plot_data['epoch_time'] += [epoch_time]
        self.plot_data['train_loss'] += [train_loss]
        self.plot_data['train_acc'] += [train_acc]
        self.plot_data['train_pr'] += [train_pr]
        self.plot_data['train_rec'] += [train_rec]
        self.plot_data['valid_loss'] += [valid_loss]
        self.plot_data['valid_acc'] += [valid_acc]
        self.plot_data['valid_pr'] += [valid_pr]
        self.plot_data['valid_rec'] += [valid_rec]
        self.plot_data['lr'] += [lr]

        util.plot_training_progress(self.plot_data)

    def predict(self, X):
        preds = self.sess.run(self.model.prediction, feed_dict={self.model.X: X})
        return preds

    def log_step(self, epoch, step, duration, loss):
        if (step + 1) * self.batch_size % self.log_every == 0:
            format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
            print(format_str % (epoch, (step + 1) * self.batch_size, self.num_batches_train * self.batch_size, loss, float(duration)))

    def validate(self, num_examples, epoch, reader, dataset_type="Unknown"):

        validBatchSize = self.batch_size
        num_batches = num_examples // validBatchSize
        losses = []

        eval_preds = np.ndarray((0,), dtype=np.int64)
        labels = np.ndarray((0,), dtype=np.int64)

        for step in range(num_batches):

            eval_tensors = [self.model.prediction, self.model.loss, self.model.accuracy, self.model.precison, self.model.recall,
                            self.model.precAtTop1, self.model.precAtTop10, self.model.recallAtTop1, self.model.recallAtTop10]
            if (step + 1) * validBatchSize % self.log_every == 0:
                print("Evaluating {}, done: {}/{}".format(dataset_type, (step + 1) * validBatchSize, num_batches * validBatchSize))
                eval_tensors += [self.merged_summary_op]

            batch_x, batch_y = self.getCurrentBatch(reader, dataset_type)

            feed_dict = {self.model.X: batch_x, self.model.Yoh: batch_y, self.model.is_training: False}

            eval_ret = self.sess.run(eval_tensors, feed_dict=feed_dict)
            eval_ret = dict(zip(eval_tensors, eval_ret))

            if self.merged_summary_op in eval_tensors:
                if dataset_type == 'train':
                    self.summary_valid_train_writer.add_summary(eval_ret[self.merged_summary_op], self.model.global_step.eval(session=self.sess))
                else:
                    self.summary_valid_valid_writer.add_summary(eval_ret[self.merged_summary_op], self.model.global_step.eval(session=self.sess))

            losses.append(eval_ret[self.model.loss])
            if eval_preds.size == 0:
                labels = np.argmax(batch_y, axis=1)
                eval_preds = np.argmax(eval_ret[self.model.prediction], axis=1)
            else:
                labels = np.concatenate((labels, np.argmax(batch_y, axis=1)), axis=0)
                eval_preds = np.concatenate((eval_preds, np.argmax(eval_ret[self.model.prediction], axis=1)), axis=0)

        total_loss = np.mean(losses)
        acc, pr, rec = util.acc_prec_rec_score(labels, eval_preds)

        print("Validation results -> {} error: epoch {} loss={} accuracy={} precision={} recall={}".format(dataset_type, epoch, total_loss, acc, pr, rec))

        return total_loss, acc, pr, rec

    def test(self):

        print('STARTED TESTING EVALUATION!')

        num_batches = self.num_examples_test // self.batch_size
        losses = []

        eval_preds = np.ndarray((0,), dtype=np.int64)
        labels = np.ndarray((0,), dtype=np.int64)

        for step in range(num_batches):

            eval_tensors = [self.model.prediction, self.model.loss, self.model.accuracy, self.model.precison,
                            self.model.recall, self.model.precAtTop1, self.model.precAtTop10, self.model.recallAtTop1,
                            self.model.recallAtTop10]
            if (step + 1) * self.batch_size % self.log_every == 0:
                print("Evaluating {}, done: {}/{}".format('test', (step + 1) * self.batch_size, num_batches * self.batch_size))
                eval_tensors += [self.merged_summary_op]

            batch_x, batch_y = self.getCurrentBatch(self.tfTestReader, 'test')
            feed_dict = {self.model.X: batch_x, self.model.Yoh: batch_y, self.model.is_training: False}

            eval_ret = self.sess.run(eval_tensors, feed_dict=feed_dict)
            eval_ret = dict(zip(eval_tensors, eval_ret))

            if self.merged_summary_op in eval_tensors:
                self.summary_test_writer.add_summary(eval_ret[self.merged_summary_op], self.model.global_step.eval(session=self.sess))

            losses.append(eval_ret[self.model.loss])
            if eval_preds.size == 0:
                labels = np.argmax(batch_y, axis=1)
                eval_preds = eval_ret[self.model.prediction]
            else:
                labels = np.concatenate((labels, np.argmax(batch_y, axis=1)), axis=0)
                eval_preds = np.concatenate((eval_preds,eval_ret[self.model.prediction]), axis=0)

        total_loss = np.mean(losses)
        acc, pr, rec = util.acc_prec_rec_score(labels, np.argmax(eval_preds, axis=1))

        prAtTop10, top5CorrectWords, top5IncorrectWords = util.testStatistics(labels, eval_preds)

        print("Validation results -> {} error: epoch {} loss={} accuracy={} precision={} recall={} prAtTop10={}".format('test', '1', total_loss, acc, pr, rec, prAtTop10))

        util.write_test_results(total_loss, acc, pr, rec, prAtTop10, top5CorrectWords, top5IncorrectWords)

    def init_plot_data(self):

        self.plot_data = {
            'train_loss': [], 'train_acc': [], 'train_pr': [], 'train_rec': [],
            'valid_loss': [], 'valid_acc': [], 'valid_pr': [], 'valid_rec': [],
            'lr': [], 'epoch_time': []
        }
        print("INITIALIZED PLOT DATA!")

    def initConfig(self):
   
        self.model_name = config.config['model_name']
        self.use_resnet = config.config['use_resnet']
        self.dataset_name = config.config['dataset']
        self.num_examples_train = util.numberOfData('train')
        self.num_examples_val = util.numberOfData('val')
        self.num_examples_test = util.numberOfData('test')
        self.max_epochs = config.config['max_epochs']
        self.batch_size = config.config['batch_size']
        self.num_batches_train = self.num_examples_train // self.batch_size
        self.log_every = config.config['log_every']
        self.frames = config.config['frames']
        self.num_classes = config.config['num_classes']
        self.summary_dir = config.config['summary_data']
        self.saved_session_dir = config.config['saved_session_root_dir']
        print('INITIALIZED CONFIGURATION!')

    def initModel(self, is_testing):

        print('Creating model: {}, with chosen dataset: {}'.format(self.model_name, self.dataset_name))

        if self.dataset_name == 'mnist':
            self.mnist = ReadMnist()
            self.num_examples_train = self.mnist.num_train
            self.num_examples_val = self.mnist.num_val
            self.num_examples_test = self.mnist.num_test
            self.num_batches_train = self.num_examples_train // self.batch_size

        if is_testing:
            with tf.device('/CPU:0'):
                self.tfTestReader = TFRecordsReader()
        else:
            with tf.device('/CPU:0'):
                self.tfTrainTrainReader = TFRecordsReader()
                self.tfValidTrainReader = TFRecordsReader()
                self.tfValidValidReader = TFRecordsReader()
        print("INITIALIZED READERS!")

        if self.use_resnet:
            self.resnet = ResNet18()

        self.model = Model()

        print("INITIALIZED MODEL!")

    def initSession(self):

        print('Creating dirs for saving session states (weights and biases): {}'.format(self.saved_session_dir))
        util.create_dir(self.saved_session_dir)

        print('Creating session saver')
        self.saver = tf.train.Saver()

        print('Creating session with saved session for dataset: {} and model: {}'.format(self.dataset_name, self.model_name))
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        gpu_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=gpu_config)
        self.sess.as_default()

    def createRestorer(self):

        if os.path.exists(os.path.join(self.saved_session_dir, self.model_name + '.ckpt.meta')):
            self.saver.restore(self.sess, os.path.join(self.saved_session_dir, config.config['model_name'] + '.ckpt'))

    def initTfRecordsReaders(self, is_testing=False):

        if self.dataset_name != 'mnist':
            print("Initializing variables.")
            if is_testing:
                self.tfTestReader.create_iterator('test', 1, self.batch_size)
            else:
                self.tfTrainTrainReader.create_iterator("train", self.max_epochs, self.batch_size)
                self.tfValidTrainReader.create_iterator("train", self.max_epochs, self.batch_size)
                self.tfValidValidReader.create_iterator("val", self.max_epochs, self.batch_size)

    def initSessionVariables(self):

        print("Initializing session variables")
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def getCurrentBatch(self, tfRecordsReader, datasettype='train'):

        if self.dataset_name == 'mnist':
            return self.mnist.get_batch(datasettype)
        else:
            batch_x, batch_y = self.sess.run([tfRecordsReader.images, tfRecordsReader.labels])
            batch_x = np.array(batch_x)[:, sorted(random.sample(range(0, 29), self.frames))]
            batch_y = util.class_to_onehot(batch_y, self.num_classes)
            return batch_x, batch_y

    def finishTraining(self):

        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()

    def initTfSummories(self, is_testing=False):

        self.merged_summary_op = tf.summary.merge_all()
        if is_testing:
            self.summary_test_writer = tf.summary.FileWriter(self.summary_dir['test'], self.sess.graph)
        else:
            self.summary_train_train_writer = tf.summary.FileWriter(self.summary_dir['train_train'], self.sess.graph)
            self.summary_valid_train_writer = tf.summary.FileWriter(self.summary_dir['valid_train'], self.sess.graph)
            self.summary_valid_valid_writer = tf.summary.FileWriter(self.summary_dir['valid_valid'], self.sess.graph)

    def initLogger(self):
        self.logger = Logger()

    def runResNet(self, batch_x):

        if self.use_resnet:
            ret_batch = []
            batch_x = np.transpose(batch_x, [1, 0, 2, 3, 4])

            for frame in batch_x:
                feed_dict = {self.resnet.X: frame, self.resnet.is_training: True}
                eval_tensors = [self.resnet.logits]
                ret_batch.append(self.sess.run(eval_tensors, feed_dict=feed_dict))

            return np.transpose(np.vstack(ret_batch), [1, 0, 2, 3, 4])
        else:
            return batch_x


