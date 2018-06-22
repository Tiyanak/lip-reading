from src.utils import util
from src import config
from src.cnn import layers
import tensorflow as tf
import time
import os
import numpy as np
from src.dataset import lrw_dataset, mnist_original_dataset, road_dataset, mnist_dataset, cifar_dataset

DATASET_TO_USE = 'lrw'
LOG_EVERY = 200
SAVE_EVERY = 0.1
DECAY_STEPS = 10000 # broj koraka za smanjivanje stope ucenja
DECAY_RATE = 0.96 # rate smanjivanja stope ucenja
REGULARIZER_SCALE = 1e-4 # faktor regularizacije
LEARNING_RATE = 1e-4
BATCH_SIZE = 200
MAX_EPOCHS = 10

class MT_VGG16:

    def __init__(self):

        with tf.device('/device:GPU:1'):
            self.initConfig() # postavi globalne varijable
            self.createPlotDataVars() # kreiraj mapu za zapisivanje metrike (pdf + csv)
            self.initDataReaders() # postavi dataset (reader za tfrecordse)
            self.buildPartialModel() # kreiraj duboki model (tensorski graf)
            self.initSummaries() # kreiraj summari writere
            self.createSession() # kreiraj session, restoraj i inizijaliziraj varijable
            self.addGraphToSummaries() # dodaj graf u summarije

    def createModel(self):

        print("CREATING MODEL")

        self.global_step = tf.Variable(0, trainable=False)
        self.is_training = tf.placeholder_with_default(True, [], name='is_training')
        self.learning_rate = layers.decayLearningRate(LEARNING_RATE, self.global_step, DECAY_STEPS, DECAY_RATE)

        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.dataset.frames, self.dataset.h, self.dataset.w, self.dataset.c])
        self.Y = tf.placeholder(dtype=tf.int32, shape=[None])

        self.Yoh = layers.toOneHot(self.Y, self.dataset.num_classes)

        reuse = None
        towersLogits = []
        for sequence_image in range(self.dataset.frames):
            net = self.vgg16(self.X[:, sequence_image], reuse)
            towersLogits.append(net)
            reuse = True

        net = layers.stack(towersLogits)
        del towersLogits[:]

        net = layers.transpose(net, [1, 2, 3, 0, 4])
        net = layers.reshape(net, [-1, net.shape[1], net.shape[2], net.shape[3] * net.shape[4]])

        net = layers.fc(net, 512, name='fc5', weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE))
        net = layers.squeeze_and_excite2d(net, indexHeight=1, indexWidth=2, name='se5', filters=512)

        net = layers.flatten(net, name='flatten')

        net = layers.fc(net, 4096, name='fc6', weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE))
        net = layers.fc(net, 4096, name='fc7', weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE))

        self.logits = layers.fc(net, self.dataset.num_classes, activation_fn=None, name='fc8',
                                weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE))

        self.preds = layers.softmax(self.logits)

        cross_entropy_loss = layers.reduce_mean(layers.softmax_cross_entropy(logits=self.logits, labels=self.Yoh))
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = cross_entropy_loss + REGULARIZER_SCALE * tf.reduce_sum(regularization_loss)

        self.opt = layers.adam(self.learning_rate)
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

        self.accuracy, self.precision, self.recall = self.createSummaries(self.Yoh, self.preds, self.loss,
                                                                              self.learning_rate)

    def vgg16(self, net, reuse=False):

        with tf.variable_scope('vgg_16', reuse=reuse):

            bn_params = {'decay': 0.999, 'center': True, 'scale': True, 'epsilon': 0.001, 'updates_collections': None,
                         'is_training': self.is_training}

            with tf.variable_scope('conv1', reuse=reuse):
                net = layers.conv2d(net, 64, name='conv1_1', reuse=reuse,
                                    weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                                    normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
                net = layers.conv2d(net, 64, name='conv1_2', reuse=reuse,
                                    weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                                    normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
                net = layers.max_pool2d(net, [2, 2], 2, name='pool1')
                net = layers.squeeze_and_excite2d(net, indexHeight=1, indexWidth=2, name='vgg_se1', filters=64)

            with tf.variable_scope('conv2', reuse=reuse):
                net = layers.conv2d(net, 128, name='conv2_1', reuse=reuse,
                                    weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                                    normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
                net = layers.conv2d(net, 128, name='conv2_2', reuse=reuse,
                                    weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                                    normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
                net = layers.max_pool2d(net, [2, 2], 2, name='pool2')
                net = layers.squeeze_and_excite2d(net, indexHeight=1, indexWidth=2, name='vgg_se2', filters=128)

            with tf.variable_scope('conv3', reuse=reuse):
                net = layers.conv2d(net, 256, name='conv3_1', reuse=reuse,
                                    weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                                    normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
                net = layers.conv2d(net, 256, name='conv3_2', reuse=reuse,
                                    weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                                    normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
                net = layers.conv2d(net, 256, name='conv3_3', reuse=reuse,
                                    weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                                    normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
                net = layers.max_pool2d(net, [2, 2], 2, name='pool3')
                net = layers.squeeze_and_excite2d(net, indexHeight=1, indexWidth=2, name='vgg_se3', filters=256)

            with tf.variable_scope('conv4', reuse=reuse):
                net = layers.conv2d(net, 512, name='conv4_1', reuse=reuse,
                                    weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                                    normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
                net = layers.conv2d(net, 512, name='conv4_2', reuse=reuse,
                                    weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                                    normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
                net = layers.conv2d(net, 512, name='conv4_3', reuse=reuse,
                                    weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                                    normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
                net = layers.max_pool2d(net, [2, 2], 2, name='pool4')
                net = layers.squeeze_and_excite2d(net, indexHeight=1, indexWidth=2, name='vgg_se4', filters=512)

            with tf.variable_scope('conv5', reuse=reuse):
                net = layers.conv2d(net, 512, name='conv5_1', reuse=reuse,
                                    weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                                    normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
                net = layers.conv2d(net, 512, name='conv5_2', reuse=reuse,
                                    weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                                    normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
                net = layers.conv2d(net, 512, name='conv5_3', reuse=reuse,
                                    weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                                    normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
                net = layers.max_pool2d(net, [2, 2], 2, name='pool5')
                net = layers.squeeze_and_excite2d(net, indexHeight=1, indexWidth=2, name='vgg_se5', filters=512)

        return net

    def createModelLowMemory(self):

        print("CREATING MODEL")

        self.global_step = tf.Variable(0, trainable=False)
        self.is_training = tf.placeholder_with_default(True, [], name='is_training')
        self.learning_rate = layers.decayLearningRate(LEARNING_RATE, self.global_step, DECAY_STEPS, DECAY_RATE)

        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.dataset.frames, self.dataset.h, self.dataset.w, self.dataset.c])
        self.Y = tf.placeholder(dtype=tf.int32, shape=[None])

        self.Yoh = layers.toOneHot(self.Y, self.dataset.num_classes)

        reuse = None
        towersLogits = []
        for sequence_image in range(self.dataset.frames):
            net = self.mt_loop_low_memory(self.X[:, sequence_image], reuse)
            towersLogits.append(net)
            reuse = True

        net = layers.stack(towersLogits)
        del towersLogits[:]

        net = layers.transpose(net, [1, 2, 3, 0, 4])
        net = layers.reshape(net, [-1, net.shape[1], net.shape[2], net.shape[3] * net.shape[4]])

        net = layers.fc(net, 512, name='fc5')

        net = layers.flatten(net, name='flatten')

        net = layers.fc(net, 4096, name='fc6')
        net = layers.fc(net, 4096, name='fc7')

        self.logits = layers.fc(net, self.dataset.num_classes, activation_fn=None, name='fc8')

        self.preds = layers.softmax(self.logits)

        self.loss = layers.reduce_mean(layers.softmax_cross_entropy(logits=self.logits, labels=self.Yoh))
        self.opt = layers.sgd(self.learning_rate)
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

        self.accuracy, self.precision, self.recall = self.createSummaries(self.Yoh, self.preds, self.loss,
                                                                              self.learning_rate)

    def mt_loop_low_memory(self, net, reuse=False):

        with tf.variable_scope('vgg_16', reuse=reuse):
            net = self.build_vgg16_low_memory(net, reuse)

        return net

    def build_vgg16_low_memory(self, net, reuse=False):

        with tf.variable_scope('conv1', reuse=reuse):
            net = layers.conv2d(net, 64, name='conv1_1', reuse=reuse)
            net = layers.conv2d(net, 64, name='conv1_2', reuse=reuse)
            net = layers.max_pool2d(net, [2, 2], 2, name='pool1')

        with tf.variable_scope('conv2', reuse=reuse):
            net = layers.conv2d(net, 128, name='conv2_1', reuse=reuse)
            net = layers.conv2d(net, 128, name='conv2_2', reuse=reuse)
            net = layers.max_pool2d(net, [2, 2], 2, name='pool2')

        with tf.variable_scope('conv3', reuse=reuse):
            net = layers.conv2d(net, 256, name='conv3_1', reuse=reuse)
            net = layers.conv2d(net, 256, name='conv3_2', reuse=reuse)
            net = layers.conv2d(net, 256, name='conv3_3', reuse=reuse)
            net = layers.max_pool2d(net, [2, 2], 2, name='pool3')

        with tf.variable_scope('conv4', reuse=reuse):
            net = layers.conv2d(net, 512, name='conv4_1', reuse=reuse)
            net = layers.conv2d(net, 512, name='conv4_2', reuse=reuse)
            net = layers.conv2d(net, 512, name='conv4_3', reuse=reuse)
            net = layers.max_pool2d(net, [2, 2], 2, name='pool4')

        with tf.variable_scope('conv5', reuse=reuse):
            net = layers.conv2d(net, 512, name='conv5_1', reuse=reuse)
            net = layers.conv2d(net, 512, name='conv5_2', reuse=reuse)
            net = layers.conv2d(net, 512, name='conv5_3', reuse=reuse)
            net = layers.max_pool2d(net, [2, 2], 2, name='pool5')

        return net

    def buildPartialModel(self):

        self.modelTower()
        self.modelLogits()
        self.modelOpt()

    def modelTower(self):

        self.is_training = tf.placeholder_with_default(True, [], name='is_training')
        self.towerImage = tf.placeholder(dtype=tf.float32, shape=[None, self.dataset.h, self.dataset.w, self.dataset.c])

        self.towerNet = self.vgg16(self.towerImage)

    def modelLogits(self):

        self.towerLogits = tf.placeholder(dtype=tf.float32, shape=[None, self.dataset.frames, 3, 3, 512])
        net = self.towerLogits

        net = layers.transpose(net, [0, 2, 3, 1, 4])
        net = layers.reshape(net, [-1, net.shape[1], net.shape[2], net.shape[3] * net.shape[4]])

        net = layers.fc(net, 512, name='fc5', weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE))
        net = layers.squeeze_and_excite2d(net, indexHeight=1, indexWidth=2, name='se5', filters=512)

        net = layers.flatten(net, name='flatten')

        net = layers.fc(net, 4096, name='fc6', weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE))
        net = layers.fc(net, 4096, name='fc7', weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE))

        self.logits = layers.fc(net, self.dataset.num_classes, activation_fn=None, name='fc8',
                                weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE))

    def modelOpt(self):

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = layers.decayLearningRate(LEARNING_RATE, self.global_step, DECAY_STEPS, DECAY_RATE)

        self.optLogits = tf.placeholder(dtype=tf.float32, shape=[None, self.dataset.num_classes])
        self.Y = tf.placeholder(dtype=tf.int32, shape=[None])
        self.Yoh = layers.toOneHot(self.Y, self.dataset.num_classes)

        self.preds = layers.softmax(self.optLogits)

        cross_entropy_loss = layers.reduce_mean(layers.softmax_cross_entropy(logits=self.optLogits, labels=self.Yoh))
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = cross_entropy_loss + REGULARIZER_SCALE * tf.reduce_sum(regularization_loss)

        self.opt = layers.adam(self.learning_rate)
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

        self.accuracy, self.precision, self.recall = self.createSummaries(self.Yoh, self.preds, self.loss,
                                                                          self.learning_rate)

    def trainLowMemory(self):

        print("TRAINING IS STARTING RIGHT NOW!")

        for epoch_num in range(1, MAX_EPOCHS + 1):

            epoch_start_time = time.time()
            currentSaveRate = SAVE_EVERY

            for step in range(self.dataset.num_batches_train):

                start_time = time.time()

                # get data
                batch_x, batch_y = self.sess.run([self.dataset.train_images, self.dataset.train_labels])

                # train towers logits
                logits = []
                for sequence_image in range(self.dataset.frames):
                    feed_dict = {self.towerImage: batch_x[:, sequence_image], self.is_training: True}
                    eval_tensors = self.towerNet
                    logits.append(self.sess.run(eval_tensors, feed_dict))
                logits = np.transpose(np.array(logits), [1, 0, 2, 3, 4])

                # logits
                feed_dict = {self.towerLogits: logits, self.is_training: False}
                eval_tensors = self.logits
                logits = self.sess.run(eval_tensors, feed_dict)

                # optimize
                feed_dict = {self.optLogits: logits, self.Y: batch_y, self.is_training: True}
                if (step + 1) * BATCH_SIZE % 5000 == 0:
                    eval_tensors = [self.loss, self.train_op, self.merged_summary_op]
                    loss_val, _, merged_ops = self.sess.run(eval_tensors, feed_dict=feed_dict)
                    self.summary_train_train_writer.add_summary(merged_ops, self.global_step.eval(session=self.sess))
                else:
                    eval_tensors = [self.loss, self.train_op]
                    loss_val, _ = self.sess.run(eval_tensors, feed_dict)

                duration = time.time() - start_time
                util.log_step(epoch_num, step, duration, loss_val, BATCH_SIZE, self.dataset.num_train_examples,
                              LOG_EVERY)

                if (step / self.dataset.num_batches_train) >= currentSaveRate:
                    self.saver.save(self.sess, self.ckptPrefix)
                    currentSaveRate += SAVE_EVERY

            epoch_time = time.time() - epoch_start_time
            print("Total epoch time training: {}".format(epoch_time))

            self.startValidationLowMemory(epoch_num, epoch_time)

        self.finishTraining()

    def startValidationLowMemory(self, epoch_num, epoch_time):

        print("EPOCH VALIDATION : ")

        train_loss, train_acc, train_pr, train_rec = self.validateLowMemory(self.dataset.num_train_examples, epoch_num, "train")
        valid_loss, valid_acc, valid_pr, valid_rec = self.validateLowMemory(self.dataset.num_valid_examples, epoch_num, "val")

        lr = self.sess.run([self.learning_rate])
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

        util.plot_training_progress(self.plot_data, self.dataset.name, self.name)

    def validateLowMemory(self, num_examples, epoch, dataset_type="train"):

        losses = []
        preds = np.ndarray((0,), dtype=np.int64)
        labels = np.ndarray((0,), dtype=np.int64)
        num_batches = num_examples // BATCH_SIZE
        if num_examples > 100000:
            num_batches = num_batches // 10

        for step in range(num_batches):

            # get data
            if dataset_type == 'train':
                batch_x, batch_y = self.sess.run([self.dataset.train_images, self.dataset.train_labels])
            else:
                batch_x, batch_y = self.sess.run([self.dataset.valid_images, self.dataset.valid_labels])

            # train towers logits
            logits = []
            for sequence_image in range(self.dataset.frames):
                feed_dict = {self.towerImage: batch_x[:, sequence_image], self.is_training: False}
                eval_tensors = self.towerNet
                logits.append(self.sess.run(eval_tensors, feed_dict))
            logits = np.transpose(np.array(logits), [1, 0, 2, 3, 4])

            # logits
            feed_dict = {self.towerLogits: logits, self.is_training: False}
            eval_tensors = self.logits
            logits = self.sess.run(eval_tensors, feed_dict)

            # get predictions
            feed_dict = {self.optLogits: logits, self.Y: batch_y, self.is_training: False}
            eval_tensors = [self.loss, self.preds]
            if (step + 1) * BATCH_SIZE % 5000 == 0:
                eval_tensors += [self.merged_summary_op]
                loss_val, predsEval, merged_ops = self.sess.run(eval_tensors, feed_dict=feed_dict)
                if dataset_type == 'train':
                    self.summary_valid_train_writer.add_summary(merged_ops, self.global_step.eval(session=self.sess))
                else:
                    self.summary_valid_valid_writer.add_summary(merged_ops, self.global_step.eval(session=self.sess))
            else:
                loss_val, predsEval = self.sess.run(eval_tensors, feed_dict)

            if (step + 1) * BATCH_SIZE % LOG_EVERY == 0:
                print("Evaluating {}, done: {}/{}".format(dataset_type, (step + 1) * BATCH_SIZE, num_batches * BATCH_SIZE))

            losses.append(loss_val)

            if preds.size == 0:
                labels = batch_y
                preds = predsEval
            else:
                labels = np.concatenate((labels, batch_y), axis=0)
                preds = np.concatenate((preds, predsEval), axis=0)

        total_loss = np.mean(losses)
        acc, pr, rec = util.acc_prec_rec_score(labels, np.argmax(preds, axis=1))
        print("Validation results -> {} error: epoch {} loss={} "
              "accuracy={} precision={} recall={}".format(dataset_type, epoch, total_loss, acc, pr, rec))

        return total_loss, acc, pr, rec

    def testLowMemory(self):

        losses = []
        preds = np.ndarray((0,), dtype=np.int64)
        labels = np.ndarray((0,), dtype=np.int64)

        for step in range(self.dataset.num_batches_test):

            # get data
            batch_x, batch_y = self.sess.run([self.dataset.test_images, self.dataset.test_labels])

            # train towers logits
            logits = []
            for sequence_image in range(self.dataset.frames):
                feed_dict = {self.towerImage: batch_x[:, sequence_image], self.is_training: False}
                eval_tensors = self.towerNet
                logits.append(self.sess.run(eval_tensors, feed_dict))
            logits = np.transpose(np.array(logits), [1, 0, 2, 3, 4])

            # logits
            feed_dict = {self.towerLogits: logits, self.is_training: False}
            eval_tensors = self.logits
            logits = self.sess.run(eval_tensors, feed_dict)

            # get predictions
            feed_dict = {self.optLogits: logits, self.Y: batch_y, self.is_training: False}
            eval_tensors = [self.loss, self.preds]
            if (step + 1) * BATCH_SIZE % 5000 == 0:
                eval_tensors += [self.merged_summary_op]
                loss_val, predsEval, merged_ops = self.sess.run(eval_tensors, feed_dict=feed_dict)
                self.summary_test_writer.add_summary(merged_ops, self.global_step.eval(session=self.sess))
            else:
                loss_val, predsEval = self.sess.run(eval_tensors, feed_dict)

            if (step + 1) * BATCH_SIZE % LOG_EVERY == 0:
                print("Evaluating {}, done: {}/{}".format('test', (step + 1) * BATCH_SIZE, self.dataset.num_test_examples))

            losses.append(loss_val)

            if preds.size == 0:
                labels = batch_y
                preds = predsEval
            else:
                labels = np.concatenate((labels, batch_y), axis=0)
                preds = np.concatenate((preds, predsEval), axis=0)

        total_loss = np.mean(losses)
        acc, pr, rec = util.acc_prec_rec_score(labels, np.argmax(preds, axis=1))
        prAtTop10, top5CorrectWords, top5IncorrectWords = util.testStatistics(labels, preds)
        util.write_test_results(total_loss, acc, pr, rec, prAtTop10, top5CorrectWords, top5IncorrectWords,
                                self.dataset.name, self.name)
        print("Validation results -> {} error: epoch {} loss={} accuracy={} precision={} recall={} prAtTop10={}".format(
            'test', '1', total_loss, acc, pr, rec, prAtTop10))

    def train(self):

        print("TRAINING IS STARTING RIGHT NOW!")

        for epoch_num in range(1, MAX_EPOCHS + 1):

            epoch_start_time = time.time()
            currentSaveRate = SAVE_EVERY

            for step in range(self.dataset.num_batches_train):

                start_time = time.time()

                batch_x, batch_y = self.sess.run([self.dataset.train_images, self.dataset.train_labels])

                feed_dict = {self.is_training: True, self.X: batch_x, self.Y: batch_y}
                eval_tensors = [self.loss, self.train_op]
                if (step + 1) * BATCH_SIZE % LOG_EVERY == 0:
                    eval_tensors += [self.merged_summary_op]

                eval_ret = self.sess.run(eval_tensors, feed_dict=feed_dict)
                eval_ret = dict(zip(eval_tensors, eval_ret))

                loss_val = eval_ret[self.loss]

                if self.merged_summary_op in eval_tensors:
                    self.summary_train_train_writer.add_summary(eval_ret[self.merged_summary_op], self.global_step.eval(session=self.sess))

                duration = time.time() - start_time
                util.log_step(epoch_num, step, duration, loss_val, BATCH_SIZE, self.dataset.num_train_examples, LOG_EVERY)

                if (step / self.dataset.num_batches_train) >= (currentSaveRate):
                    self.saver.save(self.sess, self.ckptPrefix)
                    currentSaveRate += SAVE_EVERY

            epoch_time = time.time() - epoch_start_time
            print("Total epoch time training: {}".format(epoch_time))

            self.startValidation(epoch_num, epoch_time)

        self.finishTraining()

    def startValidation(self, epoch_num, epoch_time):

        print("EPOCH VALIDATION : ")

        train_loss, train_acc, train_pr, train_rec = self.validate(self.dataset.num_train_examples, epoch_num, "train")
        valid_loss, valid_acc, valid_pr, valid_rec = self.validate(self.dataset.num_valid_examples, epoch_num, "val")

        lr = self.sess.run([self.learning_rate])
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

        util.plot_training_progress(self.plot_data, self.dataset.name, self.name)

    def validate(self, num_examples, epoch, dataset_type="train"):

        losses = []
        preds = np.ndarray((0,), dtype=np.int64)
        labels = np.ndarray((0,), dtype=np.int64)
        num_batches = num_examples // BATCH_SIZE
        if num_examples > 100000:
            num_batches = num_batches // 10

        for step in range(num_batches):

            eval_tensors = [self.Yoh, self.preds, self.loss]
            if (step + 1) * BATCH_SIZE % LOG_EVERY == 0:
                print("Evaluating {}, done: {}/{}".format(dataset_type, (step + 1) * BATCH_SIZE, num_batches * BATCH_SIZE))
                eval_tensors += [self.merged_summary_op]

            if dataset_type == 'train':
                batch_x, batch_y = self.sess.run([self.dataset.train_images, self.dataset.train_labels])
            else:
                batch_x, batch_y = self.sess.run([self.dataset.valid_images, self.dataset.valid_labels])

            feed_dict = {self.is_training: False, self.X: batch_x, self.Y: batch_y}

            eval_ret = self.sess.run(eval_tensors, feed_dict=feed_dict)
            eval_ret = dict(zip(eval_tensors, eval_ret))

            if self.merged_summary_op in eval_tensors:
                if dataset_type == 'train':
                    self.summary_valid_train_writer.add_summary(eval_ret[self.merged_summary_op], self.global_step.eval(session=self.sess))
                else:
                    self.summary_valid_valid_writer.add_summary(eval_ret[self.merged_summary_op], self.global_step.eval(session=self.sess))

            losses.append(eval_ret[self.loss])

            if preds.size == 0:
                labels = np.argmax(eval_ret[self.Yoh], axis=1)
                preds = np.argmax(eval_ret[self.preds], axis=1)
            else:
                labels = np.concatenate((labels, np.argmax(eval_ret[self.Yoh], axis=1)), axis=0)
                preds = np.concatenate((preds, np.argmax(eval_ret[self.preds], axis=1)), axis=0)

        total_loss = np.mean(losses)
        acc, pr, rec = util.acc_prec_rec_score(labels, preds)
        print("Validation results -> {} error: epoch {} loss={} accuracy={} precision={} recall={}".format(dataset_type, epoch, total_loss, acc, pr, rec))

        return total_loss, acc, pr, rec

    def test(self):

        print('STARTED TESTING EVALUATION!')

        losses = []
        preds = np.ndarray((0,), dtype=np.int64)
        labels = np.ndarray((0,), dtype=np.int64)

        for step in range(self.dataset.num_batches_test):

            eval_tensors = [self.Yoh, self.preds, self.loss]
            if (step + 1) * BATCH_SIZE % LOG_EVERY == 0:
                print("Evaluating {}, done: {}/{}".format('test', (step + 1) * BATCH_SIZE, self.dataset.num_test_examples))
                eval_tensors += [self.merged_summary_op]

            batch_x, batch_y = self.sess.run([self.dataset.test_images, self.dataset.test_labels])

            feed_dict = {self.is_training: False, self.X: batch_x, self.Y: batch_y}

            eval_ret = self.sess.run(eval_tensors, feed_dict=feed_dict)
            eval_ret = dict(zip(eval_tensors, eval_ret))

            if self.merged_summary_op in eval_tensors:
                self.summary_test_writer.add_summary(eval_ret[self.merged_summary_op], self.global_step.eval(session=self.sess))

            losses.append(eval_ret[self.loss])

            if preds.size == 0:
                labels = np.argmax(eval_ret[self.Yoh], axis=1)
                preds = eval_ret[self.preds]
            else:
                labels = np.concatenate((labels, np.argmax(eval_ret[self.Yoh], axis=1)), axis=0)
                preds = np.concatenate((preds, eval_ret[self.preds]), axis=0)

        total_loss = np.mean(losses)
        acc, pr, rec = util.acc_prec_rec_score(labels, np.argmax(preds, axis=1))
        prAtTop10, top5CorrectWords, top5IncorrectWords = util.testStatistics(labels, preds)
        util.write_test_results(total_loss, acc, pr, rec, prAtTop10, top5CorrectWords, top5IncorrectWords, self.dataset.name, self.name)
        print("Validation results -> {} error: epoch {} loss={} accuracy={} precision={} recall={} prAtTop10={}".format('test', '1', total_loss, acc, pr, rec, prAtTop10))

    def initConfig(self):

        print("INITIALIZING CONFIGURATION VARIABLES")
        self.name = 'mt_vgg16'
        self.checkpoint_dir = config.config['checkpoint_root_dir'] # direktorij gdje se nalazi checkpoint
        self.summary_dir = config.config['summary_root_dir']

    def createSummaries(self, labelsOH, predsOH, loss, learning_rate):

        labels = layers.onehot_to_class(labelsOH)
        preds = layers.onehot_to_class(predsOH)
        acc = layers.accuracy(labels, preds)
        prec = layers.precision(labels, preds)
        rec = layers.recall(labels, preds)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', acc[1])
        tf.summary.scalar('precision', prec[1])
        tf.summary.scalar('recall', rec[1])

        return acc, prec, rec

    def initSummaries(self):

        trainSummaryDir = os.path.join(self.summary_dir, self.dataset.name, self.name, 'train')
        trainEvalSummaryDir = os.path.join(self.summary_dir, self.dataset.name, self.name, 'trainEval')
        valEvalSummaryDir = os.path.join(self.summary_dir, self.dataset.name, self.name, 'valEval')
        testSummaryDir = os.path.join(self.summary_dir, self.dataset.name, self.name, 'test')

        self.merged_summary_op = tf.summary.merge_all()
        self.summary_train_train_writer = tf.summary.FileWriter(trainSummaryDir)
        self.summary_valid_train_writer = tf.summary.FileWriter(trainEvalSummaryDir)
        self.summary_valid_valid_writer = tf.summary.FileWriter(valEvalSummaryDir)
        self.summary_test_writer = tf.summary.FileWriter(testSummaryDir)

    def createSession(self):

        print('CREATING SESSION FOR: {} AND MODEL: {}'.format(self.dataset.name, self.name))

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        gpu_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

        self.sess = tf.Session(config=gpu_config)
        self.sess.as_default()

        self.initializeOrRestore()

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def initDataReaders(self):
        if DATASET_TO_USE == 'lrw':
            self.dataset = lrw_dataset.LrwDataset(batch_size=BATCH_SIZE)
        elif DATASET_TO_USE == 'road':
            self.dataset = road_dataset.RoadDataset(batch_size=BATCH_SIZE)
        elif DATASET_TO_USE == 'mnist':
            self.dataset = mnist_dataset.MnistDataset(batch_size=BATCH_SIZE)
        elif DATASET_TO_USE == 'mnist_original':
            self.dataset = mnist_original_dataset.MnistOriginalDataset(batch_size=BATCH_SIZE)
        elif DATASET_TO_USE == 'cifar':
            self.dataset = cifar_dataset.CifarDataset(batch_size=BATCH_SIZE)
        else:
            print("NIJE ODABRAN DATASET!")

    def finishTraining(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()

    def createPlotDataVars(self):
        self.plot_data = {
            'train_loss': [], 'train_acc': [], 'train_pr': [], 'train_rec': [],
            'valid_loss': [], 'valid_acc': [], 'valid_pr': [], 'valid_rec': [],
            'lr': [], 'epoch_time': []
        }
        print("INITIALIZED PLOT DATA!")

    def addGraphToSummaries(self):
        self.summary_train_train_writer.add_graph(self.sess.graph)
        self.summary_valid_train_writer.add_graph(self.sess.graph)
        self.summary_valid_valid_writer.add_graph(self.sess.graph)
        self.summary_test_writer.add_graph(self.sess.graph)

    def initializeOrRestore(self):

        self.ckptDir = os.path.join(self.checkpoint_dir, self.dataset.name)
        self.ckptPrefix = os.path.join(self.ckptDir, self.name, self.name)
        vgg_ckpt_file = os.path.join(self.ckptDir, 'vgg_16', 'vgg_16.ckpt')
        mt_ckpt_file = layers.latest_checkpoint(os.path.join(self.ckptDir, 'mt'))
        # ckpt_file = layers.latest_checkpoint(os.path.join(self.ckptDir, 'vgg_16', 'vgg_16.ckpt'))
        globalVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        if vgg_ckpt_file is not None and tf.train.checkpoint_exists(vgg_ckpt_file):
            varsInCkpt, varsNotInCkpt = layers.scan_checkpoint_for_vars(vgg_ckpt_file, globalVars)
            if len(varsInCkpt) != 0:
                restorationSaver = tf.train.Saver(varsInCkpt)
                self.sess.run(tf.report_uninitialized_variables(var_list=varsInCkpt))
                restorationSaver.restore(self.sess, vgg_ckpt_file)
        else:
            varsNotInCkpt = globalVars

        if mt_ckpt_file is not None and tf.train.checkpoint_exists(mt_ckpt_file):
            varsInCkpt, varsNotInCkpt = layers.scan_checkpoint_for_vars(mt_ckpt_file, varsNotInCkpt)
            varsInCkpt, varsNotInCkpt = layers.replaceVarInListsByName(varsInCkpt, varsNotInCkpt, 'fc6')
            if len(varsInCkpt) != 0:
                restorationSaver = tf.train.Saver(varsInCkpt)
                self.sess.run(tf.report_uninitialized_variables(var_list=varsInCkpt))
                restorationSaver.restore(self.sess, mt_ckpt_file)
        else:
            varsNotInCkpt = globalVars

        self.saver = tf.train.Saver()
        self.sess.run(tf.group(tf.variables_initializer(varsNotInCkpt), tf.local_variables_initializer()))

model = MT_VGG16()
model.trainLowMemory()