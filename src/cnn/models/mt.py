from src.utils import util
import config
from src.cnn import layers
import tensorflow as tf
import time
import os
import numpy as np
from src.dataset import lrw_dataset, mnist_original_dataset, road_dataset, mnist_dataset, cifar_dataset

DATASET_TO_USE = 'lrw'
LOG_EVERY = 200 if DATASET_TO_USE == 'road' else 1000
SAVE_EVERY = 0.2
DECAY_STEPS = 10000 # broj koraka za smanjivanje stope ucenja
DECAY_RATE = 0.96 # rate smanjivanja stope ucenja
REGULARIZER_SCALE = 0.1 # faktor regularizacije
LEARNING_RATE = 5e-4
BATCH_SIZE = 20
MAX_EPOCHS = 10

class MT:

    def __init__(self):

        self.initConfig() # postavi globalne varijable
        self.createPlotDataVars() # kreiraj mapu za zapisivanje metrike (pdf + csv)
        self.initDataReaders() # postavi dataset (reader za tfrecordse)
        with tf.device("/GPU:0"):
            self.createModel() # kreiraj duboki model (tensorski graf)
        self.createSession() # kreiraj saver i session bez inicijalizacije varijabli
        self.initSummaries() # kreiraj summarije
        self.initSessionVariables() # inicijaliziraj sve varijable u grafu
        self.createRestorer() # vrati zadnji checkpoint u slucaju da postoji

    def createModel(self):

        print("CREATING MODEL")

        self.global_step = tf.Variable(0, trainable=False)
        self.is_training = tf.placeholder_with_default(True, [], name='is_training')
        self.dataset_type = tf.placeholder_with_default('train_train', [], name='dataset_type')

        dataset_val = tf.placeholder_with_default('val', [], name='dataset_val')
        dataset_test = tf.placeholder_with_default('test', [], name='dataset_test')

        if dataset_val.__eq__(self.dataset_type):
            self.X = tf.cast(self.dataset.valid_images, dtype=tf.float32)
            self.Yoh = layers.toOneHot(self.dataset.valid_labels, self.dataset.num_classes)
        elif dataset_test.__eq__(self.dataset_type):
            self.X = tf.cast(self.dataset.test_images, dtype=tf.float32)
            self.Yoh = layers.toOneHot(self.dataset.test_labels, self.dataset.num_classes)
        else:
            self.X = tf.cast(self.dataset.train_images, dtype=tf.float32)
            self.Yoh = layers.toOneHot(self.dataset.train_labels, self.dataset.num_classes)

        reuse = None
        towersLogits = []
        for sequence_image in range(self.dataset.frames):
            net = self.mt_loop(self.X[:, sequence_image], reuse)
            towersLogits.append(net)
            reuse = True

        net = layers.stack(towersLogits)
        del towersLogits[:]

        net = layers.transpose(net, [1, 2, 3, 0, 4])
        net = layers.reshape(net, [-1, net.shape[1], net.shape[2], net.shape[3] * net.shape[4]])

        bn_params = {'decay': 0.999, 'center': True, 'scale': True, 'epsilon': 0.001,
                      'updates_collections': None, 'is_training': self.is_training}
        net = layers.conv2d(net, 96, kernel_size=1, name='conv1d', padding='valid',
                            activation_fn=layers.relu, weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                            weights_initializer=layers.xavier_initializer(), normalizer_fn=layers.batchNormalization,
                            normalizer_params=bn_params)

        net = layers.conv2d(net, 256, kernel_size=3, stride=2, padding='valid', name='conv2',
                            activation_fn=layers.relu, weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                            weights_initializer=layers.xavier_initializer(), normalizer_fn=layers.batchNormalization,
                            normalizer_params=bn_params)
        net = layers.max_pool2d(net, 3, 2, name='pool2')

        net = layers.conv2d(net, filters=512, kernel_size=[3, 3], padding='SAME', stride=1, name='conv3',
                            activation_fn=layers.relu, weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                            weights_initializer=layers.xavier_initializer(), normalizer_fn=layers.batchNormalization,
                            normalizer_params=bn_params)

        net = layers.conv2d(net, filters=512, kernel_size=[3, 3], padding='SAME', stride=1, name='conv4',
                            activation_fn=layers.relu, weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                            weights_initializer=layers.xavier_initializer(), normalizer_fn=layers.batchNormalization,
                            normalizer_params=bn_params)

        net = layers.conv2d(net, filters=512, kernel_size=[3, 3], padding='SAME', stride=1, name='conv5',
                            activation_fn=layers.relu, weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                            weights_initializer=layers.xavier_initializer(), normalizer_fn=layers.batchNormalization,
                            normalizer_params=bn_params)
        net = layers.max_pool2d(net, [3, 3], 2, padding='VALID', name='max_pool5')

        net = layers.flatten(net, name='flatten')

        net = layers.fc(net, 4096, name='fc6', weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                        normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
        net = layers.fc(net, 4096, name='fc7', weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                        normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
        net = layers.fc(net, self.dataset.num_classes, activation_fn=None, name='fc8',
                        weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE))

        self.logits = net
        self.preds = layers.softmax(self.logits)

        self.loss = layers.reduce_mean(layers.softmax_cross_entropy(logits=self.logits, labels=self.Yoh))
        self.regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([self.loss] + self.regularization_loss, name='total_loss')

        self.learning_rate = layers.decayLearningRate(self.starter_learning_rate, self.global_step, DECAY_STEPS, DECAY_RATE)

        self.opt = layers.adam(self.learning_rate)
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

        self.accuracy, self.precision, self.recall = self.createSummaries(self.Yoh, self.preds, self.loss, self.learning_rate, self.regularization_loss)

    def mt_loop(self, net, reuse=False):

        bn1_params = {'name' : 'bn1', 'decay': 0.999, 'center': True, 'scale': True, 'epsilon': 0.001, 'updates_collections': None, 'is_training': self.is_training}

        with tf.variable_scope('mt_loop', reuse=reuse):

            net = layers.conv2d(net, 48, kernel_size=3, name='conv1', reuse=reuse, stride=2, padding='valid',
                                activation_fn=layers.relu, weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE),
                                weights_initializer=layers.xavier_initializer(), normalizer_fn=layers.batchNormalization, normalizer_params=bn1_params)
            net = layers.max_pool2d(net, 3, 2, name='pool1')

        return net

    def train(self):

        print("TRAINING IS STARTING RIGHT NOW!")

        for epoch_num in range(1, MAX_EPOCHS + 1):

            epoch_start_time = time.time()
            currentSaveRate = SAVE_EVERY

            for step in range(self.dataset.num_batches_train):

                start_time = time.time()

                feed_dict = {self.is_training: True, self.dataset_type: 'train'}
                eval_tensors = [self.loss, self.train_op]
                if (step + 1) * BATCH_SIZE % LOG_EVERY == 0:
                    eval_tensors += [self.merged_summary_op, self.accuracy, self.precision, self.recall]

                eval_ret = self.sess.run(eval_tensors, feed_dict=feed_dict)
                eval_ret = dict(zip(eval_tensors, eval_ret))

                loss_val = eval_ret[self.loss]

                if self.merged_summary_op in eval_tensors:
                    self.summary_train_train_writer.add_summary(eval_ret[self.merged_summary_op], self.global_step.eval(session=self.sess))

                duration = time.time() - start_time
                util.log_step(epoch_num, step, duration, loss_val, BATCH_SIZE, self.dataset.num_train_examples, LOG_EVERY)

                if (step / self.dataset.num_batches_train) >= (currentSaveRate):
                    self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.dataset.name, self.name + '.ckpt'))
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

        # self.test()

    def validate(self, num_examples, epoch, dataset_type="train"):

        losses = []
        preds = np.ndarray((0,), dtype=np.int64)
        labels = np.ndarray((0,), dtype=np.int64)
        num_batches = num_examples // BATCH_SIZE

        for step in range(num_batches):

            eval_tensors = [self.Yoh, self.preds, self.loss, self.accuracy, self.precision, self.recall]
            if (step + 1) * BATCH_SIZE % LOG_EVERY == 0:
                print("Evaluating {}, done: {}/{}".format(dataset_type, (step + 1) * BATCH_SIZE, num_batches * BATCH_SIZE))
                eval_tensors += [self.merged_summary_op]

            feed_dict = {self.is_training: False, self.dataset_type: dataset_type}

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

            eval_tensors = [self.Yoh, self.preds, self.loss, self.accuracy, self.precision, self.recall]
            if (step + 1) * BATCH_SIZE % LOG_EVERY == 0:
                print("Evaluating {}, done: {}/{}".format('test', (step + 1) * BATCH_SIZE, self.dataset.num_test_examples))
                eval_tensors += [self.merged_summary_op]

            feed_dict = {self.is_training: False, self.dataset_type: 'test'}

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
        self.name = 'mt'
        self.starter_learning_rate = LEARNING_RATE
        self.checkpoint_dir = config.config['checkpoint_root_dir'] # direktorij gdje se nalazi checkpoint
        self.summary_dir = config.config['summary_root_dir']

    def createSummaries(self, labelsOH, predsOH, loss, learning_rate, regularization_loss):

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
        tf.summary.scalar('regularization_loss', regularization_loss[1])

        return acc, prec, rec

    def initSummaries(self):

        trainSummaryDir = os.path.join(self.summary_dir, self.dataset.name, self.name, 'train')
        trainEvalSummaryDir = os.path.join(self.summary_dir, self.dataset.name, self.name, 'trainEval')
        valEvalSummaryDir = os.path.join(self.summary_dir, self.dataset.name, self.name, 'valEval')
        testSummaryDir = os.path.join(self.summary_dir, self.dataset.name, self.name, 'test')

        self.merged_summary_op = tf.summary.merge_all()
        self.summary_train_train_writer = tf.summary.FileWriter(trainSummaryDir, self.sess.graph)
        self.summary_valid_train_writer = tf.summary.FileWriter(trainEvalSummaryDir, self.sess.graph)
        self.summary_valid_valid_writer = tf.summary.FileWriter(valEvalSummaryDir, self.sess.graph)
        self.summary_test_writer = tf.summary.FileWriter(testSummaryDir, self.sess.graph)

    def createSession(self):

        print('CREATING SESSION FOR: {} AND MODEL: {}'.format(self.dataset.name, self.name))
        self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        gpu_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=gpu_config)
        self.sess.as_default()

    def initSessionVariables(self):

        print("INITIALIZING SESSION VARIABLES")
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def initDataReaders(self):
        with tf.device("/CPU:0"):
            if DATASET_TO_USE == 'lrw':
                self.dataset = lrw_dataset.LrwDataset()
            elif DATASET_TO_USE == 'road':
                self.dataset = road_dataset.RoadDataset()
            elif DATASET_TO_USE == 'mnist':
                self.dataset = mnist_dataset.MnistDataset()
            elif DATASET_TO_USE == 'mnist_original':
                self.dataset = mnist_original_dataset.MnistOriginalDataset()
            elif DATASET_TO_USE == 'cifar':
                self.dataset = cifar_dataset.CifarDataset()
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

    def createRestorer(self):
        if os.path.exists(os.path.join(self.checkpoint_dir, self.name + '.ckpt.meta')):
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, self.dataset.name, self.name + '.ckpt'))

model = MT()
model.train()