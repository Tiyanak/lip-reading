from src.utils import util
from src import config
from src.cnn import layers
import tensorflow as tf
import time
import os
import numpy as np
from src.dataset import lrw_dataset, mnist_original_dataset, road_dataset, mnist_dataset, cifar_dataset

DATASET_TO_USE = 'lrw'
LOG_EVERY = 1
SAVE_EVERY = 0.2
DECAY_STEPS = 10000 # broj koraka za smanjivanje stope ucenja
DECAY_RATE = 0.96 # rate smanjivanja stope ucenja
REGULARIZER_SCALE = 0.1 # faktor regularizacije
LEARNING_RATE = 5e-4
BATCH_SIZE = 1
MAX_EPOCHS = 10

class MT_3_ResNet:

    def __init__(self):

        self.initConfig()  # postavi globalne varijable
        self.createPlotDataVars()  # kreiraj mapu za zapisivanje metrike (pdf + csv)
        self.initDataReaders()  # postavi dataset (reader za tfrecordse)
        self.createModel()  # kreiraj duboki model (tensorski graf)
        self.initSummaries()  # kreiraj summary writere
        self.createSession()  # kreiraj saver i session bez inicijalizacije varijabli
        self.addGraphToSummaries()  # u summary writere dodaj graf

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
            net = self.mt3_loop(self.X[:, sequence_image], reuse)
            towersLogits.append(net)
            reuse = True

        net = layers.stack(towersLogits)
        del towersLogits[:]

        net = layers.transpose(net, [1, 0, 2, 3, 4])

        bn_params = {'decay': 0.999, 'center': True, 'scale': True, 'epsilon': 0.001,
                       'updates_collections': None, 'is_training': self.is_training}

        net = layers.conv3d(net, 256, kernel_size=3, stride=2, padding='valid', name='conv2',
                            weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE), normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
        net = layers.max_pool3d(net, 3, 2, name='pool2')

        net = layers.conv3d(net, filters=512, kernel_size=3, padding='SAME', stride=1, name='conv3',
                            weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE), normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)

        net = layers.conv3d(net, filters=512, kernel_size=3, padding='SAME', stride=1, name='conv4',
                            weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE), normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)

        net = layers.transpose(net, [0, 2, 3, 1, 4])
        net = layers.reshape(net, [-1, net.shape[1], net.shape[2], net.shape[3] * net.shape[4]])

        net = layers.conv2d(net, filters=512, kernel_size=3, padding='SAME', stride=1, name='conv5',
                            weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE), normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
        net = layers.max_pool2d(net, 3, 2, padding='VALID', name='max_pool5')

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
        self.learning_rate = layers.decayLearningRate(LEARNING_RATE, self.global_step, DECAY_STEPS, DECAY_RATE)
        self.opt = layers.adam(self.learning_rate)
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

        self.accuracy, self.precision, self.recall = self.createSummaries(self.Yoh, self.preds, self.loss, self.learning_rate)

    def mt3_loop(self, net, reuse=False):

        with tf.variable_scope('mt_loop', reuse=reuse):
            net = self.build_resnet_18(net, reuse)

        return net

    def build_resnet_18(self, net, reuse=False):

        bn_params = {'decay': 0.999, 'center': True, 'scale': True, 'epsilon': 0.001, 'updates_collections': None, 'is_training': self.is_training}

        net = layers.conv2d(net, filters=64, kernel_size=7, stride=2, padding='SAME', name='resnet_conv1', reuse=reuse,
                            weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE), normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
        net = layers.max_pool2d(net, kernel_size=3, stride=2, padding='same')

        net = self.res_block(net, 64, 3, reuse=reuse, name='resnet_conv2a')
        net = self.res_block(net, 64, 3, reuse=reuse, name='resnet_conv2b')

        net = self.res_block_first(net, 128, 3, reuse=reuse, name='resnet_conv3a')
        net = self.res_block(net, 128, 3,reuse=reuse, name='resnet_conv3b')

        net = self.res_block_first(net, 256, 3, reuse=reuse,name='resnet_conv4a')
        net = self.res_block(net, 256, 3, reuse=reuse, name='resnet_conv4b')

        net = self.res_block_first(net, 512, 3, reuse=reuse, name='resnet_conv5a')
        net = self.res_block(net, 512, 3, reuse=reuse, name='resnet_conv5b')

        net = layers.avg_pool2d(net, kernel_size=7, stride=1)

        return net

    def res_block(self, net, filters=64, kernel=3, name='resnet_convX', reuse=False):

        bn_params = {'decay': 0.999, 'center': True, 'scale': True, 'epsilon': 0.001, 'updates_collections': None,
                     'is_training': self.is_training}

        tmp_logits = net
        net = layers.conv2d(net, filters=filters, kernel_size=kernel, stride=1, padding='same', name=name + '_1', reuse=reuse,
                            weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE), normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)

        net = layers.conv2d(net, filters=filters, kernel_size=kernel, stride=1, padding='same', name=name + '_2', reuse=reuse, activation_fn=None,
                            weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE), normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)
        net = layers.add(net, tmp_logits)
        net = layers.relu(net)

        return net

    def res_block_first(self, net, filters=64, kernel=3, stride=1, reuse=False, name='resnet_convX'):

        bn_params = {'decay': 0.999, 'center': True, 'scale': True, 'epsilon': 0.001, 'updates_collections': None,
                     'is_training': self.is_training}

        tmp_logits = layers.conv2d(net, filters=filters, kernel_size=kernel, stride=stride, padding='same', activation_fn=None, name=name + '_shortcut', reuse=reuse,
                                   weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE), normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)

        net = layers.conv2d(net, filters=filters, kernel_size=kernel, stride=stride, padding='same', name=name + '_1', reuse=reuse,
                            weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE), normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)

        net = layers.conv2d(net, filters=filters, kernel_size=kernel, stride=stride, padding='same', name=name + '_2', reuse=reuse, activation_fn=None,
                            weights_regularizer=layers.l2_regularizer(REGULARIZER_SCALE), normalizer_fn=layers.batchNormalization, normalizer_params=bn_params)

        net = layers.add(net, tmp_logits)
        net = layers.relu(net)

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
        self.name = 'mt_3_resnet'
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

        self.ckptDir = os.path.join(self.checkpoint_dir, self.dataset.name, self.name)
        self.ckptPrefix = os.path.join(self.ckptDir, self.name + ".ckpt")

        globalVars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        ckpt_file = layers.latest_checkpoint(self.ckptDir, "checkpoint")
        varsInCkpt, varsNotInCkpt = layers.scan_checkpoint_for_vars(ckpt_file, globalVars)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
        gpu_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)

        opsInCkpt = tf.report_uninitialized_variables(var_list=varsInCkpt)
        opsNotInCkpt = tf.variables_initializer(varsNotInCkpt)

        restorationSaver = tf.train.Saver(varsInCkpt)
        self.saver = tf.train.Saver()

        self.sess = tf.Session(config=gpu_config)
        self.sess.as_default()

        self.sess.run(opsInCkpt)
        if tf.train.checkpoint_exists(ckpt_file):
            self.restored = restorationSaver.restore(self.sess, ckpt_file)

        self.sess.run(tf.group(opsNotInCkpt, tf.local_variables_initializer()))

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def initDataReaders(self):
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

    def addGraphToSummaries(self):
        self.summary_train_train_writer.add_graph(self.sess.graph)
        self.summary_valid_train_writer.add_graph(self.sess.graph)
        self.summary_valid_valid_writer.add_graph(self.sess.graph)
        self.summary_test_writer.add_graph(self.sess.graph)

model = MT_3_ResNet()
model.train()