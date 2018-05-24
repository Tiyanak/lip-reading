from utils import config, util
from cnn import layers
import tensorflow as tf
import time
import os
import numpy as np
from dataset import lrw_dataset, mnist_dataset, road_dataset

DATASET_TO_USE = 'lrw'

class EF:

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

        self.initializer = layers.xavier_initializer()
        self.regularizer = layers.l2_regularizer(self.regularizer_scale)
        self.activation_fn = layers.relu

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

        net = self.X

        net = layers.rgb_to_grayscale(net)
        net = tf.transpose(net, [0, 2, 3, 1, 4])
        net = tf.reshape(net, [-1, net.shape[1], net.shape[2], net.shape[3] * net.shape[4]])

        net = layers.conv2d(net, filters=96, kernel_size=[3, 3], padding='VALID', stride=2, name='conv1',
                           activation_fn=self.activation_fn, weights_initializer=self.initializer, weights_regularizer=self.regularizer)
        net = layers.batchNormalization(input=net, is_training=self.is_training, name='bn1')
        net = tf.nn.relu(net)
        net = layers.max_pool2d(net, [3, 3], 2, padding='VALID', name='max_pool1')

        net = layers.conv2d(net, filters=256, kernel_size=[3, 3], padding='VALID', stride=2, name='conv2',
                            activation_fn=self.activation_fn, weights_initializer=self.initializer, weights_regularizer=self.regularizer)
        net = layers.batchNormalization(input=net, is_training=self.is_training, name='bn2')
        net = tf.nn.relu(net)
        net = layers.max_pool2d(net, [3, 3], 2, padding='VALID', name='max_pool2')

        net = layers.conv2d(net, filters=512, kernel_size=[3, 3], padding='SAME', stride=1, name='conv3',
                            activation_fn=self.activation_fn, weights_initializer=self.initializer, weights_regularizer=self.regularizer)

        net = layers.conv2d(net, filters=512, kernel_size=[3, 3], padding='SAME', stride=1, name='conv4',
                            activation_fn=self.activation_fn, weights_initializer=self.initializer,
                            weights_regularizer=self.regularizer)

        net = layers.conv2d(net, filters=512, kernel_size=[3, 3], padding='SAME', stride=1, name='conv5',
                            activation_fn=self.activation_fn, weights_initializer=self.initializer,
                            weights_regularizer=self.regularizer)
        net = layers.max_pool2d(net, [3, 3], 2, padding='VALID', name='max_pool2')

        net = layers.flatten(net, name='flatten')

        net = layers.fc(net, 4096, self.activation_fn, name='fc6', weights_initializer=self.initializer, weights_regularizer=self.regularizer)
        net = layers.fc(net, 4096, self.activation_fn, name='fc7', weights_initializer=self.initializer, weights_regularizer=self.regularizer)
        net = layers.fc(net, self.dataset.num_classes, name='fc8', weights_initializer=self.initializer, weights_regularizer=self.regularizer)

        self.logits = net
        self.preds = layers.softmax(self.logits)

        self.regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = layers.reduce_mean(layers.softmax_cross_entropy(logits=self.logits, labels=self.Yoh)) + tf.reduce_sum(self.regularization_loss) * self.regularizer_scale

        self.learning_rate = layers.decayLearningRate(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate)

        self.accuracy, self.precision, self.recall = self.createSummaries(self.Yoh, self.preds, self.loss, self.learning_rate, self.regularization_loss)

        self.opt = layers.adam(self.learning_rate)
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

    def train(self):

        print("TRAINING IS STARTING RIGHT NOW!")

        for epoch_num in range(1, self.max_epochs + 1):

            epoch_start_time = time.time()
            currentSaveRate = self.save_every

            for step in range(self.dataset.num_batches_train):

                start_time = time.time()

                feed_dict = {self.is_training: True, self.dataset_type: 'train'}
                eval_tensors = [self.loss, self.train_op]
                if (step + 1) * self.batch_size % self.log_every == 0:
                    eval_tensors += [self.merged_summary_op, self.accuracy, self.precision, self.recall]

                eval_ret = self.sess.run(eval_tensors, feed_dict=feed_dict)
                eval_ret = dict(zip(eval_tensors, eval_ret))

                loss_val = eval_ret[self.loss]

                if self.merged_summary_op in eval_tensors:
                    self.summary_train_train_writer.add_summary(eval_ret[self.merged_summary_op], self.global_step.eval(session=self.sess))

                duration = time.time() - start_time
                util.log_step(epoch_num, step, duration, loss_val, self.batch_size, self.dataset.num_train_examples, self.log_every)

                if (step / self.dataset.num_batches_train) >= (currentSaveRate):
                    self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.name + '.ckpt'))
                    currentSaveRate += self.save_every

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

        self.test()

    def validate(self, num_examples, epoch, dataset_type="train"):

        losses = []
        preds = np.ndarray((0,), dtype=np.int64)
        labels = np.ndarray((0,), dtype=np.int64)
        num_batches = num_examples // self.batch_size

        for step in range(num_batches):

            eval_tensors = [self.Yoh, self.preds, self.loss, self.accuracy, self.precision, self.recall]
            if (step + 1) * self.batch_size % self.log_every == 0:
                print("Evaluating {}, done: {}/{}".format(dataset_type, (step + 1) * self.batch_size, num_batches * self.batch_size))
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

            eval_tensors = [self.preds, self.loss, self.accuracy, self.precision, self.recall]
            if (step + 1) * self.batch_size % self.log_every == 0:
                print("Evaluating {}, done: {}/{}".format('test', (step + 1) * self.batch_size, self.dataset.num_test_examples))
                eval_tensors += [self.merged_summary_op]

            feed_dict = {self.is_training: False, self.dataset_type: 'test'}

            eval_ret = self.sess.run(eval_tensors, feed_dict=feed_dict)
            eval_ret = dict(zip(eval_tensors, eval_ret))

            if self.merged_summary_op in eval_tensors:
                self.summary_test_writer.add_summary(eval_ret[self.merged_summary_op], self.global_step.eval(session=self.sess))

            losses.append(eval_ret[self.loss])

            if preds.size == 0:
                labels = np.argmax(eval_ret[self.Yoh], axis=1)
                preds = np.argmax(eval_ret[self.preds], axis=1)
            else:
                labels = np.concatenate((labels, np.argmax(eval_ret[self.Yoh], axis=1)), axis=0)
                preds = np.concatenate((preds, np.argmax(eval_ret[self.preds], axis=1)), axis=0)

        total_loss = np.mean(losses)
        acc, pr, rec = util.acc_prec_rec_score(labels, np.argmax(preds, axis=1))
        prAtTop10, top5CorrectWords, top5IncorrectWords = util.testStatistics(labels, preds)
        util.write_test_results(total_loss, acc, pr, rec, prAtTop10, top5CorrectWords, top5IncorrectWords, self.dataset.name, self.name)
        print("Validation results -> {} error: epoch {} loss={} accuracy={} precision={} recall={} prAtTop10={}".format('test', '1', total_loss, acc, pr, rec, prAtTop10))

    def initConfig(self):

        print("INITIALIZING CONFIGURATION VARIABLES")
        self.name = 'ef'
        self.learning_rate = config.config['learning_rate']
        self.regularizer_scale = config.config['regularizer_scale']
        self.decay_steps = config.config['decay_steps'] # broj koraka za smanjivanje stope ucenja
        self.decay_rate = config.config['decay_rate'] # rate smanjivanja stope ucenja
        self.max_epochs = config.config['max_epochs'] # broj epoha ucenja
        self.log_every = config.config['log_every'] # ispisi na sout svakih x slika
        self.batch_size = config.config['batch_size'] # velicina batcha
        self.save_every = config.config['save_every'] # spremi session svakih x posto epohe
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
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, self.name + '.ckpt'))

ef = EF()
ef.train()