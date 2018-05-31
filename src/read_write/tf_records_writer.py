import os
import tensorflow as tf
import numpy as np
from random import shuffle
from src.read_write.lrw_reader import LRWReader
from src.utils import util
from src import config
from src.imaging.image_resize import ImageResizer
from src.read_write.mnist_reader import MnistReader
from src.read_write.cifar_reader import CifarReader

WRITE_BATCH_SIZE = 2000

LRW_DATASET_DIR = 'D:/faks/diplomski/lipreading-data/data/lrw'
LRW_TFRECORDS_DIR = os.path.join(config.config['tfrecords_root_dir'], 'lrw_tfrecords')

MNIST_DATASET_DIR = 'D:/faks/diplomski/lip-reading/data/datasets/mnist'
MNIST_TFRECORDS_DIR = os.path.join(config.config['tfrecords_root_dir'], 'mnist_tfrecords')

MNIST_ORIGINAL_DATASET_DIR = 'D:/faks/diplomski/lip-reading/data/datasets/mnist'
MNIST_ORIGINAL_TFRECORDS_DIR = os.path.join(config.config['tfrecords_root_dir'], 'mnist_original_tfrecords')

CIFAR_DATASET_DIR = 'D:/faks/diplomski/lip-reading/data/datasets/cifar'
CIFAR_TFRECORDS_DIR = os.path.join(config.config['tfrecords_root_dir'], 'cifar_tfrecords')

class TFRecordsWriter():

    def start_lrw_writer(self):

        util.create_dir(LRW_DATASET_DIR)
        util.create_dir(LRW_TFRECORDS_DIR)

        dataset_types = config.config['dataset_types']
        classmapfile = os.path.join(LRW_TFRECORDS_DIR, 'lrw_classmap.txt')

        if util.isDirOrFileExist(classmapfile):
            lrwClasses = util.readClassmapFile(classmapfile, numsAsKeys=False)
        else:
            lrwClasses = util.lrwWordsToNumbers(LRW_DATASET_DIR)
            util.writeClassmapFile(classmapfile, lrwClasses)

        lrwReader = LRWReader()

        for datasetType in dataset_types:
            lrwMap = lrwReader.readLRWtoMap(LRW_DATASET_DIR, datasetType)
            self.lrw_write_records(lrwMap, lrwClasses, 'lrw', datasetType)

    def start_mnist_writer(self):

        util.create_dir(MNIST_DATASET_DIR)
        util.create_dir(MNIST_TFRECORDS_DIR)

        dataset_types = config.config['dataset_types']
        classmapfile = os.path.join(MNIST_TFRECORDS_DIR, 'mnist_classmap.txt')

        if not util.isDirOrFileExist(classmapfile):
            mnistClasses = util.mnistClassmap(numAsKeys=False)
            util.writeClassmapFile(classmapfile, mnistClasses)

        mnistReader = MnistReader()

        for datasetType in dataset_types:
            self.mnist_write_records(mnistReader, MNIST_TFRECORDS_DIR, 'mnist', datasetType)

    def start_mnist_original_writer(self):

        util.create_dir(MNIST_ORIGINAL_DATASET_DIR)
        util.create_dir(MNIST_ORIGINAL_TFRECORDS_DIR)

        dataset_types = config.config['dataset_types']
        classmapfile = os.path.join(MNIST_ORIGINAL_TFRECORDS_DIR, 'mnist_classmap.txt')

        if not util.isDirOrFileExist(classmapfile):
            mnistClasses = util.mnistClassmap(numAsKeys=False)
            util.writeClassmapFile(classmapfile, mnistClasses)

        mnistReader = MnistReader()

        for datasetType in dataset_types:
            self.mnist_write_records(mnistReader, MNIST_ORIGINAL_TFRECORDS_DIR, 'mnist_original', datasetType)

    def start_cifar_writer(self):

        util.create_dir(CIFAR_DATASET_DIR)
        util.create_dir(CIFAR_TFRECORDS_DIR)

        dataset_types = config.config['dataset_types']
        classmapfile = os.path.join(CIFAR_TFRECORDS_DIR, 'cifar_classmap.txt')
        cifarClassesFile = os.path.join(CIFAR_DATASET_DIR, 'cifar_classes.txt')

        if not util.isDirOrFileExist(classmapfile):
            cifarClasses = util.cifarClassmap(cifarClassesFile)
            util.writeClassmapFile(classmapfile, cifarClasses)

        cifarReader = CifarReader()

        for datasetType in dataset_types:
            self.cifar_write_records(cifarReader, CIFAR_TFRECORDS_DIR, 'cifar', datasetType)

    def lrw_write_records(self, videoMap, videoClassMap, datasetname, dataset_type):

        print("Creating dir for saving tf records!")
        dirpath = util.create_dir(os.path.join(LRW_TFRECORDS_DIR, dataset_type))

        print("Clearing already written items!")
        videoMap = self.clear_already_written(dirpath, videoMap)

        print("Shuffling data!")
        videoList = list(videoMap.items())
        shuffle(videoList)

        print("Starting processing {} images".format(dataset_type))
        batchSize = WRITE_BATCH_SIZE
        batchIndexer = self.get_number_of_written_tfrecords(dirpath)
        totalBatches = int(len(videoList)/batchSize) + 1
        batchCounter = 0

        while len(videoList) > 0:

            print("Writing batch {}/{}".format(batchCounter + 1, totalBatches))
            currentBatch = videoList[:batchSize]
            if len(videoList) <= batchSize:
                videoList = []
            else:
                videoList = videoList[batchSize:]

            batchWithImagesList = []
            for video in currentBatch:
                resizedVideos = len(batchWithImagesList)
                if not resizedVideos % 100:
                    print("Videos read and resized: {}/{}".format(resizedVideos, len(currentBatch)))
                images = util.video_to_images(video[0])
                images = self.resize_images_lrw(images)
                batchWithImagesList.append((video[0], video[1], videoClassMap[video[1]], images))

            batch_filename = os.path.join(dirpath, '{}_{}_batch_{}.{}'.format(datasetname, dataset_type, batchIndexer, "tfrecords"))
            self.writeLRW(batch_filename, batchWithImagesList, dataset_type)

            batch_stat_filename = os.path.join(dirpath, '{}_{}_batch_{}.{}'.format(datasetname, dataset_type, batchIndexer, "txt"))
            self.write_written_images(batch_stat_filename, batchWithImagesList)

            batchIndexer += 1
            batchCounter += 1

    def writeLRW(self, filename, batchWithImagesList, dataset_type):

        print("\nWriting batch -> {}".format(filename))
        writer = tf.python_io.TFRecordWriter(filename)
        videoCounter = 0

        for item in batchWithImagesList:

            if not videoCounter % 200:
                print("Videos: {}/{}".format(videoCounter, len(batchWithImagesList)))

            images = np.ascontiguousarray(item[3])
            label = np.ascontiguousarray(item[2])

            feature = {dataset_type + '/label': util._int64_feature(int(np.asscalar(label))),
                       '{}/video'.format(dataset_type): util._bytes_feature(tf.compat.as_bytes(images.tostring()))}

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

            videoCounter += 1

        writer.close()

    def mnist_write_records(self, mnistReader, tfrecordsdir, datasetname, datasetType):

        print("Creating dir for saving tf records!")
        dirpath = util.create_dir(os.path.join(tfrecordsdir, datasetType))

        print("Starting processing {} images".format(datasetType))
        batchSize = WRITE_BATCH_SIZE
        totalBatches = int(mnistReader.getNumOfExamples(datasetType) / batchSize) + 1

        for i in range(totalBatches):

            print("Writing batch {}/{}".format(i + 1, totalBatches))

            currentBatch = mnistReader.getData(datasetType)

            images = currentBatch[0][i*batchSize:(i+1)*batchSize].astype(dtype=np.float32)
            labels = currentBatch[1][i*batchSize:(i+1)*batchSize]

            if (datasetname == 'mnist'):
                images = self.resize_images_mnist(images)

            labels = np.argmax(labels, axis=1)

            batch_filename = os.path.join(dirpath, '{}_{}_batch_{}.{}'.format(datasetname, datasetType, i, "tfrecords"))
            self.writeMNIST(batch_filename, images, labels)

    def cifar_write_records(self, cifarReader, tfrecordsdir, datasetname, datasetType):

        print("Creating dir for saving tf records!")
        dirpath = util.create_dir(os.path.join(tfrecordsdir, datasetType))

        print("Starting processing {} images".format(datasetType))
        batchSize = WRITE_BATCH_SIZE
        totalBatches = int(cifarReader.getNumOfExamples(datasetType) / batchSize) + 1

        for i in range(totalBatches):

            print("Writing batch {}/{}".format(i + 1, totalBatches))
            currentBatch = cifarReader.getData(datasetType)

            images = currentBatch[0][i*batchSize:(i+1)*batchSize].astype(dtype=np.float32)
            images = self.resize_images_cifar(images)
            labels = currentBatch[1][i*batchSize:(i+1)*batchSize]

            batch_filename = os.path.join(dirpath, '{}_{}_batch_{}.{}'.format(datasetname, datasetType, i, "tfrecords"))
            self.writeCIFAR(batch_filename, images, labels)

    def writeMNIST(self, filename, images, labels):

        print("\nWriting batch -> {}".format(filename))
        writer = tf.python_io.TFRecordWriter(filename)
        imagesSize = len(images)

        for i in range(imagesSize):

            if not i % 500:
                print("Videos: {}/{}".format(i, imagesSize))

            image = images[i]
            label = labels[i]

            feature = {'/label': util._int64_feature(int(np.asscalar(label))),
                       '/image': util._bytes_feature(tf.compat.as_bytes(image.tostring()))}

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()

    def writeCIFAR(self, filename, images, labels):

        print("\nWriting batch -> {}".format(filename))
        writer = tf.python_io.TFRecordWriter(filename)
        imagesSize = len(images)

        for i in range(imagesSize):

            if not i % 500:
                print("Videos: {}/{}".format(i, imagesSize))

            image = images[i]
            label = labels[i]

            feature = {'/label': util._int64_feature(int(np.asscalar(label))),
                       '/image': util._bytes_feature(tf.compat.as_bytes(image.tostring()))}

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

        writer.close()

    def write_written_images(self, batchStatFilename, videoList):
        file = open(batchStatFilename, 'w', encoding='utf-8')
        for item in videoList:
            file.write('{};{};{}\n'.format(item[0], item[1], item[2]))

        file.close()

    def clear_already_written(self, tfrecordsDir, videoMapToClear):

        for file in os.listdir(tfrecordsDir):
            filename = os.fsdecode(file)
            current_batch_tfrecord_stat = os.path.join(tfrecordsDir, filename)
            if current_batch_tfrecord_stat.endswith(".txt"):
                file = open(current_batch_tfrecord_stat, 'r', encoding='utf-8')
                for video in file.readlines():
                    videoMapToClear.pop(video.split(';')[0].replace('\n', ''), None)

        return videoMapToClear

    def get_number_of_written_tfrecords(self, tfrecordsDir):
        return int(round(len(os.listdir(tfrecordsDir)) / 2))

    def resize_images_lrw(self, images):
        imgResizer = ImageResizer()
        return imgResizer.start_resizing(images, 112, 112, 256, 256, 3, tf.uint8)

    def resize_images_mnist(self, images):
        imgResizer = ImageResizer()
        return imgResizer.start_resizing(images, 112, 112, 28, 28, 1, tf.float32)

    def resize_images_cifar(self, images):
        imgResizer = ImageResizer()
        return imgResizer.start_resizing(images, 112, 112, 32, 32, 3, tf.float32)

tfRecordsWriter = TFRecordsWriter()
tfRecordsWriter.start_cifar_writer()
