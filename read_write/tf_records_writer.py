import os
import tensorflow as tf
import numpy as np
from random import shuffle
from read_write.lrw_reader import LRWReader
from utils import config
from utils import util
from imaging.image_resize import ImageResizer

LRW_DATASET_DIR = 'D:/faks/diplomski/lipreading-data/data/lrw'
TFRECORDS_DIR = os.path.join(config.config['tfrecords_root_dir'], 'lrw_tfrecords')
WRITE_BATCH_SIZE = 2000

class TFRecordsWriter():

    def start_lrw_writer(self):

        util.create_dir(LRW_DATASET_DIR)
        util.create_dir(TFRECORDS_DIR)

        dataset_types = config.config['dataset_types']
        classmapfile = os.path.join(TFRECORDS_DIR, 'lrw_classmap.txt')

        if util.isDirOrFileExist(classmapfile):
            lrwClasses = util.readClassmapFile(classmapfile, numsAsKeys=False)
        else:
            lrwClasses = util.lrwWordsToNumbers(LRW_DATASET_DIR)
            util.writeClassmapFile(classmapfile, lrwClasses)

        lrwReader = LRWReader()

        for datasetType in dataset_types:
            lrwMap = lrwReader.readLRWtoMap(LRW_DATASET_DIR, datasetType)
            self.write_records(lrwMap, lrwClasses, 'lrw', datasetType)

    def write_records(self, videoMap, videoClassMap, datasetname, dataset_type):

        print("Creating dir for saving tf records!")
        dirpath = util.create_dir(os.path.join(TFRECORDS_DIR, dataset_type))

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
                images = self.resize_images(images)
                batchWithImagesList.append((video[0], video[1], videoClassMap[video[1]], images))

            batch_filename = os.path.join(dirpath, '{}_{}_batch_{}.{}'.format(datasetname, dataset_type, batchIndexer, "tfrecords"))
            self.write(batch_filename, batchWithImagesList, dataset_type)

            batch_stat_filename = os.path.join(dirpath, '{}_{}_batch_{}.{}'.format(datasetname, dataset_type, batchIndexer, "txt"))
            self.write_written_images(batch_stat_filename, batchWithImagesList)

            batchIndexer += 1
            batchCounter += 1
			
            print('BATCH FINISHED')

            # EXIT BECAUSE THIS SCRIPT SHOULD BE STARTED WITH BASH SCRIPT write_tfrecords.bat
            # BECAUSE SOMETHING IS BROKEN AND ITS GOING SLOW AFTER FIRST WRITEN BATCH, SO ALWAYS RESTART SCRIPT
            # AFTER BATCH IS FINISHED TO REFRESH SPEED OF WRITING
            exit(1)

    def write(self, filename, batchWithImagesList, dataset_type):

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

    def resize_images(self, images):
        imgResizer = ImageResizer()
        return imgResizer.start_resizing(images, 112, 112)


tfRecordsWriter = TFRecordsWriter()
tfRecordsWriter.start_lrw_writer()
