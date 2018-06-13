import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import os
import skimage as ski
import skimage.io
import sklearn as sk
import operator

from sklearn.metrics import confusion_matrix
from src import config
import pickle

def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def class_to_onehot(Y, max_value):
    Yoh = np.zeros((len(Y), max_value))
    Yoh[range(len(Y)), Y] = 1
    return Yoh

def onehot_to_class(Y):
    return np.argmax(Y, axis=1)

def plot_training_progress(data, datasetname, modelname):
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)

    linewidth = 1
    legend_size = 10
    train_color = 'r'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)

    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color, linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color, linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)

    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color, linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color, linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)

    ax3.set_title('Average class precision')
    ax3.plot(x_data, data['train_pr'], marker='o', color=train_color, linewidth=linewidth, linestyle='-', label='train')
    ax3.plot(x_data, data['valid_pr'], marker='o', color=val_color, linewidth=linewidth, linestyle='-', label='validation')
    ax3.legend(loc='upper left', fontsize=legend_size)

    ax4.set_title('Average class recall')
    ax4.plot(x_data, data['train_rec'], marker='o', color=train_color, linewidth=linewidth, linestyle='-', label='train')
    ax4.plot(x_data, data['valid_rec'], marker='o', color=val_color, linewidth=linewidth, linestyle='-', label='validation')
    ax4.legend(loc='upper left', fontsize=legend_size)

    ax5.set_title('Learning rate')
    ax5.plot(x_data, data['lr'], marker='o', color=train_color, linewidth=linewidth, linestyle='-', label='learning_rate')
    ax5.legend(loc='upper right', fontsize=legend_size)

    result_dir = os.path.join(config.config['results_root_dir'], datasetname, modelname)
    create_dir(result_dir)
    pdfFile = os.path.join(result_dir, config.config['filename_pattern'] + config.PDF_EXT)
    print('Plotting in: ', pdfFile)
    plt.savefig(pdfFile)

    csvFile = os.path.join(result_dir, config.config['filename_pattern'] + config.CSV_EXT)

    with open(csvFile, 'w') as f:

        line = 'epoch_num;train_loss;train_acc;train_pr;train_rec;valid_loss;valid_acc;valid_pr;valid_rec;lr;epoch_time\n'
        f.write(line)

        for i in range(0, num_points):
            line = str(i+1) + ';' + \
            str(data['train_loss'][i]) + ';' + str(data['train_acc'][i]) + ';' + str(data['train_pr'][i]) + ';' + str(data['train_rec'][i]) + ';' + \
            str(data['valid_loss'][i]) + ';' + str(data['valid_acc'][i]) + ';' + str(data['valid_pr'][i]) + ';' + str(data['valid_rec'][i]) + ';' + \
            str(data['lr'][i][0]) + ';' + str(data['epoch_time'][i]) + '\n'

            f.write(line)

def eval_perf_multi(Y, Y_):
    pr = []
    n = max(Y_) + 1
    M = confusion_matrix(Y, Y_)
    for i in range(n):
        tp_i = M[i, i]
        fn_i = np.sum(M[i, :]) - tp_i
        fp_i = np.sum(M[:, i]) - tp_i
        tn_i = np.sum(M) - fp_i - fn_i - tp_i
        recall_i = tp_i / (tp_i + fn_i)
        precision_i = tp_i / (tp_i + fp_i)
        pr.append((precision_i, recall_i))

    accuracy = np.trace(M) / np.sum(M)

    return accuracy, pr

def acc_prec_rec_score(Ytrue, Ypred):
    return sk.metrics.accuracy_score(Ytrue, Ypred), sk.metrics.precision_score(Ytrue, Ypred, average='macro'), sk.metrics.recall_score(Ytrue, Ypred, average='macro')

def video_to_images(path):
    videoCapture = cv.VideoCapture(path)
    images = []
    while 1:
        success, image = videoCapture.read()
        if success:
            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            images.append(rgb)
        else:
            break

    return images

def draw_image(img, figureIndex=0):
    plt.figure(figureIndex)
    if img.shape[-1] == 1:
        img = np.reshape(img, [img.shape[0], img.shape[1]])
    ski.io.imshow(img)
    ski.io.show()

def draw_image_gauss(img, figureIndex, mean, std):
    plt.figure(figureIndex)
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()

def create_dir(dir):

    dir_peaces = os.path.splitdrive(dir)

    if len(dir_peaces) == 0 or len(dir_peaces[1]) == 0:
        return

    build_dir = dir_peaces[0] + os.path.sep
    for part in dir_peaces[1].split(os.path.sep):
        build_dir = os.path.join(build_dir, part)
        if not os.path.exists(build_dir):
            os.mkdir(build_dir)

    return build_dir

def isDirOrFileExist(thePath):
    return os.path.exists(thePath)

def createNecesseryDirs(dirKeys):
    for dirPath in dirKeys:
        create_dir(dirPath)

createNecesseryDirs(config.DIRS_TO_CREATE)

def write_test_results(total_loss, acc, pr, rec, prAtTop10, top5CorrectWords, top5IncorrectWords, datasetname, modelname):

    classmap = readClassmapFile(os.path.join(config.config['tfrecords_root_dir'], datasetname + '_tfrecords', datasetname + '_classmap.txt'))
    testStatFile = os.path.join(config.config['results_root_dir'], datasetname, modelname, 'test_' + config.config['filename_pattern'] + config.TXT_EXT)

    with open(testStatFile, 'w') as f:

        line = 'train_loss;accuracy;precision;recall;prAtTop10\n'
        f.write(line)

        line = str(total_loss) + ';' + str(acc) + ';' + str(pr) + ';' + str(rec) + ';' + str(prAtTop10) + '\n\n'
        f.write(line)

        f.write('Top 5 CORRECT items\n\n')
        for item in top5CorrectWords:
            f.write('label={}  correct_counter={} prob_rank={} prob={}\n'.format(item[0], item[1], item[2], 0 if item[2] > 9 else item[4][item[2]]))
            f.write(classmap[str(item[0])] + ' -> ' + ' :: '.join(['{} ({}%)'.format(classmap[str(item[3][i])], decimalToPercent(str(item[4][i]))) for i in range(len(item[3]))]) + '\n')

        f.write('\nTop 5 INCORRECT items\n\n')
        for item in top5IncorrectWords:
            f.write('label={}  correct_counter={} prob_rank={} prob={}\n'.format(item[0], item[1], item[2],  0 if item[2] > 9 else item[4][item[2]]))
            f.write(classmap[str(item[0])] + ' -> ' + ' :: '.join(['{} ({}%)'.format(classmap[str(item[3][i])], decimalToPercent(str(item[4][i]))) for i in range(len(item[3]))]) + '\n')

    f.close()

def decimalToPercent(decimalNumber):
    return '%.2f' % (float(decimalNumber) * 100)

def lrwWordsToNumbers(videos_dir):
    words = {}
    indexer = 0
    for word in os.listdir(videos_dir):
        if word not in words:
            words[word] = indexer
            indexer += 1

    return words

def readClassmapFile(filepath, numsAsKeys=True):
    labelsToNums = {}
    if not isDirOrFileExist(filepath):
        return labelsToNums
    with open(filepath, 'r') as file:
        for line in file.readlines():
            label, num = line.replace('\n', '').split('->')
            if numsAsKeys:
                labelsToNums[num] = label
            else:
                labelsToNums[label] = num
    return labelsToNums

def writeClassmapFile(fileapath, labelsToNumsMap):
    sorted_x = sorted(labelsToNumsMap.items(), key=operator.itemgetter(1))
    with open(fileapath, 'w') as file:
        for item in sorted_x:
            file.write(item[0] + '->' + str(item[1]) + '\n')
    file.close()

def mapTop10ForEveryLabel(labels, preds_onehot):
    top10List = []
    for i in range(len(labels)):
        top10indices = np.argsort(preds_onehot[i])[::-1][:50]
        top10List.append((labels[i], top10indices, preds_onehot[i][top10indices]))

    return top10List

# CALCULATING PRECISION AT TOP 10, AND RETURNING TOP 5 CORRECT WORDS, AND TOP 5 INCORRECT WORDS
# IN SHAPE OF [(LABEL, CORRECTCOUNTER, INDEX OF LABEL POSITION IN PROBABILITIES TOP 10 LIST, TOP 10 CLASSES, TOP 10 CLASS PROBS)...]
def testStatistics(labels, preds_onehot):

    inTop10 = 0
    classNums = len(preds_onehot[0])
    labelsResultsDict = {}
    labelsStat = {}
    allItems = []
    numOfTopItems = 50 if classNums >= 10 else classNums

    for i in range(classNums):
        labelsResultsDict[i] = (0)
        labelsStat[i] = (numOfTopItems, [], [])

    for item in mapTop10ForEveryLabel(labels, preds_onehot):
        if item[0] in item[1]:
            inTop10 += 1
            labelsProbIndex = labelsStat[item[0]][0]
            labelsTopIndexes = labelsStat[item[0]][1]
            labelsTopProbs = labelsStat[item[0]][2]
            allItems.append((item[0], labelsProbIndex, item[1], item[2]))

            for j in range(numOfTopItems):
                if item[1][j] == item[0] and j <= labelsProbIndex:
                    labelsProbIndex = j
                    labelsTopIndexes = item[1]
                    labelsTopProbs = item[2]
                    break

            labelsResultsDict[item[0]] = labelsResultsDict[item[0]] + 1
            labelsStat[item[0]] = (labelsProbIndex, labelsTopIndexes, labelsTopProbs)

        else:

            if labelsStat[item[0]][0] == numOfTopItems:
                labelsStat[item[0]] = (numOfTopItems, item[1], item[2])
                allItems.append((item[0], numOfTopItems, item[1], item[2]))

    sorted_x = sorted(labelsResultsDict.items(), key=operator.itemgetter(1))

    top5Correct = []
    top5Incorrect = []
    sorted_x_len = len(sorted_x)

    if sorted_x_len > 4:

        i = 0
        foundCounter = 0
        while i < sorted_x_len and foundCounter < 5:
            if labelsStat[sorted_x[i][0]][0] > 0:
                top5Incorrect.append((sorted_x[i][0], sorted_x[i][1], labelsStat[sorted_x[i][0]][0], labelsStat[sorted_x[i][0]][1], labelsStat[sorted_x[i][0]][2]))
                foundCounter = foundCounter + 1
            i = i + 1

        for i in range(len(sorted_x)-5, len(sorted_x)):
            top5Correct.append((sorted_x[i][0], sorted_x[i][1], labelsStat[sorted_x[i][0]][0], labelsStat[sorted_x[i][0]][1], labelsStat[sorted_x[i][0]][2]))

    i = 0
    foundCounter = 0
    lenAllItems = len(allItems)
    while i < lenAllItems and foundCounter < 30:
        if allItems[i][1] > 0:
            top5Incorrect.append((allItems[i][0], sorted_x[allItems[i][0]][1], allItems[i][1],
                                  allItems[i][2], allItems[i][3]))
            foundCounter = foundCounter + 1
        i = i + 1


    return (float(inTop10 / len(labels)), top5Correct, top5Incorrect)

def log_step(epoch, step, duration, loss, batch_size, num_examples, log_every):
    if (step + 1) * batch_size % log_every == 0:
        format_str = 'epoch %d, step %d / %d, loss = %.2f (%.3f sec/batch)'
        print(format_str % (epoch, (step + 1) * batch_size, num_examples, loss, float(duration)))

def vgg_normalization(images, rgb_mean, axis=3):
    r, g, b = tf.split(axis=axis, num_or_size_splits=3, value=images)
    rgb = tf.concat(axis=axis, values=[
        r - rgb_mean[0],
        g - rgb_mean[1],
        b - rgb_mean[2]
    ])
    return rgb


def mnistClassmap(numAsKeys=True):

    mnistMap = {}
    for i in range(10):
        if numAsKeys:
            mnistMap[i] = str(i)
        else:
            mnistMap[str(i)] = i

    return mnistMap


def cifarClassmap(cifarClassesFile):

    cifarMap = {}
    with open(cifarClassesFile, 'r') as file:
        counter = 0
        for line in file.readlines():
            cifarMap[line.replace('\n', '')] = counter
            counter = counter + 1
    file.close()
    return cifarMap