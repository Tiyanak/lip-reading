from src.utils import util
from src import config
import os

DATASET = 'lrw'
MODEL = 'mt'
CSV_FILE_NAME = os.path.join(config.config['results_root_dir'], DATASET, MODEL, '2018_06_16_15_56.csv')

plot_data = {
            'train_loss': [], 'train_acc': [],
            'valid_loss': [], 'valid_acc': [],
        }

with open(CSV_FILE_NAME, 'r') as file:

    lines = file.readlines()

    for i in range(1, len(lines)):

        parsedLine = lines[i].split(';')

        plot_data['train_loss'] += [float(parsedLine[1])]
        plot_data['train_acc'] += [float(parsedLine[2])]
        plot_data['valid_loss'] += [float(parsedLine[5])]
        plot_data['valid_acc'] += [float(parsedLine[6])]

util.plot_loss_acc(plot_data, DATASET, MODEL)


