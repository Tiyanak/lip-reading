import os
import datetime

config = {}

config['dataset_types'] = ['train', 'val', 'test']

config['use_se'] = False

# APPLICATION ROOT DIR
config['root_dir'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config['data_root_dir'] = os.path.join(config['root_dir'], 'data')
config['results_root_dir'] = os.path.join(config['root_dir'], 'evaluation', 'results')
config['summary_root_dir'] = os.path.join(config['root_dir'], 'evaluation', 'summary')
config['checkpoint_root_dir'] = os.path.join(config['data_root_dir'], 'checkpoint')
config['datasets_root_dir'] = os.path.join(config['data_root_dir'], 'datasets')
config['tfrecords_root_dir'] = os.path.join(config['data_root_dir'], 'tfrecords')

# FILE PATTERN
config['filename_pattern'] = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')

# EXTENSIONS
PDF_EXT = '.pdf'
CSV_EXT = '.csv'
TXT_EXT = '.txt'

DIRS_TO_CREATE = [config['data_root_dir'], config['results_root_dir'], config['checkpoint_root_dir'], config['summary_root_dir']]

