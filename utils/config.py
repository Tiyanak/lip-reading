import os
import datetime

config = {}

config['dataset_types'] = ['train', 'val', 'test']

config['max_epochs'] = 10
config['batch_size'] = 20
config['log_every'] = 200
config['save_every'] = 0.05
config['learning_rate'] = 5e-4
config['decay_steps'] = 10000
config['decay_rate'] = 0.96
config['regularizer_scale'] = 1e-4
config['use_se'] = False

# APPLICATION ROOT DIR
config['root_dir'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config['data_root_dir'] = os.path.join(config['root_dir'], 'data')
config['results_root_dir'] = os.path.join(config['root_dir'], 'results')
config['checkpoint_root_dir'] = os.path.join(config['root_dir'], 'checkpoint')
config['summary_root_dir'] = os.path.join(config['root_dir'], 'summary')
config['datasets_root_dir'] = os.path.join(config['data_root_dir'], 'datasets')
config['tfrecords_root_dir'] = os.path.join(config['data_root_dir'], 'tfrecords')
config['logs_root_dir'] = os.path.join(config['root_dir'], 'logs')

# FILE PATTERN
config['filename_pattern'] = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')

# EXTENSIONS
PDF_EXT = '.pdf'
CSV_EXT = '.csv'
TXT_EXT = '.txt'

DIRS_TO_CREATE = [config['data_root_dir'], config['results_root_dir'], config['checkpoint_root_dir'], config['summary_root_dir'], config['logs_root_dir']]

