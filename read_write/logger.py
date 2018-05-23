from utils import config
import os

class Logger():

    def __init__(self):

        self.name = 'logger'
        self.log_dir = config.config['log_dir']
        self.log_file = os.path.join(self.log_dir, config.config['results_filename'] + config.TXT_EXT)

    def log(self, key, message):

        with open(self.log_file, 'a') as file:
            file.write(key + ' ->: ' + message + '\n')
        file.close()