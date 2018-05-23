import os

class LRWReader:

    def readLRWtoMap(self, lrw_dir_path, datasetType):

        return self.read_word_video_files(self.read_word_dirs(lrw_dir_path), datasetType)

    def read_word_dirs(self, lrw_dir_path):

        wordDirs = {}

        for file in os.listdir(lrw_dir_path):
            filename = os.fsdecode(file)
            current_word_path = os.path.join(lrw_dir_path, filename)
            if os.path.isdir(current_word_path):
                wordDirs[filename] = current_word_path

        return wordDirs

    def read_word_video_files(self, wordDirsMap, datasetType):

        wordsVideoFiles = {}

        for key in wordDirsMap:
            wordDir = os.path.join(wordDirsMap[key], datasetType)
            for file in os.listdir(wordDir):
                filename = os.fsdecode(file)
                if filename.endswith(".mp4"):
                    wordsVideoFiles[os.path.join(wordDir, filename)] = key


        return wordsVideoFiles
