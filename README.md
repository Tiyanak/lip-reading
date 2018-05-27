Python project made with Tensorflow, purpose is to learn to read from peoples lips.

Data is only local, its too big for git ofc.

But, for local testing without real data, there is a tf_records_writer.py file, that you can run, taking care on bottom lines to 
choose start_mnist_writer(), it will download mnist and write it as tfrecords, resizing to 112, 112 or keeping original size if using 
start_mnist_original_writer()

Models are in src/cnn/models, every model is a separate running script, so its easyer to run parralel models at same time (if you have enough graphic power :D), in script on top you can change some parameters.