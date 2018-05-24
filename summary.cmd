start cmd /k tensorboard --logdir=summary/lrw/ef/train/ --host=localhost --port=6006
start cmd /k tensorboard --logdir=summary/lrw/ef/trainEval/ --host=localhost --port=6007
start cmd /k tensorboard --logdir=summary/lrw/ef/valEval/ --host=localhost --port=6008
start cmd /k tensorboard --logdir=summary/lrw/ef/test/ --host=localhost --port=6009
start cmd /k explorer http://localhost:6006 & explorer http://localhost:6007 & explorer http://localhost:6008 & explorer http://localhost:6009