set DATASET=%1
set MODEL=%2

start cmd /k tensorboard --logdir=evaluation/summary/%DATASET%/%MODEl%/train/ --host=localhost --port=6006
start cmd /k tensorboard --logdir=evaluation/summary/%DATASET%/%MODEl%/trainEval/ --host=localhost --port=6007
start cmd /k tensorboard --logdir=evaluation/summary/%DATASET%/%MODEl%/valEval/ --host=localhost --port=6008
start cmd /k tensorboard --logdir=evaluation/summary/%DATASET%/%MODEl%/test/ --host=localhost --port=6009
start cmd /k explorer http://localhost:6006 & explorer http://localhost:6007 & explorer http://localhost:6008 & explorer http://localhost:6009