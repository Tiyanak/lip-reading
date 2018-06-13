set DATASET=%1
set MODEL=%2
set TYPE=%3

IF /I %TYPE%==train set TRAINSUM=1 & set TRAINEVALSUM=0 & set VALSUM=0 & set TESTSUM=0
IF /I %TYPE%==traineval set TRAINSUM=0 & set TRAINEVALSUM=1 & set VALSUM=0 & set TESTSUM=0
IF /I %TYPE%==val set TRAINSUM=0 & set TRAINEVALSUM=0 & set VALSUM=1 & set TESTSUM=0
IF /I %TYPE%==test set TRAINSUM=0 & set TRAINEVALSUM=0 & set VALSUM=0 & set TESTSUM=1
IF /I %TYPE%==all set TRAINSUM=1 & set TRAINEVALSUM=1 & set VALSUM=1 & set TESTSUM=1

echo %TRAINSUM% %TRAINEVALSUM% %VALSUM% %TESTSUM%

IF %TRAINSUM% EQU 1 explorer http://localhost:6006 & start cmd /k tensorboard --logdir=evaluation/summary/%DATASET%/%MODEl%/train/ --host=localhost --port=6006
IF %TRAINEVALSUM% EQU 1 explorer http://localhost:6007 & start cmd /k tensorboard --logdir=evaluation/summary/%DATASET%/%MODEl%/trainEval/ --host=localhost --port=6007
IF %VALSUM% EQU 1 explorer http://localhost:6008 & start cmd /k tensorboard --logdir=evaluation/summary/%DATASET%/%MODEl%/valEval/ --host=localhost --port=6008
IF %TESTSUM% EQU 1 explorer http://localhost:6009 & start cmd /k tensorboard --logdir=evaluation/summary/%DATASET%/%MODEl%/test/ --host=localhost --port=6009