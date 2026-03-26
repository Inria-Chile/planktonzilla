#!/bin/bash
#OAR -n train_trial64
#OAR -l host=1,walltime=2:30:00
#OAR -p gpu_count >= 2 AND gpu_mem >= 20000
#OAR -q besteffort
#OAR -O train_%jobid%.out
#OAR -E train_%jobid%.err

VENV_PYTHON="/home/svasquez/.pyenv/versions/neglabel/bin/python"
WORK_DIR="/home/svasquez/clip_prompt_learning_planktonzilla"

cd $WORK_DIR

SCRIPT="supreme.train"
CONFIG="/home/svasquez/clip_prompt_learning_planktonzilla/config/experiments_supreme_base_optimize_trial64.yaml"

$VENV_PYTHON -m  $SCRIPT --config $CONFIG