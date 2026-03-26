#!/bin/bash
#OAR -n kmeans-4shots
#OAR -l host=1,walltime=5:00:00
#OAR -p gpu_count >= 2 AND gpu_mem >= 20000
#OAR -q besteffort
#OAR -O train_%jobid%.out
#OAR -E train_%jobid%.err

VENV_PYTHON="/home/svasquez/.pyenv/versions/neglabel/bin/python"
WORK_DIR="/home/svasquez/clip_prompt_learning_planktonzilla"

cd $WORK_DIR

SCRIPT="scripts.fs_selection"
CONFIG="/home/svasquez/clip_prompt_learning_planktonzilla/config/vitL.yaml"
SHOTS=4
OUT="/home/svasquez/clip_prompt_learning_planktonzilla/data/VITL_planktonzilla_4shot_indices.json"



$VENV_PYTHON -m  $SCRIPT --config $CONFIG --shots $SHOTS --out $OUT

