#!/bin/bash 

#OAR -p mercantour6-1
#OAR -q besteffort 
#OAR -l host=1,walltime=20:00:00
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err 

# export HF_DATASETS_OFFLINE=1

source "$HOME/planktonzilla/.venv/bin/activate"

export HF_HOME="/home/acontreras/group_storage_rennes/acontreras/hf_datasets"

# HF_HOME mueve también el lookup del token, por lo que aquí no se encontraría.
# Exportamos HF_TOKEN desde el login existente (tiene prioridad sobre el archivo)
# para que push_to_hub pueda autenticarse y escribir en el Hub.
export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"

python "$HOME/planktonzilla/dataset_generation/update_planktonzilla.py"