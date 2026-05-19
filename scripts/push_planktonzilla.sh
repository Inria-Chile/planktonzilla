#!/bin/bash
#SBATCH --partition=compil
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out  

# === Preparación del entorno ===
cd $WORK/am/planktonzilla/
module purge
source .venv/bin/activate
cd notebooks


# === Ejecutar torchrun ===
# WIRE-01 (CONCERNS #10): repointed at notebooks/push_planktonzilla.py. Was previously a non-existent 'push_planktonzilla<N>.py' filename that broke job submission.
srun python push_planktonzilla.py