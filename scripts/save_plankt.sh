#!/bin/bash
#SBATCH --partition=archive
#SBATCH --cpus-per-task=5
#SBATCH --hint=nomultithread
#SBATCH --time=7:00:00
#SBATCH --output=logs/%x_%j.out  

# === Preparación del entorno ===
cd $WORK/am/planktonzilla/
module purge
source .venv/bin/activate
cd notebooks


# === Ejecutar torchrun ===
# WIRE-01 (CONCERNS #10): repointed at notebooks/save_planktonzilla_for_clip.py. Was previously a non-existent 'save_planktonzilla<N>.py' filename that broke job submission.
srun python save_planktonzilla_for_clip.py