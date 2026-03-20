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
srun python save_planktonzilla2.py