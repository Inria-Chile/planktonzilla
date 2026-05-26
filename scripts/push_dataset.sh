#!/bin/bash
#SBATCH --partition=compil
#SBATCH --cpus-per-task=5
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out  

# === Preparación del entorno ===
cd $WORK/am/planktonzilla/
module purge
source .venv/bin/activate

# === Ejecutar torchrun ===
# WIRE-03 (CONCERNS #12): data_dir parameterized via $WORK (Jean Zay) or $PWD (local checkout fallback).
srun pz_import_dataset action=import "dataset_import.data_dir=${WORK:-$PWD}/data" "$@"