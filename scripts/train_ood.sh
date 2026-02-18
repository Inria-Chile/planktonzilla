#!/bin/bash
#SBATCH --account=tec@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --constraint=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2
#SBATCH --hint=nomultithread
#SBATCH --time=0:20:00
#SBATCH --output=logs/%x_%j.out  

# === Preparación del entorno ===
cd $WORK/planktonzilla/
module purge
module load arch/h100
source .venv/bin/activate
cd planktonzilla

export SLURM_JOB_ID=${SLURM_JOB_ID:-local}
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

# === Configuración multi-nodo ===
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

# === Ejecutar torchrun ===
srun torchrun \
  --nproc_per_node=2 \
  --nnodes=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train_ood.py "$@"