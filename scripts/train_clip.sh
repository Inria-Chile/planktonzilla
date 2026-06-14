#!/bin/bash
#SBATCH --account=tec@h100
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --constraint=h100
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --time=15:00:00
#SBATCH --output=logs/%x_%j.out  

# === Preparación del entorno ===
cd $WORK/am/open_clip
module purge
module load arch/h100
source .env/bin/activate

export TORCH_DISTRIBUTED_TIMEOUT=7200
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export SLURM_JOB_ID=${SLURM_JOB_ID:-local}
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

# === Configuración multi-nodo ===
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29501

# === Ejecutar torchrun ===
srun torchrun \
  --nproc_per_node=4 \
  --nnodes=16 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m planktonzilla.clip_train.main \
  --save-frequency 5 \
  --train-data "${WORK:-$PWD}/data/shards/train/shard_{00000..01771}.tar" \
  --train-num-samples 1771611 \
  --dataset-type webdataset \
  --lr 1e-4 \
  --wd 0.2 \
  --batch-size 256 \
  --accum-freq 1 \
  --epochs 100 \
  --warmup 1000 \
  --workers 4 \
  --model EVA02-L-14 \
  --seed 0 \
  --local-loss \
  --gather-with-grad \
  --grad-checkpointing \
  "$@"
