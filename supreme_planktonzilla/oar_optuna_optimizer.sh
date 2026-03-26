#!/bin/bash
#OAR -n optuna_multinode
#OAR -l host=1,walltime=5:00:00
#OAR -p gpu_count >= 2 AND gpu_mem >= 20000
#OAR --array 2
#OAR -q besteffort
#OAR -O optimize_array_%jobid%.out
#OAR -E optimize_array_%jobid%.err

# 1. Rutas del Entorno
VENV_PYTHON="/home/svasquez/.pyenv/versions/neglabel/bin/python"
WORK_DIR="/home/svasquez/clip_prompt_learning_planktonzilla"

# 2. Configuración del Script
# Ajusta el nombre de tu SCRIPT si lo guardaste en otra carpeta (ej: scripts.optimize)
SCRIPT="scripts.optimize_hyperparams"
CONFIG="/home/svasquez/clip_prompt_learning_planktonzilla/config/experiments_supreme_base_optimize.yaml"

# ¿Cuántos trials ejecutará CADA GPU? 
# Si el array es de 3 nodos, con 2 GPUs c/u = 6 GPUs totales.
# 6 GPUs * 10 trials = 60 trials totales en la base de datos de Optuna.
TRIALS_PER_WORKER=5

cd $WORK_DIR

# 3. Detección Dinámica de Recursos Locales
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "=== NODO OAR: $OAR_NODE_NAME | GPUs detectadas: $NUM_GPUS ==="
echo "Lanzando $NUM_GPUS workers locales. Cada worker hará $TRIALS_PER_WORKER trials."

# 4. Lanzamiento de Workers Distribuidos
for ((i=0; i<NUM_GPUS; i++)); do
    echo "Iniciando worker de Optuna en GPU $i..."
    
    # CUDA_VISIBLE_DEVICES aísla el proceso a una sola GPU
    # El símbolo '&' envía el proceso al background
    CUDA_VISIBLE_DEVICES=$i $VENV_PYTHON -m $SCRIPT \
        --config "$CONFIG" \
        --trials $TRIALS_PER_WORKER & 
done

# 5. Barrera de Sincronización
echo "Todos los workers del nodo $OAR_NODE_NAME han sido lanzados. Esperando a que terminen..."
wait

echo "¡Optimización completada exitosamente en el nodo $OAR_NODE_NAME!"