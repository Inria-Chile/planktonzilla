#!/bin/bash
#OAR -n baseline_wout_ft
#OAR -l host=4,walltime=4:00:00
#OAR -p gpu_count >= 2 AND gpu_mem >= 20000
#OAR -q besteffort
#OAR -O scores_%jobid%.out
#OAR -E scores_%jobid%.err

# 2. Configuration
VENV_PYTHON="/home/svasquez/.pyenv/versions/neglabel/bin/python"
WORK_DIR="/home/svasquez/clip_prompt_learning_planktonzilla"

cd $WORK_DIR

SCRIPT="supreme.baselines"
CONFIG="/home/svasquez/clip_prompt_learning_planktonzilla/config/experiments_supreme_vitb_wout_ft.yaml"
SAVE_DIR="/home/svasquez/clip_prompt_learning_planktonzilla/results/baselines/experiments_supreme_vitb_wout_ft"
CKPT="/home/svasquez/clip_prompt_learning_planktonzilla/models/supreme/supreme_vitb16_idx_precalculated_50e_norm_wout_mean_4shots_kmeans/trial0.pth"


# 3. Get Node Info
# Get unique list of nodes allocated
NODES=$(cat $OAR_NODEFILE | sort -u)
NUM_NODES=$(echo "$NODES" | wc -l)

echo "--- Launching on $NUM_NODES nodes ---"
echo "Nodes: $NODES"
echo "Using Python at: $VENV_PYTHON"

# 4. Parallel Execution
# We use GNU Parallel to ssh into each node and run the python script.
# {#} gives the job sequence number (0, 1, 2, 3...), which maps perfectly to --chunk_id.
# % is the remainder, ensuring we cycle through nodes if tasks > nodes (though here tasks=nodes).

parallel --sshloginfile <(echo "$NODES") \
         --ssh oarsh \
         --jobs 1 \
         --wd $WORK_DIR \
         --ungroup \
         "$VENV_PYTHON -m $SCRIPT \
            --config $CONFIG \
            --save_dir $SAVE_DIR \
            --num_chunks 20 \
            --batch_size 256 \
            --chunk_id {}" \
         ::: $(seq 0 19)

echo "--- All Extraction Jobs Submitted ---"