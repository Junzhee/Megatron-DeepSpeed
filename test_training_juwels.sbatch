#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=gpt2_model_TP2_PP2
#SBATCH --account=cstdl
#SBATCH --partition=develbooster
#SBATCH --nodes=2
#SBATCH --gres=gpu:4                    # numbe of GPUs
#SBATCH --ntasks-per-node=1             # Crucial - only one task per dist per node !
#SBATCH --cpus-per-task=48           # Slurm 22.05: srun doesnot inherit this variable from sbatch
#SBATCH --time=02:00:00                 # maximum execution time (HH:MM:SS)
#SBATCH --threads-per-core=1            # using only real cores, no SMT
#SBATCH --hint=nomultithread
#SBATCH --output=./log/%x-%j.out              # output file name
#SBATCH --error=./log/%x-%j.err               # error file name

#### Comments ####
# Use `develbooster` for debugging, `booster` for "normal" jobs, and
# `largebooster` for jobs on more than 256 nodes.

# explicitly setting srun environment variable to inherit from SBATCH
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Enable logging
set -euo pipefail
set -x

# INSTALLATION_DIR="$1"
# RUN_DIR="$2"

INSTALLATION_DIR="/p/scratch/ccstdl/xu17/jz/Megatron-DeepSpeed"
RUN_DIR="/p/scratch/ccstdl/xu17/jz/Megatron-DeepSpeed/output"

mkdir -p "$RUN_DIR"

# if [ -z "$SETUP_SCRIPTS_DIR" ]; then
#     echo "SETUP_SCRIPTS_DIR is not set. Please set the path to the OpenGPTX-Bigscience-Megatron-DeepSpeed-Setup repository." >&2
#     exit 1
# fi
# source "$SETUP_SCRIPTS_DIR"/setupworkenv.sh "$INSTALLATION_DIR"|| exit 1 
source /p/scratch/ccstdl/xu17/miniconda3/bin/activate /p/scratch/ccstdl/xu17/miniconda3/envs/jz-deepspeed

ml GCC
ml OpenMPI
ml CUDA
ml cuDNN
ml NCCL
ml git

echo "START TIME: $(date)"

# Code Base path
MEGATRON_DEEPSPEED_REPO="$INSTALLATION_DIR"

#### Input data ####
# Should be on scratch filesystem!
VOCAB_FILE=/p/scratch/ccstdl/xu17/jz/Megatron-DeepSpeed/data/gpt2-vocab.json
MERGE_FILE=/p/scratch/ccstdl/xu17/jz/Megatron-DeepSpeed/data/gpt2-merges.txt
# Path to a singular, preprocessed dataset.
DATA_PATH=/p/scratch/ccstdl/xu17/jz/Megatron-DeepSpeed/data/meg-gpt2-oscar-en-10k_text_document


#### Output paths ####
DATA_OUTPUT_PATH="$RUN_DIR"/"${0##*/}"
CHECKPOINT_PATH="$DATA_OUTPUT_PATH"/checkpoints
TENSORBOARD_PATH="$DATA_OUTPUT_PATH"/tensorboard
CODECARBON_PATH="$DATA_OUTPUT_PATH"/codecarbon
CACHE_DIR="$DATA_OUTPUT_PATH/.cache"
LOGS_PATH="$DATA_OUTPUT_PATH"/logs
TORCHELASTIC_ERROR_FILE=$LOGS_PATH/torch_dirtribute_error.txt

mkdir -p $LOGS_PATH

# copy this batch script into log directory for reproducibility
if [ -e "$0" ]; then
    cp -p "$0" "$LOGS_PATH/batch-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.sh"
fi


#### Environment variables ####
export LOAD_CHECKPOINTS=false
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CXX=g++
export CC=gcc
# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# handle timeouts
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
# setting IB for out of band communication
export NCCL_SOCKET_IFNAME=ib0

# For debugging 
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO


##### Network parameters #####
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Allow communication over InfiniBand cells.
MASTER_ADDR="${MASTER_ADDR}i"
# Get IP for hostname.
MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
MASTER_PORT=29400

cd $MEGATRON_DEEPSPEED_REPO
CLEAN_PREV_JIT_BUILD=0
rm -f megatron/fused_kernels/build/lock
# Not rebuilding fused kernels have led to errors, so they are
# optionally deleted.
((CLEAN_PREV_JIT_BUILD)) && rm -rf megatron/fused_kernels/{build,__pycache__}


# MODEL_SIZE=6_7
# ##### Parallel model layouting #####
# # This is an example configuration, not an optimized one
# GPUS_PER_NODE=4          # Minimum required number of GPUs: PP_SIZE * TP_SIZE 
# NNODES=1
# PP_SIZE=2                # NLAYERS must be a multiple of PP_SIZE here
# TP_SIZE=2                # TP_SIZE <= GPUS_PER_NODE (preferred)
# # Gradient accumulation steps, used here to compute a reasonable
# # global batch size, not given to training code.
# GAS=8
# MICRO_BATCH_SIZE=2
# GLOBAL_BATCH_SIZE=$((( NNODES * GPUS_PER_NODE / (PP_SIZE * TP_SIZE) ) * MICRO_BATCH_SIZE * GAS))


# #### Notes ####
# # NGPUS = NNODES * GPUS_PER_NODE = TP_SIZE * PP_SIZE * DP_SIZE.
# # Given NNODES, GPUS_PER_NODE, PP_SIZE and TP_SIZE. DP_SIZE is
# # inferred automatically as NGPUS/(PP_SIZE * TP_SIZE).
# # MINI_BATCH_SIZE = MICRO_BATCH_SIZE * GAS
# # GLOBAL_BATCH_SIZE = MINI_BATCH_SIZE * DP_SIZE
# # Given MICRO_BATCH_SIZE and GLOBAL_BATCH_SIZE. DP_SIZE has been
# # inferred as described above.
# # GAS = GLOBAL_BATCH_SIZE/(DP_SIZE * MICRO_BATCH_SIZE)
# # From GAS formula ==> GLOBAL_BATCH_SIZE has to be divisible by MICRO_BATCH_SIZE*DP_size


# #### Hyperparameters ####
# NLAYERS=32
# NHIDDEN=4096
# NHEADS=32
# SEQ_LEN=2048
# VOCAB_SIZE=50257

# SAVE_INTERVAL=3000
# LOG_INTERVAL=1
# EVAL_INTERVAL=1000

# TRAIN_SAMPLES=69_335_938
# TRAIN_TOKENS=142_000_000_000

# LR_DECAY_SAMPLES=126_953_125
# LR_WARMUP_SAMPLES=183_105

LOG_INTERVAL=1
EVAL_INTERVAL=500

GPUS_PER_NODE=4
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
TP_SIZE=2
PP_SIZE=2

NLAYERS=16
NHIDDEN=1024
NHEADS=16
SEQ_LEN=512
VOCAB_SIZE=50257
NNODES=2
LR_WARMUP_SAMPLES=5000
LR_DECAY_SAMPLES=12000
SAVE_INTERVAL=2000

TRAIN_SAMPLES=10_000_000



OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1.2e-4 \
    --min-lr 1.2e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples $LR_DECAY_SAMPLES \
    --lr-warmup-samples $LR_WARMUP_SAMPLES \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

EXIT_OPTS=" \
    --exit-duration-in-mins 58 \
    "
    	

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --embed-layernorm \
    --fp16 \
    --seed 42 \
    --checkpoint-activations \
    --init-method-std 0.0048 \
    --loss-scale 12 \
    --clip-grad 1.0 \
    $OPTIMIZER_ARGS \
    $EXIT_OPTS \
    "
    # --train-tokens $TRAIN_TOKENS \


# GPT_ARGS=" \
#     --num-layers $NLAYERS \
#     --hidden-size $NHIDDEN \
#     --num-attention-heads $NHEADS \
#     --seq-length $SEQ_LEN \
#     --max-position-embeddings $SEQ_LEN \
#     --micro-batch-size $MICRO_BATCH_SIZE \
#     --global-batch-size $GLOBAL_BATCH_SIZE \
#     --train-samples $TRAIN_SAMPLES \
#     --train-tokens $TRAIN_TOKENS \
#     --vocab-file $VOCAB_FILE \
#     --merge-file $MERGE_FILE \
#     --fp16 \
#     --seed 42 \
#     --checkpoint-activations \
#     --init-method-std 0.0048 \
#     --loss-scale 12 \
#     --clip-grad 1.0 \
#     $OPTIMIZER_ARGS \
#     $EXIT_OPTS \
#     "

OUTPUT_ARGS=" \
    --log-interval $LOG_INTERVAL \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters 5 \
    --codecarbon-dir $CODECARBON_PATH \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=1


config_json="$RUN_DIR/ds_config_$SLURM_JOBID.json"
# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --max_restarts 0 \
    --tee 3 \
    "


# export LAUNCHER="python -u -m torch.distributed.run \
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --rdzv_endpoint 127.0.0.1:$MASTER_PORT \
#     --rdzv_backend c10d \
#     --max_restarts 0 \
#     --tee 3 \
#     "

export CMD=" \
    `pwd`/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend nccl \
    $DEEPSPEED_ARGS \
    "

if [ "$LOAD_CHECKPOINTS" = true ] ; then
    export CMD="$CMD\
        --load $CHECKPOINT_PATH \
        "
fi

echo $CMD

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "
clear; srun $SRUN_ARGS --jobid $SLURM_JOBID \
        bash -c "$LAUNCHER --node_rank \$SLURM_PROCID $CMD" 2>&1 \
    | tee -a "$LOGS_PATH"/main_log.txt

echo "END TIME: $(date)"