#!/bin/bash

# Enable logging
set -euo pipefail
set -x

# Set installation and run directories
INSTALLATION_DIR="/p/scratch/ccstdl/xu17/jz/Megatron-DeepSpeed"
RUN_DIR="/p/scratch/ccstdl/xu17/jz/Megatron-DeepSpeed/output"

mkdir -p "$RUN_DIR"

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
VOCAB_FILE=/p/scratch/ccstdl/xu17/jz/Megatron-DeepSpeed/data/gpt2-vocab.json
MERGE_FILE=/p/scratch/ccstdl/xu17/jz/Megatron-DeepSpeed/data/gpt2-merges.txt
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

if [ -e "$0" ]; then
    cp -p "$0" "$LOGS_PATH/batch-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.sh"
fi

#### Environment variables ####
export LOAD_CHECKPOINTS=false
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CXX=g++
export CC=gcc
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
export NCCL_SOCKET_IFNAME=ib0

# For debugging 
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO

##### Network parameters #####
GPUS_PER_NODE=4
PP_SIZE=2
TP_SIZE=2
GAS=64
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=$(( GPUS_PER_NODE / (PP_SIZE * TP_SIZE) * MICRO_BATCH_SIZE * GAS))

#### Hyperparameters ####
NLAYERS=32
NHIDDEN=4096
NHEADS=32
SEQ_LEN=2048
VOCAB_SIZE=50257

SAVE_INTERVAL=3000
LOG_INTERVAL=1
EVAL_INTERVAL=1000

TRAIN_SAMPLES=69_335_938
TRAIN_TOKENS=142_000_000_000

LR_DECAY_SAMPLES=126_953_125
LR_WARMUP_SAMPLES=183_105

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
    --train-tokens $TRAIN_TOKENS \
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

config_json="$RUN_DIR/ds_config.json"
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
    export CMD="$CMD --load $CHECKPOINT_PATH"
fi

echo $CMD

# Execute the command directly
python -u $CMD 2>&1 | tee -a "$LOGS_PATH"/main_log.txt

echo "END TIME: $(date)"
