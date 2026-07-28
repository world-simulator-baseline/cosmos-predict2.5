export COSMOS_REASON_CHECKPOINT_PATH="${COSMOS_REASON_CHECKPOINT_PATH:-/data1/fangxuebin/models/hub/models--nvidia--Cosmos-Reason1-7B/snapshots/375e24000b24baed78f4618d3dd779e47cd96323}"
export COSMOS_TEXT_ENCODER_RANK0_LOAD="${COSMOS_TEXT_ENCODER_RANK0_LOAD:-1}"
export IMAGINAIRE_OUTPUT_ROOT=/data1/fangxuebin/models/

ACTION_EMBEDDER_INIT_METHOD="${ACTION_EMBEDDER_INIT_METHOD:-random}"
case "$ACTION_EMBEDDER_INIT_METHOD" in
  random)
    ACTION_EMBEDDER_TARGET=as_baseline.scripts.action_embedder_init_weight_patch.RobotwinActionChunkConditionedDiT
    DEFAULT_JOB_NAME=robotwin_14d_finetune_pretrained_4gpu_onetask_random
    ;;
  zero)
    ACTION_EMBEDDER_TARGET=as_baseline.scripts.action_embedder_zero_init_weight_patch.RobotwinActionChunkConditionedDiT
    DEFAULT_JOB_NAME=robotwin_14d_finetune_pretrained_4gpu_fc1_random_fc2_zero
    ;;
  *)
    echo "Unsupported ACTION_EMBEDDER_INIT_METHOD: $ACTION_EMBEDDER_INIT_METHOD (expected random or zero)" >&2
    exit 2
    ;;
esac

JOB_NAME="${JOB_NAME:-$DEFAULT_JOB_NAME}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-12343}"
MAX_ITER="${MAX_ITER:-10000}"
CHECKPOINT_SAVE_ITER="${CHECKPOINT_SAVE_ITER:-1000}"
ENABLE_EMA="${ENABLE_EMA:-true}"

.venv/bin/torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" -m as_baseline.scripts.train_local_checkpoints_launcher \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- \
  experiment=robotwin_14d_finetune \
  job.name="$JOB_NAME" \
  trainer.max_iter="$MAX_ITER" \
  checkpoint.save_iter="$CHECKPOINT_SAVE_ITER" \
  trainer.straggler_detection.enabled=False \
  model.config.net._target_="$ACTION_EMBEDDER_TARGET" \
  +model.config.tokenizer.vae_pth=/data1/fangxuebin/models/models--nvidia--Cosmos-Predict2.5-2B/snapshots/85f8ae7bfe8f5525c8d103429524dcf12f98bf7b/tokenizer.pth \
  model.config.fsdp_shard_size=4 \
  model.config.ema.enabled="$ENABLE_EMA" \
  checkpoint.load_path=308eb96c-c4c0-4a06-9cc1-103a43beff28  \
  checkpoint.load_training_state=False \
  checkpoint.strict_resume=False \
  checkpoint.load_from_object_store.enabled=False \
  checkpoint.save_to_object_store.enabled=False \
  trainer.callbacks.dataloader_speed.save_s3=False \
  trainer.callbacks.device_monitor.save_s3=False \
  "~trainer.callbacks.every_n_sample_ema" \
  "~trainer.callbacks.every_n_sample_reg" \
  trainer.callbacks.heart_beat.save_s3=False \
  trainer.callbacks.iter_speed.save_s3=False \
  trainer.callbacks.wandb.save_s3=False \
  trainer.callbacks.wandb_10x.save_s3=False \
  ~dataloader_train.dataloaders

# CUDA_VISIBLE_DEVICES=4,5,6,7 bash as_baseline/experiment/robotwin_fine_tune.sh
# checkpoint.load_path=308eb96c-c4c0-4a06-9cc1-103a43beff28 #if loading from pretrained model
