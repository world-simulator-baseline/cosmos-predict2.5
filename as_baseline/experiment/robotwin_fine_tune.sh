NCCL_SHM_DISABLE=1 .venv/bin/torchrun --nproc_per_node=2 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- \
  experiment=robotwin_14d_finetune \
  job.name=robotwin_14d_finetune_pretrained_2gpu_fix \
  trainer.max_iter=1000 \
  checkpoint.save_iter=1000 \
  trainer.straggler_detection.enabled=False \
  model.config.net._target_=as_baseline.scripts.action_embedder_init_weight_patch.RobotwinActionChunkConditionedDiT \
  model.config.fsdp_shard_size=2 \
  model.config.ema.enabled=False \
  checkpoint.load_path=308eb96c-c4c0-4a06-9cc1-103a43beff28 \
  checkpoint.load_training_state=False \
  checkpoint.strict_resume=False \
  checkpoint.load_from_object_store.enabled=False \
  checkpoint.save_to_object_store.enabled=False \
  trainer.callbacks.dataloader_speed.save_s3=False \
  trainer.callbacks.device_monitor.save_s3=False \
  trainer.callbacks.every_n_sample_ema.save_s3=False \
  trainer.callbacks.every_n_sample_reg.save_s3=False \
  trainer.callbacks.heart_beat.save_s3=False \
  trainer.callbacks.iter_speed.save_s3=False \
  trainer.callbacks.wandb.save_s3=False \
  trainer.callbacks.wandb_10x.save_s3=False \
  ~dataloader_train.dataloaders

# export 
# CUDA_LAUNCH_BLOCKING=1 bash as_baseline/experiment/robotwin_fine_tune.sh
# checkpoint.load_path=308eb96c-c4c0-4a06-9cc1-103a43beff28 #if loading from pretrained model