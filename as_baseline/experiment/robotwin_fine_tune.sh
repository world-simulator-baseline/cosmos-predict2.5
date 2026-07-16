NCCL_SHM_DISABLE=1 .venv/bin/torchrun --nproc_per_node=2 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  -- \
  experiment=robotwin_14d_finetune \
  job.name=robotwin_14d_finetune_pretrained_2gpu_smoke \
  trainer.max_iter=20 \
  trainer.straggler_detection.enabled=False \
  model.config.fsdp_shard_size=2 \
  model.config.ema.enabled=False \
  checkpoint.load_path=308eb96c-c4c0-4a06-9cc1-103a43beff28 \
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
