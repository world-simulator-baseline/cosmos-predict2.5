"""Run training while reusing the locally installed Cosmos-Reason checkpoint."""

import os
import runpy

import torch.distributed as dist

from cosmos_predict2._src.imaginaire.utils import checkpoint_db
from cosmos_predict2._src.predict2.action.datasets import dataset_local
from cosmos_predict2._src.predict2.text_encoders import text_encoder

_COSMOS_REASON = os.environ.get("COSMOS_REASON_CHECKPOINT_PATH")
_download_checkpoint = checkpoint_db.download_checkpoint


def _download_local(checkpoint_uri: str, check_exists: bool = True) -> str:
    if _COSMOS_REASON and "cosmos_reasoning1/" in checkpoint_uri:
        return _COSMOS_REASON
    return _download_checkpoint(checkpoint_uri, check_exists=check_exists)


checkpoint_db.download_checkpoint = _download_local
checkpoint_db.get_checkpoint_path = _download_local

_load_state_dict_from_folder = text_encoder.load_state_dict_from_folder


def _load_text_encoder_on_rank_zero(checkpoint_path: str) -> dict:
    rank_zero_only = os.environ.get("COSMOS_TEXT_ENCODER_RANK0_LOAD", "").lower() in {"1", "true", "yes"}
    if rank_zero_only and dist.is_initialized() and dist.get_rank() != 0:
        return {}
    return _load_state_dict_from_folder(checkpoint_path)


text_encoder.load_state_dict_from_folder = _load_text_encoder_on_rank_zero

_get_frames = dataset_local.Dataset_3D._get_frames


def _get_frames_from_dataset_root(self, label: dict, frame_ids, cam_id, pre_encode):
    video_path = label["videos"][cam_id]["video_path"]
    if os.path.isabs(video_path) and not os.path.exists(video_path):
        label["videos"][cam_id]["video_path"] = os.path.basename(video_path)
    return _get_frames(self, label, frame_ids, cam_id, pre_encode)


dataset_local.Dataset_3D._get_frames = _get_frames_from_dataset_root

if __name__ == "__main__":
    runpy.run_module("scripts.train", run_name="__main__")
