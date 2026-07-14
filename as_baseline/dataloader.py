#!/usr/bin/env python3
"""Convert RoboTwin HDF5 episodes to Cosmos action-conditioned JSON annotations.

The Cosmos action-conditioned training dataset reads:
  output_root/
    annotation/train/*.json
    annotation/val/*.json
    videos/<task>/<episode>.mp4

Each JSON contains one episode with a video path plus bimanual end-effector
state and gripper arrays. The state is ordered as:
  [left_xyz, left_rpy, right_xyz, right_rpy]
and actions are ordered as:
  [left_delta_xyz, left_delta_rpy, left_gripper,
   right_delta_xyz, right_delta_rpy, right_gripper]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) to XYZ roll/pitch/yaw radians."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q.T

    norm = np.sqrt(w * w + x * x + y * y + z * z)
    norm = np.where(norm == 0, 1.0, norm)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1.0, np.sign(sinp) * (math.pi / 2.0), np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1)


def euler_to_rotm(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rotm_to_euler(rotm: np.ndarray) -> np.ndarray:
    pitch = math.atan2(-rotm[2, 0], math.sqrt(rotm[0, 0] ** 2 + rotm[1, 0] ** 2))
    roll = math.atan2(rotm[2, 1], rotm[2, 2])
    yaw = math.atan2(rotm[1, 0], rotm[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64)


def arm_motion_score(endpose: np.ndarray, gripper: np.ndarray) -> float:
    """Estimate which arm is active from motion in pose and gripper state."""
    if len(endpose) < 2:
        return 0.0
    translation = np.linalg.norm(np.diff(endpose[:, :3], axis=0), axis=1).sum()
    rotation = np.linalg.norm(np.diff(endpose[:, 3:7], axis=0), axis=1).sum()
    gripper_motion = np.abs(np.diff(gripper)).sum()
    return float(translation + 0.25 * rotation + 0.01 * gripper_motion)


def get_arm_motion_scores(f: h5py.File) -> dict[str, float]:
    scores = {}
    for arm in ["left", "right"]:
        endpose = np.asarray(f[f"endpose/{arm}_endpose"], dtype=np.float64)
        gripper = np.asarray(f[f"endpose/{arm}_gripper"], dtype=np.float64)
        scores[arm] = arm_motion_score(endpose, gripper)
    return scores


def read_episode(h5_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    states = []
    grippers = []
    with h5py.File(h5_path, "r") as f:
        arm_scores = get_arm_motion_scores(f)
        for arm in ["left", "right"]:
            endpose = np.asarray(f[f"endpose/{arm}_endpose"], dtype=np.float64)
            gripper = np.asarray(f[f"endpose/{arm}_gripper"], dtype=np.float64)

            if endpose.ndim != 2 or endpose.shape[1] != 7:
                raise ValueError(f"{h5_path}: expected endpose/{arm}_endpose shape (T, 7), got {endpose.shape}")
            if gripper.ndim != 1 or gripper.shape[0] != endpose.shape[0]:
                raise ValueError(f"{h5_path}: endpose/{arm}_gripper shape {gripper.shape} does not match endpose length {endpose.shape[0]}")
            if states and endpose.shape[0] != states[0].shape[0]:
                raise ValueError(f"{h5_path}: left/right arm episode lengths do not match")

            xyz = endpose[:, :3]
            rpy = quat_to_euler(endpose[:, 3:7])
            states.append(np.concatenate([xyz, rpy], axis=-1))
            grippers.append(gripper)

    return np.concatenate(states, axis=-1), np.stack(grippers, axis=-1), arm_scores


def compute_relative_actions(state: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Store unscaled relative bimanual actions; Dataset_3D recomputes this during training."""
    num_arms = state.shape[1] // 6
    action = np.zeros((len(state) - 1, num_arms * 7), dtype=np.float64)
    for k in range(1, len(state)):
        for arm_i in range(num_arms):
            state_offset = arm_i * 6
            action_offset = arm_i * 7
            prev_xyz = state[k - 1, state_offset : state_offset + 3]
            prev_rotm = euler_to_rotm(state[k - 1, state_offset + 3 : state_offset + 6])
            curr_xyz = state[k, state_offset : state_offset + 3]
            curr_rotm = euler_to_rotm(state[k, state_offset + 3 : state_offset + 6])
            rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
            rel_rotm = prev_rotm.T @ curr_rotm
            action[k - 1, action_offset : action_offset + 3] = rel_xyz
            action[k - 1, action_offset + 3 : action_offset + 6] = rotm_to_euler(rel_rotm)
            action[k - 1, action_offset + 6] = gripper[k, arm_i]
    return action


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported video mode: {mode}")


def episode_index(path: Path) -> int:
    stem = path.stem
    if stem.startswith("episode"):
        return int(stem.removeprefix("episode"))
    raise ValueError(f"Cannot parse episode index from {path}")


def collect_episodes(source_root: Path) -> list[tuple[str, Path, Path]]:
    episodes = []
    if (source_root / "dataset").is_dir():
        h5_paths = source_root.glob("dataset/*/*/data/episode*.hdf5")
    else:
        h5_paths = source_root.glob("*/*/data/episode*.hdf5")

    def sort_key(path: Path) -> tuple[str, str, int]:
        run_dir = path.parent.parent
        return run_dir.parent.name, run_dir.name, episode_index(path)

    for h5_path in sorted(h5_paths, key=sort_key):
        run_dir = h5_path.parent.parent
        task = run_dir.parent.name
        video_path = run_dir / "video" / f"{h5_path.stem}.mp4"
        if video_path.exists():
            episodes.append((task, h5_path, video_path))
    return episodes


def split_name(i: int, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"
    period = max(int(round(1.0 / val_ratio)), 1)
    return "val" if i % period == 0 else "train"


def convert(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    episodes = collect_episodes(source_root)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise RuntimeError(f"No paired episode*.hdf5 and video/episode*.mp4 files found under {source_root}")

    counts = {"train": 0, "val": 0}
    skipped = 0
    for global_i, (task, h5_path, video_src) in enumerate(episodes):
        try:
            state, gripper, arm_scores = read_episode(h5_path)
        except Exception as exc:
            skipped += 1
            print(f"[skip] {h5_path}: {exc}")
            continue

        split = split_name(global_i, args.val_ratio)
        episode_name = h5_path.stem
        rel_video = Path("videos") / task / f"{episode_name}.mp4"
        video_dst = output_root / rel_video
        ann_name = f"{task}_{episode_name}.json"
        ann_path = output_root / "annotation" / split / ann_name


        link_or_copy(video_src, video_dst, args.video_mode)
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        action = compute_relative_actions(state, gripper)
        payload = {
            "task": task,
            "texts": [task.replace("_", " ")],
            "videos": [{"video_path": rel_video.as_posix()}],
            "state": state.tolist(),
            "continuous_gripper_state": gripper.tolist(),
            "action": action.tolist(),
            "episode_id": f"{task}/{episode_name}",
            "original_path": str(h5_path),
            "episode_metadata": {
                "episode_id": f"{task}/{episode_name}",
                "task": task,
                "source_hdf5": str(h5_path),
                "source_video": str(video_src),
                "arms": ["left", "right"],
                "arm_motion_scores": arm_scores,
                "state_format": ["left_xyzrpy", "right_xyzrpy"],
                "action_format": ["left_delta_xyzrpy_gripper", "right_delta_xyzrpy_gripper"],
                "action_dim": 14,
                "quat_order": "wxyz",
                "is_eval": split == "val",
            },
        }
        with ann_path.open("w") as f:
            json.dump(payload, f)

        counts[split] += 1

    print(f"source_root: {source_root}")
    print(f"output_root: {output_root}")
    print(f"converted: train={counts['train']} val={counts['val']} skipped={skipped}")
    print("action format: bimanual 14D [left 7D, right 7D]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("as_baseline/RoboTwin2.0_480_640"),
        help="RoboTwin root containing dataset/<task>/<run>/{data,video}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination root for Cosmos-style annotation/ and videos/ directories.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Deterministic validation ratio.")
    parser.add_argument("--video-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit conversion for smoke tests.")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
