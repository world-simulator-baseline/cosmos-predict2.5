#!/usr/bin/env python3
"""Convert RoboTwin LeRobot episodes to Cosmos action-conditioned JSON annotations.

This reverses the field mapping used by data_convert/robotwin.py:
  joint_action/vector -> joint_abs
  [left_endpose, left_gripper, right_endpose, right_gripper] -> eef_abs

The output schema matches as_baseline/converted_dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data_adapter_robotwin import compute_relative_actions, quat_to_euler


def episode_index(path: Path) -> int:
    stem = path.stem
    if stem.startswith("episode_"):
        return int(stem.removeprefix("episode_"))
    raise ValueError(f"Cannot parse episode index from {path}")


def load_episode_records(meta_path: Path) -> dict[int, dict]:
    records = {}
    with meta_path.open() as f:
        for line in f:
            record = json.loads(line)
            records[int(record["episode_index"])] = record
    return records


def collect_episodes(source_root: Path, subset: str) -> list[tuple[str, Path, Path, dict]]:
    episodes = []
    if not source_root.is_dir():
        raise RuntimeError(f"source_root does not exist or is not a directory: {source_root}")

    for task_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        run_dir = task_dir / subset
        data_dir = run_dir / "data"
        meta_path = run_dir / "meta" / "episodes.jsonl"
        if not data_dir.exists() or not meta_path.exists():
            continue

        records = load_episode_records(meta_path)
        for parquet_path in sorted(data_dir.glob("chunk-*/episode_*.parquet"), key=episode_index):
            ep_idx = episode_index(parquet_path)
            video_path = (
                run_dir
                / "videos"
                / f"chunk-{ep_idx // 1000:03d}"
                / "observation.images.head"
                / f"episode_{ep_idx:06d}.mp4"
            )
            if video_path.exists():
                episodes.append((task_dir.name, parquet_path, video_path, records.get(ep_idx, {})))
    return episodes


def read_episode(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path, columns=["eef_abs"])
    eef_abs = np.stack(df["eef_abs"].to_numpy()).astype(np.float64)
    if eef_abs.ndim != 2 or eef_abs.shape[1] != 16:
        raise ValueError(f"{parquet_path}: expected eef_abs shape (T, 16), got {eef_abs.shape}")

    left_endpose = eef_abs[:, :7]
    left_gripper = eef_abs[:, 7]
    right_endpose = eef_abs[:, 8:15]
    right_gripper = eef_abs[:, 15]

    left_state = np.concatenate([left_endpose[:, :3], quat_to_euler(left_endpose[:, 3:7])], axis=-1)
    right_state = np.concatenate([right_endpose[:, :3], quat_to_euler(right_endpose[:, 3:7])], axis=-1)
    state = np.concatenate([left_state, right_state], axis=-1)
    gripper = np.stack([left_gripper, right_gripper], axis=-1)
    return state, gripper


def convert(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    episodes = collect_episodes(source_root, args.subset)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise RuntimeError(f"No paired LeRobot parquet and head-camera mp4 files found under {source_root}")

    counts = {"train": 0, "val": 0}
    skipped = 0
    for global_i, (task, parquet_path, video_src, record) in enumerate(episodes):
        try:
            state, gripper = read_episode(parquet_path)
        except Exception as exc:
            skipped += 1
            print(f"[skip] {parquet_path}: {exc}")
            continue

        ep_idx = episode_index(parquet_path)
        episode_name = f"episode{ep_idx}"
        is_eval = global_i % 50 in range(40, 50)
        split = "val" if is_eval else "train"
        video_dst = output_root / split / "videos" / f"{task}_{episode_name}.mp4"
        ann_path = output_root / split / "annotation" / f"{task}_{episode_name}.json"

        video_dst.parent.mkdir(parents=True, exist_ok=True)
        if not video_dst.exists() and not video_dst.is_symlink():
            os.symlink(video_src.resolve(), video_dst)

        ann_path.parent.mkdir(parents=True, exist_ok=True)
        action = compute_relative_actions(state, gripper)
        payload = {
            "task": task,
            "texts": [task.replace("_", " ")],
            "videos": [{"video_path": video_dst.as_posix()}],
            "state": state.tolist(),
            "continuous_gripper_state": gripper.tolist(),
            "action": action.tolist(),
            "episode_id": f"{task}/{episode_name}",
            "original_path": str(parquet_path),
            "episode_metadata": {
                "episode_id": f"{task}/{episode_name}",
                "task": task,
                "source_parquet": str(parquet_path),
                "source_video": str(video_src),
                "instruction": record.get("instruction"),
                "action_format": ["left_delta_xyzrpy_gripper", "right_delta_xyzrpy_gripper"],
                "action_dim": 14,
                "is_eval": is_eval,
            },
        }
        with ann_path.open("w") as f:
            json.dump(payload, f, indent=4)
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
        default=Path("/data1/wangzequn/code/dataset/RoboTwin2.0_640_480_lerobot"),
        help="RoboTwin LeRobot root containing <task>/<subset>/{data,videos,meta}.",
    )
    parser.add_argument("--subset", default="aloha-agilex_clean_50", help="Per-task LeRobot subset directory.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination root for Cosmos-style annotations/ and videos/ directories.",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit conversion for smoke tests.")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
