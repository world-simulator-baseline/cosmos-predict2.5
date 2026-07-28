"""Verify that a DCP checkpoint contains action-dependent encoder weights.

Run this after the first recovery checkpoint.  It deliberately loads only the
eight action-MLP tensors, rather than materializing the 2B-parameter model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner


ACTION_KEYS = (
    "net.action_embedder_B_D.fc1.weight",
    "net.action_embedder_B_D.fc1.bias",
    "net.action_embedder_B_D.fc2.weight",
    "net.action_embedder_B_D.fc2.bias",
    "net.action_embedder_B_3D.fc1.weight",
    "net.action_embedder_B_3D.fc1.bias",
    "net.action_embedder_B_3D.fc2.weight",
    "net.action_embedder_B_3D.fc2.bias",
)


def _dtype(metadata_entry) -> torch.dtype:
    properties = metadata_entry.properties
    return properties.dtype if isinstance(properties.dtype, torch.dtype) else getattr(torch, properties.dtype)


def load_action_tensors(checkpoint: Path) -> dict[str, torch.Tensor]:
    reader = FileSystemReader(checkpoint)
    metadata = reader.read_metadata()
    missing = [key for key in ACTION_KEYS if key not in metadata.state_dict_metadata]
    if missing:
        raise ValueError(f"Checkpoint is missing action tensors: {', '.join(missing)}")

    state = {
        key: torch.empty(tuple(metadata.state_dict_metadata[key].size), dtype=_dtype(metadata.state_dict_metadata[key]))
        for key in ACTION_KEYS
    }
    dcp.load(state, storage_reader=reader, planner=DefaultLoadPlanner(allow_partial_load=True))
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="DCP model directory (the directory containing .metadata)")
    parser.add_argument(
        "--require-action-dependent",
        action="store_true",
        help="fail if either MLP still has both fc1.weight and fc2.weight exactly zero",
    )
    args = parser.parse_args()

    tensors = load_action_tensors(args.checkpoint)
    for name, tensor in tensors.items():
        tensor_float = tensor.float()
        print(
            f"{name}: shape={tuple(tensor.shape)} nonzero={torch.count_nonzero(tensor).item()}/{tensor.numel()} "
            f"mean={tensor_float.mean().item():+.6e} std={tensor_float.std().item():.6e} "
            f"absmax={tensor_float.abs().max().item():.6e}"
        )

    dead = []
    for branch in ("B_D", "B_3D"):
        fc1 = tensors[f"net.action_embedder_{branch}.fc1.weight"]
        fc2 = tensors[f"net.action_embedder_{branch}.fc2.weight"]
        if torch.count_nonzero(fc1) == 0 and torch.count_nonzero(fc2) == 0:
            dead.append(branch)

    if args.require_action_dependent and dead:
        print(f"FAIL: action encoder branch(es) cannot depend on actions: {', '.join(dead)}", file=sys.stderr)
        return 1
    if args.require_action_dependent:
        print("PASS: both action encoder branches contain an action-dependent path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
