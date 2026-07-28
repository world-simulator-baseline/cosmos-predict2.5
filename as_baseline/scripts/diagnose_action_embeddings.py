"""Extract and compare action embeddings from a Cosmos Predict2.5 DCP checkpoint.

This is a lightweight CPU diagnostic: it loads only the eight Action-MLP
tensors, reconstructs actions exactly as Dataset_3D does, and does not
materialize the 2B-parameter DiT.

Example:
    .venv/bin/python as_baseline/scripts/diagnose_action_embeddings.py \
      --checkpoint /path/to/checkpoints/iter_000002000 \
      --annotation-dir as_baseline/converted_dataset3/train/annotation \
      --output-dir as_baseline/notes/action_embedding_debug/iter_000002000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from as_baseline.scripts.check_action_encoder_checkpoint import ACTION_KEYS, load_action_tensors

BRANCHES = ("B_D", "B_3D")


def resolve_model_checkpoint(path: Path) -> Path:
    """Accept either an iteration directory or its model subdirectory."""
    path = path.expanduser().resolve()
    candidates = (path, path / "model")
    for candidate in candidates:
        if (candidate / ".metadata").is_file():
            return candidate
    raise FileNotFoundError(
        f"No DCP .metadata found in {path} or {path / 'model'}. "
        "Point --checkpoint at iter_XXXXXXXX or iter_XXXXXXXX/model."
    )


def resolve_annotations(path: Path) -> list[Path]:
    """Accept a JSON file, an annotation directory, or a dataset/split root."""
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix != ".json":
            raise ValueError(f"Expected a JSON annotation, got {path}")
        return [path]

    candidates = (path, path / "annotation", path / "train" / "annotation")
    for candidate in candidates:
        files = sorted(candidate.glob("*.json"))
        if files:
            return files
    raise FileNotFoundError(f"No JSON annotations found under {path}")


def euler_to_rotation(euler: np.ndarray) -> np.ndarray:
    """Match imaginaire.utils.dataset_utils.euler2rotm (Rz @ Ry @ Rx)."""
    x, y, z = euler
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def rotation_to_euler(rotation: np.ndarray) -> np.ndarray:
    """Match imaginaire.utils.dataset_utils.rotm2euler."""
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    if sy >= 1e-6:
        x = math.atan2(rotation[2, 1], rotation[2, 2])
        y = math.atan2(-rotation[2, 0], sy)
        z = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        x = math.atan2(-rotation[1, 2], rotation[1, 1])
        y = math.atan2(-rotation[2, 0], sy)
        z = 0.0
    return np.array([x, y, z])


def actions_from_annotation(
    annotation: Path,
    start_frame: int,
    chunk_size: int,
    gripper_rescale_factor: float,
    accumulate_action: bool,
) -> torch.Tensor:
    """Reproduce Dataset_3D._get_actions, including the training scaler."""
    with annotation.open() as file:
        label = json.load(file)

    states = np.asarray(label["state"], dtype=np.float64)
    grippers = np.asarray(label["continuous_gripper_state"], dtype=np.float64)
    if grippers.ndim == 1:
        grippers = grippers[:, None]

    stop_frame = start_frame + chunk_size + 1
    if start_frame < 0 or stop_frame > len(states):
        raise ValueError(
            f"{annotation} has {len(states)} frames, but frames [{start_frame}, {stop_frame}) are required."
        )
    states = states[start_frame:stop_frame]
    grippers = grippers[start_frame:stop_frame]

    if states.shape[1] % 6:
        raise ValueError(f"State dimension must be a multiple of 6, got {states.shape[1]}")
    num_arms = states.shape[1] // 6
    if grippers.shape[1] != num_arms:
        raise ValueError(f"Expected {num_arms} gripper values per frame, got {grippers.shape[1]}")

    actions = np.zeros((chunk_size, num_arms * 7), dtype=np.float64)
    base_xyz = [states[0, arm * 6 : arm * 6 + 3].copy() for arm in range(num_arms)]
    base_rot = [euler_to_rotation(states[0, arm * 6 + 3 : arm * 6 + 6]) for arm in range(num_arms)]

    for step in range(1, chunk_size + 1):
        for arm in range(num_arms):
            state_offset = arm * 6
            action_offset = arm * 7
            current_xyz = states[step, state_offset : state_offset + 3]
            current_rot = euler_to_rotation(states[step, state_offset + 3 : state_offset + 6])

            if accumulate_action:
                reference_xyz = base_xyz[arm]
                reference_rot = base_rot[arm]
            else:
                reference_xyz = states[step - 1, state_offset : state_offset + 3]
                reference_rot = euler_to_rotation(states[step - 1, state_offset + 3 : state_offset + 6])

            actions[step - 1, action_offset : action_offset + 3] = reference_rot.T @ (current_xyz - reference_xyz)
            actions[step - 1, action_offset + 3 : action_offset + 6] = rotation_to_euler(reference_rot.T @ current_rot)
            actions[step - 1, action_offset + 6] = grippers[step, arm]

        if accumulate_action and step % 4 == 0:
            for arm in range(num_arms):
                state_offset = arm * 6
                base_xyz[arm] = states[step, state_offset : state_offset + 3].copy()
                base_rot[arm] = euler_to_rotation(states[step, state_offset + 3 : state_offset + 6])

    scaler = np.asarray(
        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, gripper_rescale_factor] * num_arms,
        dtype=np.float64,
    )
    return torch.from_numpy(actions * scaler).float()


def reshape_for_action_mlp(actions: torch.Tensor, in_features: int) -> tuple[torch.Tensor, int]:
    action_dim = actions.shape[1]
    if in_features % action_dim:
        raise ValueError(
            f"MLP in_features={in_features} is not divisible by annotation action_dim={action_dim}. "
            "The checkpoint and dataset likely use different action dimensions."
        )
    actions_per_token = in_features // action_dim
    if actions.shape[0] % actions_per_token:
        raise ValueError(
            f"chunk_size={actions.shape[0]} is not divisible by checkpoint actions_per_token={actions_per_token}."
        )
    return actions.reshape(-1, in_features), actions_per_token


def encode_branch(inputs: torch.Tensor, tensors: dict[str, torch.Tensor], branch: str) -> torch.Tensor:
    prefix = f"net.action_embedder_{branch}"
    hidden = F.linear(
        inputs.float(),
        tensors[f"{prefix}.fc1.weight"].float(),
        tensors[f"{prefix}.fc1.bias"].float(),
    )
    hidden = F.gelu(hidden, approximate="tanh")
    return F.linear(
        hidden,
        tensors[f"{prefix}.fc2.weight"].float(),
        tensors[f"{prefix}.fc2.bias"].float(),
    )


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    value = value.float()
    return {
        "shape": list(value.shape),
        "mean": value.mean().item(),
        "std": value.std().item(),
        "rms": value.square().mean().sqrt().item(),
        "l2": value.norm().item(),
        "absmax": value.abs().max().item(),
        "token_std_rms": value.std(dim=0).square().mean().sqrt().item() if value.shape[0] > 1 else 0.0,
    }


def comparison(reference: torch.Tensor, candidate: torch.Tensor, action_delta_rms: float) -> dict[str, float]:
    reference = reference.float()
    candidate = candidate.float()
    delta = candidate - reference
    ref_flat = reference.flatten()
    candidate_flat = candidate.flatten()
    delta_rms = delta.square().mean().sqrt().item()
    cosine = F.cosine_similarity(ref_flat[None], candidate_flat[None]).item()
    return {
        "embedding_delta_rms": delta_rms,
        "embedding_relative_l2": (delta.norm() / reference.norm().clamp_min(1e-12)).item(),
        "embedding_cosine": cosine,
        "embedding_gain_vs_action_rms": delta_rms / max(action_delta_rms, 1e-12),
    }


def select_alternate(files: list[Path], primary: Path) -> Path | None:
    return next((path for path in files if path != primary), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="iter directory or its model subdirectory")
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help="JSON annotation, annotation directory, split root, or dataset root",
    )
    parser.add_argument("--annotation", type=Path, help="specific primary annotation JSON")
    parser.add_argument("--other-annotation", type=Path, help="specific annotation for cross-trajectory comparison")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--other-start-frame", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--gripper-rescale-factor", type=float, default=1.0)
    parser.add_argument("--accumulate-action", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("as_baseline/notes/action_embedding_debug"))
    args = parser.parse_args()

    checkpoint = resolve_model_checkpoint(args.checkpoint)
    annotation_files = resolve_annotations(args.annotation_dir)
    primary = args.annotation.expanduser().resolve() if args.annotation else annotation_files[0]
    alternate = (
        args.other_annotation.expanduser().resolve()
        if args.other_annotation
        else select_alternate(annotation_files, primary)
    )

    tensors = load_action_tensors(checkpoint)
    missing = [key for key in ACTION_KEYS if key not in tensors]
    if missing:
        raise ValueError(f"Missing action tensors: {missing}")

    original = actions_from_annotation(
        primary,
        args.start_frame,
        args.chunk_size,
        args.gripper_rescale_factor,
        args.accumulate_action,
    )
    generator = torch.Generator().manual_seed(args.seed)
    variants = {
        "original": original,
        "zero": torch.zeros_like(original),
        "reversed": original.flip(0),
        "temporal_shuffle": original[torch.randperm(len(original), generator=generator)],
    }
    if alternate is not None:
        variants["other_trajectory"] = actions_from_annotation(
            alternate,
            args.other_start_frame,
            args.chunk_size,
            args.gripper_rescale_factor,
            args.accumulate_action,
        )

    in_features = tensors["net.action_embedder_B_D.fc1.weight"].shape[1]
    if tensors["net.action_embedder_B_3D.fc1.weight"].shape[1] != in_features:
        raise ValueError("The two action-embedding branches have different input widths.")

    mlp_inputs: dict[str, torch.Tensor] = {}
    embeddings: dict[str, dict[str, torch.Tensor]] = {}
    actions_per_token = 0
    for name, actions in variants.items():
        mlp_inputs[name], actions_per_token = reshape_for_action_mlp(actions, in_features)
        embeddings[name] = {branch: encode_branch(mlp_inputs[name], tensors, branch) for branch in BRANCHES}

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "primary_annotation": str(primary),
        "alternate_annotation": str(alternate) if alternate else None,
        "chunk_size": args.chunk_size,
        "action_dim": original.shape[1],
        "actions_per_embedding_token": actions_per_token,
        "embedding_tokens": mlp_inputs["original"].shape[0],
        "accumulate_action": args.accumulate_action,
        "weight_diagnostics": {},
        "action_summaries": {name: tensor_summary(value) for name, value in variants.items()},
        "embedding_summaries": {},
        "comparisons_to_original": {},
    }

    for branch in BRANCHES:
        prefix = f"net.action_embedder_{branch}"
        fc1_weight = tensors[f"{prefix}.fc1.weight"]
        fc2_weight = tensors[f"{prefix}.fc2.weight"]
        report["weight_diagnostics"][branch] = {
            "fc1_nonzero": torch.count_nonzero(fc1_weight).item(),
            "fc1_numel": fc1_weight.numel(),
            "fc2_nonzero": torch.count_nonzero(fc2_weight).item(),
            "fc2_numel": fc2_weight.numel(),
            "action_independent_by_zero_fc2": torch.count_nonzero(fc2_weight).item() == 0,
        }
        report["embedding_summaries"][branch] = {
            name: tensor_summary(value[branch]) for name, value in embeddings.items()
        }
        report["comparisons_to_original"][branch] = {}
        for name in variants:
            if name == "original":
                continue
            action_delta_rms = (mlp_inputs[name] - mlp_inputs["original"]).square().mean().sqrt().item()
            report["comparisons_to_original"][branch][name] = {
                "action_delta_rms": action_delta_rms,
                **comparison(embeddings["original"][branch], embeddings[name][branch], action_delta_rms),
            }

    max_delta = max(
        comparison_values["embedding_delta_rms"]
        for branch_values in report["comparisons_to_original"].values()
        for comparison_values in branch_values.values()
    )
    dead_branches = [
        branch for branch, values in report["weight_diagnostics"].items() if values["action_independent_by_zero_fc2"]
    ]
    report["diagnosis"] = {
        "dead_branches": dead_branches,
        "all_tested_embeddings_identical": max_delta <= 1e-12,
        "max_embedding_delta_rms": max_delta,
        "note": (
            "A nonzero delta proves that the Action Encoder depends on action input; "
            "it does not by itself prove that the full DiT follows actions correctly."
        ),
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    features_path = output_dir / "features.pt"
    with report_path.open("w") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    torch.save(
        {
            "actions": variants,
            "mlp_inputs": mlp_inputs,
            "embeddings": embeddings,
            "metadata": {
                "checkpoint": str(checkpoint),
                "primary_annotation": str(primary),
                "alternate_annotation": str(alternate) if alternate else None,
            },
        },
        features_path,
    )

    print(f"checkpoint: {checkpoint}")
    print(f"primary annotation: {primary}")
    print(
        f"action shape={tuple(original.shape)}, actions/token={actions_per_token}, "
        f"embedding tokens={mlp_inputs['original'].shape[0]}"
    )
    for branch in BRANCHES:
        weights = report["weight_diagnostics"][branch]
        print(
            f"{branch}: fc2_nonzero={weights['fc2_nonzero']}/{weights['fc2_numel']} "
            f"action_independent={weights['action_independent_by_zero_fc2']}"
        )
        for name, values in report["comparisons_to_original"][branch].items():
            print(
                f"  original vs {name}: delta_rms={values['embedding_delta_rms']:.6e} "
                f"relative_l2={values['embedding_relative_l2']:.6e} "
                f"cosine={values['embedding_cosine']:.8f}"
            )
    print(f"report: {report_path}")
    print(f"features: {features_path}")
    return 2 if dead_branches else 0


if __name__ == "__main__":
    raise SystemExit(main())
