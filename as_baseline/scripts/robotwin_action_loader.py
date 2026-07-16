"""Action loader for RoboTwin bimanual converted annotations."""

from __future__ import annotations

import math

import mediapy
import numpy as np


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


def get_bimanual_actions(data: dict, fps_downsample_ratio: int) -> np.ndarray:
    if "action" in data:
        return np.asarray(data["action"], dtype=np.float64)[::fps_downsample_ratio]

    state = np.asarray(data["state"], dtype=np.float64)[::fps_downsample_ratio]
    gripper = np.asarray(data["continuous_gripper_state"], dtype=np.float64)[::fps_downsample_ratio]
    actions = np.zeros((len(state) - 1, 14), dtype=np.float64)

    for k in range(1, len(state)):
        for arm_i in range(2):
            state_offset = arm_i * 6
            action_offset = arm_i * 7
            prev_xyz = state[k - 1, state_offset : state_offset + 3]
            prev_rotm = euler_to_rotm(state[k - 1, state_offset + 3 : state_offset + 6])
            curr_xyz = state[k, state_offset : state_offset + 3]
            curr_rotm = euler_to_rotm(state[k, state_offset + 3 : state_offset + 6])
            rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
            rel_rotm = prev_rotm.T @ curr_rotm
            actions[k - 1, action_offset : action_offset + 3] = rel_xyz
            actions[k - 1, action_offset + 3 : action_offset + 6] = rotm_to_euler(rel_rotm)
            actions[k - 1, action_offset + 6] = gripper[k, arm_i]

    return actions


def load_robotwin_bimanual_action_fn():
    def load_fn(json_data: dict, video_path: str, args) -> dict:
        actions = get_bimanual_actions(json_data, args.fps_downsample_ratio)
        scaler = np.array(
            [args.action_scaler, args.action_scaler, args.action_scaler, args.action_scaler, args.action_scaler, args.action_scaler, args.gripper_scale]
            * 2,
            dtype=np.float64,
        )
        actions *= scaler

        video_array = mediapy.read_video(video_path)
        img_array = video_array[args.start_frame_idx]
        if args.resolution != "none":
            h, w = map(int, args.resolution.split(","))
            img_array = mediapy.resize_image(img_array, (h, w))

        return {
            "actions": actions,
            "initial_frame": img_array,
            "video_array": video_array,
            "video_path": video_path,
        }

    return load_fn
