"""
Convert RoboCasa v0.2 HDF5 datasets (human + MimicGen) into a single zarr file
compatible with OAT's ZarrDataset / ReplayBuffer.
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import numpy as np
import h5py
import zarr
from tqdm import tqdm

from oat.common.replay_buffer import ReplayBuffer


# Keys to extract from HDF5 obs, mapped to zarr key and target dtype
OBS_KEYS = {
    "robot0_agentview_left_image":  ("robot0_agentview_left_image",  np.uint8),
    "robot0_agentview_right_image": ("robot0_agentview_right_image", np.uint8),
    "robot0_eye_in_hand_image":     ("robot0_eye_in_hand_image",     np.uint8),
    "robot0_eef_pos":               ("robot0_eef_pos",               np.float32),
    "robot0_eef_quat":              ("robot0_eef_quat",              np.float32),
    "robot0_gripper_qpos":          ("robot0_gripper_qpos",          np.float32),
    "robot0_base_pos":              ("robot0_base_pos",              np.float32),
    "robot0_base_quat":             ("robot0_base_quat",             np.float32),
}

IMAGE_KEYS = {
    "robot0_agentview_left_image",
    "robot0_agentview_right_image",
    "robot0_eye_in_hand_image",
}


def read_demos_from_hdf5(hdf5_path, max_demos, label=""):
    """Read up to max_demos episodes from an HDF5 file."""
    episodes = []
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )
        n_available = len(demo_keys)
        n_to_read = min(max_demos, n_available)
        print(f"[{label}] {n_available} demos available, reading {n_to_read}")
        if n_to_read < max_demos:
            print(f"  WARNING: requested {max_demos} but only {n_available} available")

        for i in tqdm(range(n_to_read), desc=f"Reading {label}"):
            demo = f[f"data/{demo_keys[i]}"]
            ep = {
                "action": demo["actions"][:].astype(np.float32),
            }
            for hdf5_key, (zarr_key, dtype) in OBS_KEYS.items():
                arr = demo["obs"][hdf5_key][:]
                if arr.dtype != dtype:
                    arr = arr.astype(dtype)
                ep[zarr_key] = arr
            episodes.append(ep)

    return episodes, n_to_read


def convert_robocasa_to_zarr(
    human_hdf5_path,
    mg_hdf5_path,
    output_zarr_path,
    n_human=50,
    n_mg=150,
):
    """Convert RoboCasa HDF5 datasets to a single zarr."""
    # Read episodes
    human_eps, n_human_read = read_demos_from_hdf5(
        human_hdf5_path, n_human, label="Human"
    )
    mg_eps, n_mg_read = read_demos_from_hdf5(
        mg_hdf5_path, n_mg, label="MimicGen"
    )

    all_episodes = human_eps + mg_eps
    n_total_eps = len(all_episodes)
    print(f"\nTotal episodes: {n_total_eps} ({n_human_read} human + {n_mg_read} MimicGen)")

    # Build ReplayBuffer
    replay_buffer = ReplayBuffer.create_empty_zarr()
    for ep in tqdm(all_episodes, desc="Building zarr"):
        replay_buffer.add_episode(ep)

    print(f"Total steps: {replay_buffer.n_steps}")
    print(f"Total episodes: {replay_buffer.n_episodes}")

    # Save with appropriate chunking
    if os.path.exists(output_zarr_path):
        import shutil
        shutil.rmtree(output_zarr_path)
    os.makedirs(output_zarr_path, exist_ok=True)

    n_steps = replay_buffer.n_steps
    chunks = {
        "action": (n_steps, 12),
        "robot0_eef_pos": (n_steps, 3),
        "robot0_eef_quat": (n_steps, 4),
        "robot0_gripper_qpos": (n_steps, 2),
        "robot0_base_pos": (n_steps, 3),
        "robot0_base_quat": (n_steps, 4),
        "robot0_agentview_left_image": (1, 128, 128, 3),
        "robot0_agentview_right_image": (1, 128, 128, 3),
        "robot0_eye_in_hand_image": (1, 128, 128, 3),
    }
    replay_buffer.save_to_path(
        output_zarr_path,
        chunks=chunks,
        compressors="disk",
    )
    print(f"\nSaved to {output_zarr_path}")

    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    root = zarr.open(output_zarr_path, "r")
    print(root.tree())
    print()
    for key in sorted(root["data"].keys()):
        arr = root["data"][key]
        print(f"data/{key}: shape={arr.shape}, dtype={arr.dtype}, chunks={arr.chunks}")
    print()
    ends = root["meta"]["episode_ends"][:]
    print(f"meta/episode_ends: shape={ends.shape}, dtype={ends.dtype}")
    print(f"First 5 episode_ends: {ends[:5]}")
    print(f"Last episode_end: {ends[-1]}")
    print(f"\nTotal steps: {ends[-1]}")
    print(f"Total episodes: {len(ends)}")
    print(f"Human demos read: {n_human_read}")
    print(f"MimicGen demos read: {n_mg_read}")


if __name__ == "__main__":
    HUMAN_HDF5 = (
        "/workspace/oat_casa/robocasa/datasets/v0.1/single_stage/"
        "kitchen_sink/TurnOffSinkFaucet/human/demo_gentex_im128_randcams.hdf5"
    )
    MG_HDF5 = (
        "/workspace/oat_casa/robocasa/datasets/v0.1/single_stage/"
        "kitchen_sink/TurnOffSinkFaucet/mg/demo_gentex_im128_randcams.hdf5"
    )
    OUTPUT_ZARR = "data/robocasa/TurnOffSinkFaucet_N200.zarr"

    convert_robocasa_to_zarr(
        human_hdf5_path=HUMAN_HDF5,
        mg_hdf5_path=MG_HDF5,
        output_zarr_path=OUTPUT_ZARR,
        n_human=50,
        n_mg=150,
    )
