"""Convert RoboCasa **v1.0** LeRobot datasets into an OAT zarr using **v0.2**
conventions.

Why this exists
---------------
``convert_robocasa_to_zarr.py`` reads robomimic HDF5 (RoboCasa v0.2 releases).
Some tasks -- notably ``CupcakeCleanup`` and ``PastryDisplay`` -- were never
released for v0.2; they only exist in the RoboCasa **v1.0** drop, which ships
**LeRobot v2.1** datasets (parquet + mp4 + json) and uses *different* action and
state conventions. This script bridges that gap so the resulting zarr is
byte-compatible with everything downstream (ZarrDataset, the multi-task
composer, the tokenizer, and the policy configs).

Three non-obvious transforms are applied. Each was verified empirically against
the live v0.2 ``RoboCasaEnv``; see the notes below.

1. ACTION PERMUTATION.  The 12-dim PandaOmron action is ordered differently.

       v1.0 (LeRobot modality.json):
           [0:4] base_motion   [4] control_mode  [5:8] eef_pos
           [8:11] eef_rot      [11] gripper
       v0.2 (robosuite HybridMobileBase ``_action_split_indexes``):
           [0:3] eef_pos  [3:6] eef_rot  [6] gripper
           [7:10] base    [10] torso     [11] base_mode

   Verified: the live env reports ``right(0,6), right_gripper(6,7),
   base(7,10), torso(10,11)`` and ``HybridMobileBase.set_goal`` reads
   ``all_action[-1]`` as ``base_mode`` (default -1) -- which matches v1.0's
   ``control_mode``, observed constant -1.0 in the data.

   Getting this wrong is silent: the arrays still have shape [T, 12], training
   still converges, and the policy is simply wrong. It would also mis-target the
   SO(3) augmentation, whose ``rot_start=3, rot_end=6`` assumes v0.2 ordering.

2. EEF FRAME.  v1.0 stores the end-effector pose **relative to the robot base**;
   the v0.2 env's ``robot0_eef_pos``/``robot0_eef_quat`` observations are in
   **world** frame. Training on relative poses while evaluating against world
   poses would break rollouts. We compose base * relative -> world.

   Verified: for CupcakeCleanup episode 0 the composed world pose gives
   ``|eef - base| = 0.642`` and z in ``[0.825, 1.295]``; the live v0.2 env at
   reset reports ``|eef - base| = 0.64`` and eef z ``1.291``.

3. IMAGES.  v1.0 videos are 256x256; OAT configs expect 128x128. Frames are
   decoded with OpenCV (which yields **BGR**) and converted to RGB to match the
   robomimic HDF5 convention, then area-resized.

Example
-------
    python scripts/convert_robocasa_lerobot_to_zarr.py \
        --lerobot-dir /path/to/CupcakeCleanup/lerobot \
        --out data/robocasa/CupcakeCleanup_N101.zarr
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import glob
import json
import os
import shutil

import click
import cv2
import numpy as np
import pandas as pd
import zarr
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from oat.common.replay_buffer import ReplayBuffer


# v1.0 action index -> v0.2 action index. See docstring (1).
#   v0.2[0:3]=eef_pos   <- v1.0[5:8]
#   v0.2[3:6]=eef_rot   <- v1.0[8:11]
#   v0.2[6]  =gripper   <- v1.0[11]
#   v0.2[7:10]=base     <- v1.0[0:3]
#   v0.2[10] =torso     <- v1.0[3]
#   v0.2[11] =base_mode <- v1.0[4]
ACTION_PERM_V10_TO_V02 = np.array([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4])

# observation.state (16-dim) slices, from meta/modality.json.
STATE_SLICES = {
    "base_pos": (0, 3),
    "base_quat": (3, 7),
    "eef_pos_rel": (7, 10),
    "eef_quat_rel": (10, 14),
    "gripper_qpos": (14, 16),
}

CAMERAS = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


def decode_video(path: str, n_expected: int, image_size: int) -> np.ndarray:
    """Decode an mp4 to [T, image_size, image_size, 3] uint8 RGB."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # OpenCV decodes BGR; robomimic HDF5 images are RGB.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != image_size or frame.shape[1] != image_size:
            frame = cv2.resize(
                frame, (image_size, image_size), interpolation=cv2.INTER_AREA
            )
        frames.append(frame)
    cap.release()

    if len(frames) != n_expected:
        raise ValueError(
            f"{path}: decoded {len(frames)} frames but parquet has {n_expected} rows"
        )
    return np.asarray(frames, dtype=np.uint8)


def relative_eef_to_world(state: np.ndarray):
    """Compose base pose with base-relative eef pose -> world-frame eef pose.

    Quaternions are (x, y, z, w) in both robosuite and scipy, so no reordering
    is needed.
    """
    lo, hi = STATE_SLICES["base_pos"]
    base_pos = state[:, lo:hi]
    lo, hi = STATE_SLICES["base_quat"]
    base_quat = state[:, lo:hi]
    lo, hi = STATE_SLICES["eef_pos_rel"]
    eef_pos_rel = state[:, lo:hi]
    lo, hi = STATE_SLICES["eef_quat_rel"]
    eef_quat_rel = state[:, lo:hi]

    rot_base = Rotation.from_quat(base_quat)
    eef_pos_world = base_pos + rot_base.apply(eef_pos_rel)
    eef_quat_world = (rot_base * Rotation.from_quat(eef_quat_rel)).as_quat()
    return eef_pos_world, eef_quat_world


def read_episode(lerobot_dir: str, ep_idx: int, image_size: int) -> dict:
    parquet = os.path.join(
        lerobot_dir, "data", "chunk-000", f"episode_{ep_idx:06d}.parquet"
    )
    df = pd.read_parquet(parquet)
    n = len(df)

    action_v10 = np.stack(df["action"].values).astype(np.float64)
    state = np.stack(df["observation.state"].values).astype(np.float64)

    if action_v10.shape[1] != 12:
        raise ValueError(f"expected 12-dim action, got {action_v10.shape[1]}")
    if state.shape[1] != 16:
        raise ValueError(f"expected 16-dim state, got {state.shape[1]}")

    action = action_v10[:, ACTION_PERM_V10_TO_V02]
    eef_pos, eef_quat = relative_eef_to_world(state)

    lo, hi = STATE_SLICES["base_pos"]
    base_pos = state[:, lo:hi]
    lo, hi = STATE_SLICES["base_quat"]
    base_quat = state[:, lo:hi]
    lo, hi = STATE_SLICES["gripper_qpos"]
    gripper_qpos = state[:, lo:hi]

    ep = {
        "action": action.astype(np.float32),
        "robot0_eef_pos": eef_pos.astype(np.float32),
        "robot0_eef_quat": eef_quat.astype(np.float32),
        "robot0_gripper_qpos": gripper_qpos.astype(np.float32),
        "robot0_base_pos": base_pos.astype(np.float32),
        "robot0_base_quat": base_quat.astype(np.float32),
    }

    for cam in CAMERAS:
        vid = os.path.join(
            lerobot_dir,
            "videos",
            "chunk-000",
            f"observation.images.{cam}",
            f"episode_{ep_idx:06d}.mp4",
        )
        ep[f"{cam}_image"] = decode_video(vid, n, image_size)

    return ep


def _build_chunks(n_steps: int, image_size: int) -> dict:
    return {
        "action": (n_steps, 12),
        "robot0_eef_pos": (n_steps, 3),
        "robot0_eef_quat": (n_steps, 4),
        "robot0_gripper_qpos": (n_steps, 2),
        "robot0_base_pos": (n_steps, 3),
        "robot0_base_quat": (n_steps, 4),
        "robot0_agentview_left_image": (1, image_size, image_size, 3),
        "robot0_agentview_right_image": (1, image_size, image_size, 3),
        "robot0_eye_in_hand_image": (1, image_size, image_size, 3),
    }


@click.command()
@click.option("--lerobot-dir", required=True, help="Path to the extracted `lerobot/` dir.")
@click.option("--out", required=True, help="Output zarr path.")
@click.option("--max-demos", type=int, default=None, help="Cap episodes (default: all).")
@click.option("--image-size", type=int, default=128)
@click.option("-f", "--overwrite", is_flag=True)
def main(lerobot_dir, out, max_demos, image_size, overwrite):
    info = json.load(open(os.path.join(lerobot_dir, "meta", "info.json")))
    print(f"robot_type={info['robot_type']} fps={info['fps']} "
          f"episodes={info['total_episodes']} frames={info['total_frames']}")
    if info["robot_type"] != "PandaOmron":
        raise ValueError(f"expected PandaOmron, got {info['robot_type']}")

    n_eps = info["total_episodes"]
    available = len(glob.glob(os.path.join(lerobot_dir, "data", "chunk-000", "*.parquet")))
    if available != n_eps:
        print(f"WARNING: info.json says {n_eps} episodes but found {available} parquet files")
        n_eps = min(n_eps, available)
    if max_demos is not None:
        n_eps = min(n_eps, max_demos)

    if os.path.exists(out):
        if not overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite/-f")
        shutil.rmtree(out)

    replay_buffer = ReplayBuffer.create_empty_zarr()
    for i in tqdm(range(n_eps), desc="Converting"):
        replay_buffer.add_episode(read_episode(lerobot_dir, i, image_size))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    replay_buffer.save_to_path(
        out, chunks=_build_chunks(replay_buffer.n_steps, image_size), compressors="disk"
    )

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    root = zarr.open(out, "r")
    for key in sorted(root["data"].keys()):
        arr = root["data"][key]
        print(f"data/{key}: shape={arr.shape} dtype={arr.dtype}")
    ends = root["meta"]["episode_ends"][:]
    print(f"\nepisodes={len(ends)} total_steps={ends[-1]}")
    act = root["data"]["action"][:]
    print(f"action[6] (gripper) unique: {np.unique(act[:, 6])[:5]}")
    print(f"action[11] (base_mode) unique: {np.unique(act[:, 11])[:5]}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
