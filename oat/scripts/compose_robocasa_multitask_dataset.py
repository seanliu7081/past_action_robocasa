"""Compose a multi-task RoboCasa zarr from per-task RoboCasa zarrs.

This is the RoboCasa analogue of ``compose_libero_multitask_dataset.py``.

Unlike LIBERO — whose per-task zarrs already embed ``task_uid`` (added during
HDF5->zarr conversion) so they can be merged with the generic ``merge_data.py``
— the RoboCasa per-task zarrs contain *no* ``task_uid``. This script therefore
does the interleave-merge AND injects a ``task_uid`` column per episode, using
the canonical mapping in ``oat.env.robocasa.factory``.

The shuffle/interleave logic matches ``scripts/merge_data.py`` (episodes are
drawn round-robin from a random buffer when ``--shuffle`` is set) so episode
ordering statistics match the LIBERO multi-task dataset.

Example:
    python scripts/compose_robocasa_multitask_dataset.py -mt robocasa3 --shuffle

Produces (for 3 tasks x 200 demos):
    data/robocasa/robocasa3_N600.zarr
with the same data keys as the per-task zarrs plus ``data/task_uid``.
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import re
import glob

import click
import numpy as np
import zarr

from oat.common.replay_buffer import ReplayBuffer
from oat.env.robocasa.factory import MT_TASKS, get_task_uid


# Match the on-disk chunking used by scripts/convert_robocasa_to_zarr.py so the
# composed dataset has the same layout as the per-task zarrs.
def _build_chunks(n_steps: int) -> dict:
    return {
        "action": (n_steps, 12),
        "robot0_eef_pos": (n_steps, 3),
        "robot0_eef_quat": (n_steps, 4),
        "robot0_gripper_qpos": (n_steps, 2),
        "robot0_base_pos": (n_steps, 3),
        "robot0_base_quat": (n_steps, 4),
        "robot0_agentview_left_image": (1, 128, 128, 3),
        "robot0_agentview_right_image": (1, 128, 128, 3),
        "robot0_eye_in_hand_image": (1, 128, 128, 3),
        "task_uid": (n_steps, 1),
    }


def _resolve_zarr_paths(multitask_name: str, root_dir: str):
    """Return [(task_name, zarr_path, n_demo), ...] for a multi-task suite."""
    resolved = []
    for task_name in MT_TASKS[multitask_name]:
        pattern = os.path.join(root_dir, f"{task_name}_N*.zarr")
        matching = sorted(glob.glob(pattern))
        if not matching:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        zarr_file = matching[0]
        match = re.search(rf"{task_name}_N(\d+)\.zarr", os.path.basename(zarr_file))
        if not match:
            raise ValueError(f"Could not extract demo count from filename: {zarr_file}")
        resolved.append((task_name, zarr_file, int(match.group(1))))
    return resolved


@click.command()
@click.option("-mt", "--multitask_name", type=click.Choice(list(MT_TASKS.keys())), required=True)
@click.option("--root_dir", type=str, default="data/robocasa")
@click.option("-r", "--shuffle", is_flag=True, help="Interleave episodes across tasks (recommended).")
@click.option("--seed", type=int, default=42, help="Seed for the shuffle interleave order.")
@click.option("-f", "--overwrite", is_flag=True, help="Overwrite the output zarr if it exists.")
def compose_robocasa_multitask_dataset(
    multitask_name: str,
    root_dir: str,
    shuffle: bool,
    seed: int,
    overwrite: bool,
):
    resolved = _resolve_zarr_paths(multitask_name, root_dir)
    num_demo = sum(n for _, _, n in resolved)
    save_path = os.path.join(root_dir, f"{multitask_name}_N{num_demo}.zarr")

    if os.path.exists(save_path):
        if not overwrite:
            raise FileExistsError(
                f"{save_path} already exists. Re-run with --overwrite/-f to replace it."
            )
        print(f"Overwriting existing {save_path}")
        os.system(f"rm -rf {save_path}")

    # Open per-task buffers (lazy, on-disk) and record their task_uid.
    buffers, task_uids, buffer_lens, buffer_idx = [], [], [], []
    for task_name, zarr_path, n_demo in resolved:
        buf = ReplayBuffer.create_from_path(zarr_path, mode="r")
        uid = get_task_uid(task_name)
        print("-" * 60)
        print(f"{task_name} (uid={uid}, demos={buf.n_episodes}) <- {zarr_path}")
        buffers.append(buf)
        task_uids.append(uid)
        buffer_lens.append(buf.n_episodes)
        buffer_idx.append(0)

    total_eps = sum(buffer_lens)
    rng = np.random.default_rng(seed)
    merged = ReplayBuffer.create_empty_zarr()

    # Interleave episodes (mirrors scripts/merge_data.py). When a buffer is
    # exhausted it is removed from the pool, so each task contributes exactly
    # its episodes.
    for _ in range(total_eps):
        idx = rng.integers(0, len(buffers)) if shuffle else 0
        ep = buffers[idx].get_episode(buffer_idx[idx], copy=True)

        # Inject the per-step task_uid column (int64, shape [T, 1]) — same
        # dtype/shape as LIBERO's task_uid.
        ep_len = len(ep["action"])
        ep["task_uid"] = np.full((ep_len, 1), task_uids[idx], dtype=np.int64)
        merged.add_episode(ep)

        buffer_idx[idx] += 1
        if buffer_idx[idx] >= buffer_lens[idx]:
            buffers.pop(idx)
            task_uids.pop(idx)
            buffer_lens.pop(idx)
            buffer_idx.pop(idx)

    assert len(buffers) == 0, "Some buffers were not fully merged."

    print("-" * 60)
    print(f"Saving merged multi-task dataset -> {save_path}")
    print(f"  episodes={merged.n_episodes}, steps={merged.n_steps}")
    merged.save_to_path(save_path, chunks=_build_chunks(merged.n_steps), compressors="disk")

    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    root = zarr.open(save_path, "r")
    print(root.tree())
    uids = root["data"]["task_uid"][:, 0]
    vals, counts = np.unique(uids, return_counts=True)
    print(f"\ntask_uid values -> step counts: {dict(zip(vals.tolist(), counts.tolist()))}")
    ends = root["meta"]["episode_ends"][:]
    print(f"episodes={len(ends)}, total_steps={ends[-1]}")


if __name__ == "__main__":
    compose_robocasa_multitask_dataset()
