"""Convert a single RoboCasa *door* task (human + MimicGen HDF5) into a per-task
``<Task>_N200.zarr``, ready to be interleaved into the multi-task ``doors4``
dataset by ``scripts/compose_robocasa_multitask_dataset.py``.

This is a thin, task-parameterised driver around the reusable
``convert_robocasa_to_zarr`` function in ``scripts/convert_robocasa_to_zarr.py``
(kept as a *separate* file so the base converter is not edited). It differs from
that script's hard-coded ``__main__`` only in that it:

  * takes the task name + HDF5 paths on the command line, and
  * picks the human/MimicGen split *adaptively* so the output always has exactly
    ``--n_total`` (default 200) demos even if a task ships fewer than 50 human
    demos: ``n_human = min(50, available)`` and ``n_mg = n_total - n_human``.

Example:
    python scripts/convert_robocasa_doors.py \
        --task OpenSingleDoor \
        --human datasets/.../OpenSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5 \
        --mg    datasets/.../OpenSingleDoor/mg/.../demo_gentex_im128_randcams.hdf5 \
        --out   data/robocasa/OpenSingleDoor_N200.zarr
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse

import h5py

from convert_robocasa_to_zarr import convert_robocasa_to_zarr


def count_demos(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        return len([k for k in f["data"].keys() if k.startswith("demo_")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--human", required=True, help="human_im HDF5 path")
    parser.add_argument("--mg", required=True, help="mg_im HDF5 path")
    parser.add_argument("--out", required=True, help="output zarr path")
    parser.add_argument("--n_total", type=int, default=200)
    parser.add_argument("--n_human", type=int, default=50)
    args = parser.parse_args()

    human_avail = count_demos(args.human)
    mg_avail = count_demos(args.mg)
    n_human = min(args.n_human, human_avail)
    n_mg = args.n_total - n_human
    print(
        f"[{args.task}] human_avail={human_avail} mg_avail={mg_avail} "
        f"-> using {n_human} human + {n_mg} mg = {n_human + n_mg}"
    )
    if n_mg > mg_avail:
        raise RuntimeError(
            f"[{args.task}] need {n_mg} mg demos but only {mg_avail} available"
        )

    convert_robocasa_to_zarr(
        human_hdf5_path=args.human,
        mg_hdf5_path=args.mg,
        output_zarr_path=args.out,
        n_human=n_human,
        n_mg=n_mg,
    )


if __name__ == "__main__":
    main()
