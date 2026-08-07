# OAT on RoboCasa — Implementation Guide

End-to-end guide for the RoboCasa pipeline: environment setup → data preparation →
Phase 1 tokenizer training → Phase 2 policy training.

This merges and supersedes `robocasa_training.md` and
`robocasa_so3aug_enriched_past_training.md`.

## Overview

Training is always **two phases**:

| Phase | What trains | Inputs | Config |
|---|---|---|---|
| 1 | Action tokenizer (actions only, no observations) | `data/robocasa/<Task>_N<num_demo>.zarr` | `train_oattok` / `train_oattok_so3aug` |
| 2 | Policy (images + states → action tokens) | same zarr + frozen Phase-1 checkpoint | `train_oatpolicy` / `train_oatpolicy_with_enriched_past{,_robocasa}` |

Variants available at each phase:

- **Tokenizer** — `OATTok` (base) or `OATTokSO3Aug` (adds SO(3) augmentation on the raw
  action chunk + FSQ quantizer).
- **Policy** — `OATPolicy` (standard) or `OATPolicyWithEnrichedPast` (conditions on a
  7-step action history plus explicit acceleration/jerk features).

The two axes are independent. `OATTok.from_checkpoint` rebuilds the tokenizer from the
config saved inside the checkpoint, so an SO(3)-aug tokenizer checkpoint loads
transparently into any policy config — the augmentation only runs during tokenizer
training; the policy uses the frozen tokenizer's `tokenize`/`detokenize` only.

---

## 1. Environment setup

### 1.1 Create the conda environment

```bash
# Using miniforge3
conda create -n oat_casa python=3.10 -y
conda activate oat_casa
```

### 1.2 Install conda packages

```bash
conda install -c conda-forge \
    einops hydra-core wandb dill zarr numba \
    diffusers accelerate gsutil transformers av gymnasium -y
```

### 1.3 Install pip packages

```bash
# PyTorch (match your CUDA driver version — check with nvidia-smi)
# For CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# OAT dependencies
pip install vector-quantize-pytorch easydict bddl cloudpickle lxml

# Robomimic and robosuite (pip versions for base dependencies)
pip install robomimic==0.2.0 robosuite==1.4.0
```

### 1.4 Install OAT

```bash
cd /workspace/oat_casa/oat
pip install -e .
```

### 1.5 Local robosuite and robocasa

The local checkouts at `/workspace/oat_casa/robosuite` and `/workspace/oat_casa/robocasa`
contain features missing from the pip packages (PandaOmron robot, composite controllers,
Kitchen envs). They are loaded via `PYTHONPATH` at runtime — no separate install needed.

Remove the strict version assertions in `robocasa/__init__.py` if they block import
(numpy, mujoco, robosuite version checks).

### 1.6 Download RoboCasa kitchen assets

Required for environment rollouts (~5 GB):

```bash
PYTHONPATH=/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH \
python -m robocasa.scripts.setup_macros

PYTHONPATH=/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH \
python -m robocasa.scripts.download_kitchen_assets
```

### 1.7 Verify the installation

```bash
PYTHONPATH=/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH \
python -c "from oat.env.robocasa.env import RoboCasaEnv; print('OK')"
```

### 1.8 Make sure `import oat` resolves to *this* repo

`oat` is a namespace package. If `import oat` picks up a *different* checkout (e.g. a
LIBERO `/workspace/oat`), Phase 2 fails with
`Error locating target 'oat.env_runner.robocasa_runner.RoboCasaRunner'` (that repo has no
RoboCasa runner). The usual cause is being in the wrong conda env — use `oat_casa`, where
this repo was `pip install -e`'d. Verify:

```bash
python -c "import oat; print(oat.__path__)"   # should point at /workspace/oat_casa/oat
```

If you must run from an env with a different `oat` installed, either
`pip install -e /workspace/oat_casa/oat` (rewrites the editable pointer to this repo) or
prepend `/workspace/oat_casa/oat` to `PYTHONPATH` per command.

---

## 2. Data preparation

### 2.1 Download the RoboCasa HDF5 datasets

Each task has two official releases: `human_im` (teleop) and `mg_im` (MimicGen
machine-generated).

```bash
cd /workspace/oat_casa/robocasa

PYTHONPATH=/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH \
python -m robocasa.scripts.download_datasets --tasks TurnOffMicrowave --ds_types human_im

PYTHONPATH=/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH \
python -m robocasa.scripts.download_datasets --tasks TurnOffMicrowave --ds_types mg_im
```

Files land under:

```
robocasa/datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/
├── human/demo_gentex_im128_randcams.hdf5    (~50–54 human demos)
└── mg/demo_gentex_im128_randcams.hdf5       (3000 MimicGen demos)
```

If `import robocasa` fails in your env (e.g. missing `lxml`), download directly with
`curl` using the box URLs in `robocasa/utils/dataset_registry.py`. MimicGen files are
large (30–49 GB per task) — process one task at a time and delete the HDF5 after
converting.

### 2.2 Convert HDF5 → zarr (single task)

```bash
cd /workspace/oat_casa/oat
python scripts/convert_robocasa_to_zarr.py
```

**Data combination: 50 human + 150 MimicGen = 200 demos per task.** The converter takes
the *first* N demos of each file (sorted by demo index, not random) and concatenates
`human_eps + mg_eps`, so episodes 0–49 are human and 50–199 are MimicGen. No provenance
column is written — after conversion the two sources are indistinguishable to the loader.
Output:

```
data/robocasa/CloseDrawer_N200.zarr
```

Each zarr holds `action` (12-dim), 3 camera streams
(`robot0_agentview_left/right_image`, `robot0_eye_in_hand_image`) and the state ports
(`robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, `robot0_base_pos`,
`robot0_base_quat`), plus `meta/episode_ends`.

To customise paths and counts, edit the `__main__` block at the bottom of
`scripts/convert_robocasa_to_zarr.py`.

For the door tasks use the parameterised driver, which keeps the total at exactly 200
even when a task ships fewer than 50 human demos (`n_human = min(50, available)`,
`n_mg = 200 - n_human`):

```bash
python scripts/convert_robocasa_doors.py \
    --task OpenSingleDoor \
    --human datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5 \
    --mg    datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/mg/2024-05-04-22-37-39/demo_gentex_im128_randcams.hdf5 \
    --out   data/robocasa/OpenSingleDoor_N200.zarr
```

### 2.3 Compose a multi-task dataset

Multi-task suites stack the per-task `N200` zarrs, interleaving episodes round-robin from
a seeded RNG and injecting a per-step `task_uid` column (the only multi-task conditioning
signal, consumed by the policy as a `type: state` observation):

```bash
python scripts/compose_robocasa_multitask_dataset.py -mt doors4 --shuffle
```

| Suite | Tasks | Output |
|---|---|---|
| `robocasa3` | CoffeePressButton, TurnOffMicrowave, TurnOffSinkFaucet | `data/robocasa/robocasa3_N600.zarr` |
| `sink3` | TurnOnSinkFaucet, TurnOffSinkFaucet, TurnSinkSpout | `data/robocasa/sink3_N600.zarr` |
| `doors4` | OpenSingleDoor, CloseSingleDoor, OpenDoubleDoor, CloseDoubleDoor | `data/robocasa/doors4_N800.zarr` |

`task_uid` values come from the append-only registry in `oat/env/robocasa/factory.py`:
CoffeePressButton 0, TurnOffMicrowave 1, TurnOffSinkFaucet 2, CloseDrawer 3 (reserved),
TurnOnSinkFaucet 4, TurnSinkSpout 5, OpenSingleDoor 6, CloseSingleDoor 7, OpenDoubleDoor
8, CloseDoubleDoor 9. Never reorder that list — existing checkpoints and datasets would
silently change meaning.

### 2.4 Train/val split

Both phases split by episode at load time with `val_ratio: 0.1` and the config `seed`, so
~10 % of episodes are held out and validation mixes human and MimicGen demos.

---

## 3. Phase 1 — Train the tokenizer

The tokenizer trains on action sequences only (no observations), so it needs neither the
local `robosuite`/`robocasa` nor `MUJOCO_GL`.

`training.num_demo` is **only a filename token** — it selects
`data/robocasa/<task>_N<num_demo>.zarr`, it does not subsample. Always pass the value
matching the zarr you built (200 single-task, 600 for `robocasa3`/`sink3`, 800 for
`doors4`).

### Option A — OATTok (base)

```bash
conda activate oat_casa
cd /workspace/oat_casa/oat

HYDRA_FULL_ERROR=1 accelerate launch \
    --num_machines 1 \
    --multi_gpu \
    --num_processes 2 \
    scripts/run_workspace.py \
    --config-name=train_oattok \
    task/tokenizer=robocasa/CloseDrawer \
    training.num_demo=200
```

Checkpoints: `output/<date>/<time>_train_oattok_CloseDrawer_N200/checkpoints/`.

### Option B — OATTokSO3Aug (SO(3) raw-action augmentation + FSQ)

```bash
conda activate oat_casa
cd /workspace/oat_casa/oat

PYTHONPATH=/workspace/oat_casa/oat:$PYTHONPATH \
HYDRA_FULL_ERROR=1 accelerate launch \
    --num_machines [num_node] \
    --multi_gpu \
    --num_processes [num_gpu] \
    scripts/run_workspace.py \
    --config-name=train_oattok_so3aug \
    task/tokenizer=robocasa/CloseDrawer \
    training.num_demo=200
```

Checkpoints: `output/<date>/<time>_train_oattok_so3aug_CloseDrawer_N200/checkpoints/`,
e.g. `ep-0650_mse-0.002.ckpt`.

Drop the `PYTHONPATH` prefix if you used `pip install -e /workspace/oat_casa/oat`
(see §1.8). Adjust `--num_processes` to your GPU count.

**Why the SO(3) aug is correct for RoboCasa.** The `PandaOmron` action is 12-dim:

| Index | Meaning |
|---|---|
| `[0:3]` | arm delta position |
| `[3:6]` | arm delta orientation — **axis-angle (rotation vector)** |
| `[6]` | gripper |
| `[7:12]` | mobile base (forward/side/yaw velocities) + torso |

The aug is configured with `rot_start=3, rot_end=6, augment_position=false`, so it
perturbs **only** the `[3:6]` EEF-orientation slice and leaves position, gripper, and base
untouched — exactly the convention RoboCasa records (verified: the env runner passes
actions straight to `env.step` with no reordering). Defaults `max_angle_deg=30`,
`mode=left_noise`, `p=1` match the source LIBERO config.

---

## 4. Phase 2 — Train the policy

All variants take the frozen Phase-1 checkpoint via
`policy.action_tokenizer.checkpoint=<path>`, e.g.

```
output/20260602/093210_train_oattok_so3aug_CloseDrawer_N200/checkpoints/ep-0650_mse-0.002.ckpt
```

Keep the Phase-1 tokenizer task matched to the Phase-2 policy task.

### Option A — OATPolicy (standard)

```bash
conda activate oat_casa
cd /workspace/oat_casa/oat

MUJOCO_GL=egl \
PYTHONPATH=/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH \
HYDRA_FULL_ERROR=1 accelerate launch \
    --num_machines 1 \
    --multi_gpu \
    --num_processes 2 \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy \
    task/policy=robocasa/CloseDrawer \
    training.num_demo=200 \
    task.policy.lazy_eval=false \
    training.rollout_every=100 \
    policy.action_tokenizer.checkpoint=<path_to_tokenizer_checkpoint>
```

### Option B — OATPolicy with Enriched Past

Conditions on a 7-step action history plus explicit acceleration/jerk features
(`past_n: 7`). The generic config:

```bash
MUJOCO_GL=egl \
PYTHONPATH=/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH \
HYDRA_FULL_ERROR=1 accelerate launch \
    --num_machines 1 \
    --multi_gpu \
    --num_processes 2 \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy_with_enriched_past \
    task/policy=robocasa/CloseDrawer_with_past \
    training.num_demo=200 \
    task.policy.lazy_eval=false \
    training.rollout_every=100 \
    policy.action_tokenizer.checkpoint=<path_to_tokenizer_checkpoint>
```

### Option C — Enriched Past, RoboCasa-default config

Same policy, but `train_oatpolicy_with_enriched_past_robocasa.yaml` already defaults to a
RoboCasa task (`robocasa/CloseDrawer_with_past`) so no task override is strictly needed:

```bash
conda activate oat_casa
cd /workspace/oat_casa/oat

MUJOCO_GL=egl \
PYTHONPATH=/workspace/oat_casa/oat:/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH \
HYDRA_FULL_ERROR=1 accelerate launch \
    --num_machines [num_node] \
    --multi_gpu \
    --num_processes [num_gpu] \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy_with_enriched_past_robocasa \
    task/policy=robocasa/CloseDrawer_with_past \
    training.num_demo=200 \
    task.policy.lazy_eval=false \
    policy.action_tokenizer.checkpoint=<path_to_so3aug_tokenizer_checkpoint>
```

Drop the leading `/workspace/oat_casa/oat:` if you used `pip install -e`. The local
`robosuite`/`robocasa` entries are still required so `robosuite` resolves to the
PandaOmron-capable local version with `load_composite_controller_config`.

### Multi-task example (doors4)

```bash
MUJOCO_GL=egl \
PYTHONPATH=/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH \
HYDRA_FULL_ERROR=1 accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 2 \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy_with_enriched_past_robocasa \
    task/policy=robocasa/doors4_with_past \
    training.num_demo=800 \
    policy.action_tokenizer.checkpoint=<path_to_tokenizer_checkpoint>
```

Multi-task configs default to `lazy_eval: true` (no rollouts during training) and route
evaluation through `RoboCasaMultiTaskRunner`, which uses one `max_episode_steps` for all
subtasks (1000 for `doors4`, since `OpenDoubleDoor`'s horizon is the longest).

### Switching to LIBERO

The same configs run on LIBERO by overriding the task, e.g.
`task/tokenizer=libero/libero10` / `task/policy=libero/libero10_with_past`.

---

## 5. Config reference

### Training configs

| Config | Default task | Notable defaults |
|---|---|---|
| `train_oattok` | `libero/libero10` | `num_demo: 200`, batch 256 |
| `train_oattok_so3aug` | `robocasa/robocasa3_with_past` | `num_demo: 200`, batch 256 |
| `train_oatpolicy` | `libero/libero10` | `rollout_every: 50`, `num_demo: 500`, batch 64 |
| `train_oatpolicy_with_enriched_past` | `libero/libero10_with_past` | `past_n: 7`, `rollout_every: 100`, `num_demo: 200`, batch 32 |
| `train_oatpolicy_with_enriched_past_robocasa` | `robocasa/CloseDrawer_with_past` | `past_n: 7`, `rollout_every: 100`, `num_demo: 600`, batch 32 |

Because the defaults differ from config to config, **always pass `task/...` and
`training.num_demo` explicitly** rather than relying on a config default.

### Task configs

| Kind | Path |
|---|---|
| Tokenizer task | `oat/config/task/tokenizer/robocasa/<Task>.yaml` |
| Policy task (standard) | `oat/config/task/policy/robocasa/<Task>.yaml` |
| Policy task (with past) | `oat/config/task/policy/robocasa/<Task>_with_past.yaml` |

Available single tasks: `CloseDrawer`, `CoffeePressButton`, `TurnOffMicrowave`,
`TurnOffSinkFaucet`, `TurnOnSinkFaucet`, `TurnSinkSpout`. Multi-task: `robocasa3`,
`sink3`, `doors4` (and their `_with_past` variants).

### New files added for the SO(3)-aug / enriched-past variants

Nothing existing was modified — variants live in their own files:

| File | Purpose |
|---|---|
| `oat/tokenizer/oat/augment/so3_action_chunk_aug.py` | `SO3ActionChunkAug` — per-chunk SO(3) action augmentation |
| `oat/tokenizer/oat/tokenizer_so3_aug.py` | `OATTokSO3Aug` — OATTok variant applying the aug in `forward` |
| `oat/config/train_oattok_so3aug.yaml` | SO(3)-aug tokenizer training config |
| `oat/config/train_oatpolicy_with_enriched_past_robocasa.yaml` | RoboCasa-default enriched-past policy config |

---

## 6. Notes and troubleshooting

- **`MUJOCO_GL=egl`** is required for offscreen rendering during rollout evaluation
  (Phase 2 only).
- **`PYTHONPATH`** must include the local `robosuite` and `robocasa` directories for
  Phase 2 — they carry PandaOmron, composite controllers, and the Kitchen envs.
- **Rollout evaluation.** The commands above pass `task.policy.lazy_eval=false` (and
  `training.rollout_every=100` where the config default differs) so rollouts run
  regardless of the task config's default. `lazy_eval` only works as a **CLI override** —
  putting it in a train-config yaml is clobbered because `_self_` precedes the task group
  in the defaults list. Drop the overrides, or set `task.policy.lazy_eval=true`, to skip
  rollouts and track validation loss only.
- **GPU memory.** RoboCasa uses 3 cameras (vs LIBERO's 2). On OOM, reduce batch sizes:
  `dataloader.batch_size=32 val_dataloader.batch_size=32` (16 for the enriched-past
  configs).
- **EGL framebuffer crash during rollouts.** `n_parallel_envs: 20` (the `doors4` default)
  can fail at startup with
  `mujoco.FatalError: Offscreen framebuffer is not complete, 0x8cdd` — 10 concurrent EGL
  contexts per GPU is too many on a 2×RTX 4090 box. Use `n_parallel_envs=16` (8/GPU), and
  keep it a multiple of the subtask count so each task keeps its env slot across chunks.
- **Disk.** The zarrs are large (`doors4_N800.zarr` ≈ 26 GB) and the source MimicGen HDF5s
  are 30–49 GB each. Convert one task at a time and delete the HDF5 afterwards.
- **Logging.** Runs are logged to Weights & Biases under the `oat_dev` project; local
  outputs go to `output/<date>/<time>_<name>_<task>_N<num_demo>/`.
- Both policy variants share the same tokenizer, zarr data, env, and runner. The only
  differences are the dataset class (`ZarrDatasetWithPastAction` adds past action history)
  and the policy class (`OATPolicyWithEnrichedPast` conditions on past actions).
