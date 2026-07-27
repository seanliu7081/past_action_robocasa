# Training OAT on RoboCasa Tasks

## 0. Environment Setup

### Create the conda environment

```bash
# Using miniforge3
conda create -n oat_casa python=3.10 -y
conda activate oat_casa
```

### Install conda packages

```bash
conda install -c conda-forge \
    einops hydra-core wandb dill zarr numba \
    diffusers accelerate gsutil transformers av gymnasium -y
```

### Install pip packages

```bash
# PyTorch (match your CUDA driver version — check with nvidia-smi)
# For CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# OAT dependencies
pip install vector-quantize-pytorch easydict bddl cloudpickle lxml

# Robomimic and robosuite (pip versions for base dependencies)
pip install robomimic==0.2.0 robosuite==1.4.0
```

### Install OAT

```bash
cd /workspace/oat_casa/oat
pip install -e .
```

### Local robosuite and robocasa

The local versions at `/workspace/oat_casa/robosuite` and `/workspace/oat_casa/robocasa` contain
features not in the pip-installed packages (PandaOmron robot, composite controllers, Kitchen envs).
They are loaded via `PYTHONPATH` at runtime — no separate install needed.

Remove the strict version assertions in `robocasa/__init__.py` if they block import
(numpy, mujoco, robosuite version checks).

### Download RoboCasa kitchen assets

Required for environment rollouts (~5GB):

```bash
PYTHONPATH=/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH \
python -m robocasa.scripts.setup_macros

PYTHONPATH=/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH \
python -m robocasa.scripts.download_kitchen_assets
```

### Verify installation

```bash
PYTHONPATH=/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH \
python -c "from oat.env.robocasa.env import RoboCasaEnv; print('OK')"
```

## 1. Download RoboCasa Datasets

Download the HDF5 demo files for your desired task. For example, `CloseDrawer`:

```bash
cd /workspace/oat_casa/robocasa

PYTHONPATH=/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH python -m robocasa.scripts.download_datasets --tasks TurnOffMicrowave --ds_types human_im

PYTHONPATH=/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH python -m robocasa.scripts.download_datasets --tasks TurnOffMicrowave --ds_types mg_im
```

The datasets will be saved to:
```
robocasa/datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/
├── human/demo_gentex_im128_randcams.hdf5    (54 human demos)
└── mg/demo_gentex_im128_randcams.hdf5       (3000 MimicGen demos)
```

## 2. Convert HDF5 to Zarr

Convert the HDF5 datasets into a single zarr file compatible with OAT's `ZarrDataset`:

```bash
cd /workspace/oat_casa/oat

python scripts/convert_robocasa_to_zarr.py
```

By default this reads 50 human demos + 150 MimicGen demos and outputs:
```
data/robocasa/CloseDrawer_N200.zarr
```

To customize, edit the paths and counts at the bottom of `scripts/convert_robocasa_to_zarr.py`.

## 3. Train the Tokenizer (Phase 1)

The OATTok tokenizer trains on action sequences only (no observations).

```bash
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

Adjust `--num_processes` to match your GPU count.

Checkpoints are saved to `output/<date>/<time>_train_oattok_CloseDrawer_N200/checkpoints/`.

## 4. Train the Policy (Phase 2)

Two policy variants are available. Both use the same tokenizer checkpoint from Phase 1.

### Option A: OATPolicy (standard)

```bash
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

### Option B: OATPolicy with Enriched Past

Conditions on past actions (7-step history) plus explicit acceleration/jerk features.

```bash
cd /workspace/oat_casa/oat

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

Replace `<path_to_tokenizer_checkpoint>` with the path from Phase 1, e.g.:
```
output/20260324/112742_train_oattok_CloseDrawer_N200/checkpoints/ep-0650_mse-0.002.ckpt
```

### GPU memory

RoboCasa uses 3 cameras (vs LIBERO's 2), which increases GPU memory usage. If you encounter OOM errors, reduce batch sizes:

```bash
... dataloader.batch_size=32 val_dataloader.batch_size=32
```

## Config Files

| Config | Path |
|---|---|
| Tokenizer task | `oat/config/task/tokenizer/robocasa/CloseDrawer.yaml` |
| Policy task (standard) | `oat/config/task/policy/robocasa/CloseDrawer.yaml` |
| Policy task (with past) | `oat/config/task/policy/robocasa/CloseDrawer_with_past.yaml` |

## Notes

- `MUJOCO_GL=egl` is required for offscreen rendering during rollout evaluation.
- `PYTHONPATH` must include the local `robosuite` and `robocasa` directories (they contain features not in the pip-installed versions).
- The commands above pass `task.policy.lazy_eval=false training.rollout_every=100` so rollout evaluation runs every 100 epochs regardless of the task config's default (multi-task configs like `doors4` default to `lazy_eval: true`; the config-default `rollout_every` is 50 for standard, 200 for with_past). Drop both overrides — or set `task.policy.lazy_eval=true` — to skip rollouts and only track validation loss.
- Logs are sent to Weights & Biases under the `oat_dev` project.
- Both policy variants share the same tokenizer, zarr data, env, and runner. The only difference is the dataset class (`ZarrDatasetWithPastAction` adds past action history) and the policy class (`OATPolicyWithEnrichedPast` conditions on past actions).
