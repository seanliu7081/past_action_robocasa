# Training OAT (SO(3)-aug tokenizer + Enriched-Past policy) on RoboCasa

This guide covers the two configs adapted into RoboCasa:

- **Phase 1 tokenizer** — `train_oattok_so3aug.yaml`: OATTok with SO(3) raw-action
  augmentation (`OATTokSO3Aug`) + FSQ quantizer.
- **Phase 2 policy** — `train_oatpolicy_with_enriched_past_robocasa.yaml`:
  `OATPolicyWithEnrichedPast` (7-step action history + acceleration/jerk features).

Both configs **default to the `CloseDrawer` task** so they run with just
`--config-name` (no task override). Override `task/tokenizer` / `task/policy` to run
other RoboCasa tasks or LIBERO.

> Environment setup, dataset download, and HDF5→zarr conversion are unchanged —
> follow steps 0–2 of [`robocasa_training.md`](robocasa_training.md) first. You need
> `data/robocasa/<Task>_N<num_demo>.zarr` to exist.

> **Important — make sure `import oat` resolves to this repo.** `oat` is a namespace
> package. If `import oat` picks up a *different* checkout (e.g. a LIBERO
> `/workspace/oat`), Phase 2 fails with
> `Error locating target 'oat.env_runner.robocasa_runner.RoboCasaRunner'` (that repo
> has no robocasa runner). The usual cause is being in the wrong conda env — use the
> `oat_casa` env, where this repo was `pip install -e`'d. Verify with:
> ```bash
> python -c "import oat; print(oat.__path__)"   # should point at /workspace/oat_casa/oat
> ```
> If you must run from an env that has a different `oat` installed, either
> `pip install -e /workspace/oat_casa/oat` (rewrites the editable pointer to this repo)
> or prepend `/workspace/oat_casa/oat` to `PYTHONPATH` per-command.

## New files added (nothing existing was modified)

| File | Purpose |
|---|---|
| `oat/tokenizer/oat/augment/so3_action_chunk_aug.py` | `SO3ActionChunkAug` — per-chunk SO(3) action augmentation (ported) |
| `oat/tokenizer/oat/tokenizer_so3_aug.py` | `OATTokSO3Aug` — OATTok variant that applies the aug in `forward` (ported) |
| `oat/config/train_oattok_so3aug.yaml` | RoboCasa-default SO(3)-aug tokenizer training config |
| `oat/config/train_oatpolicy_with_enriched_past_robocasa.yaml` | RoboCasa-default enriched-past policy training config |

## Why the SO(3) aug is correct for RoboCasa

The RoboCasa `PandaOmron` action is **12-dim**:

| Index | Meaning |
|---|---|
| `[0:3]` | arm delta position |
| `[3:6]` | arm delta orientation — **axis-angle (rotation vector)** |
| `[6]` | gripper |
| `[7:12]` | mobile base (forward/side/yaw velocities) + torso |

The aug is configured with `rot_start=3, rot_end=6, augment_position=false`, so it
perturbs **only** the `[3:6]` end-effector orientation and leaves position, gripper,
and base untouched — exactly the convention RoboCasa records (verified: the env runner
passes actions straight to `env.step` with no reordering). The default
`max_angle_deg=30`, `mode=left_noise`, `p=1` matches the source LIBERO config.

## Phase 1 — Train the SO(3)-aug tokenizer

The tokenizer trains on action sequences only (no observations), so it does not need
the local `robosuite`/`robocasa` or `MUJOCO_GL`. It still needs `oat` to resolve to this
repo — the `PYTHONPATH` prefix below is the per-command way (drop it if you used
`pip install -e /workspace/oat_casa/oat`).

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

- Other task: append `task/tokenizer=robocasa/TurnOffMicrowave` (or `CoffeePressButton`,
  `TurnOffSinkFaucet`).
- Checkpoints land in
  `output/<date>/<time>_train_oattok_so3aug_CloseDrawer_N200/checkpoints/`,
  e.g. `ep-0650_mse-0.002.ckpt`.

## Phase 2 — Train the Enriched-Past policy

Uses the Phase-1 tokenizer checkpoint. `OATTok.from_checkpoint` rebuilds the tokenizer
from the config saved inside the checkpoint, so it loads the `OATTokSO3Aug` checkpoint
transparently — no config change needed (the augmentation only runs during tokenizer
training; the policy uses the frozen tokenizer's `tokenize`/`detokenize` only).

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
    policy.action_tokenizer.checkpoint=<path_to_so3aug_tokenizer_checkpoint>
```

(Drop the leading `/workspace/oat_casa/oat:` from `PYTHONPATH` if you used
`pip install -e /workspace/oat_casa/oat`. The local `robosuite`/`robocasa` entries are
still required so `robosuite` resolves to the PandaOmron-capable local version with
`load_composite_controller_config`.)

Replace `<path_to_so3aug_tokenizer_checkpoint>` with the Phase-1 path, e.g.:
```
output/20260602/093210_train_oattok_so3aug_CloseDrawer_N200/checkpoints/ep-0650_mse-0.002.ckpt
```

- Other task: append `task/policy=robocasa/TurnOffMicrowave_with_past` — keep the
  tokenizer task in Phase 1 matched to the same task.
- Rollout evaluation runs every `training.rollout_every` (100) epochs via
  `RoboCasaRunner`. Set `task.policy.lazy_eval=true` to skip rollouts and track only
  validation loss.

## Notes

- `MUJOCO_GL=egl` and the `PYTHONPATH` to local `robosuite`/`robocasa` are required for
  Phase 2 rollouts (offscreen rendering + PandaOmron / Kitchen envs), same as the base
  RoboCasa workflow.
- RoboCasa uses 3 cameras; if you hit OOM in Phase 2, reduce batch sizes:
  `dataloader.batch_size=16 val_dataloader.batch_size=16`.
- To run the same configs on LIBERO instead, override the task, e.g.
  `task/tokenizer=libero/libero10` / `task/policy=libero/libero10_with_past`.
