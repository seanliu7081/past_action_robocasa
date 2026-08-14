# Training the Self-Past policy on RoboCasa

`OATPolicyWithSelfPast` is `OATPolicyWithEnrichedPast` with **one** change: during
training the 7 past actions are the policy's **own generated actions** instead of the
dataset's ground truth. Architecture, condition layout, loss and the inference path are
inherited unchanged.

Read [`robocasa_so3aug_enriched_past_training.md`](robocasa_so3aug_enriched_past_training.md)
first — Phase 1 (tokenizer) is identical and everything there about environment,
`PYTHONPATH` and `import oat` resolution applies here too. This guide only covers the
Phase-2 replacement.

## Why

At rollout, `past_action` for chunk *k* is literally what the policy produced on chunk
*k-1* (`oat_policy_with_enriched_past.py:337-343`), so training on ground-truth past is
an exposure-bias gap. Each training step therefore re-runs the policy on the observation
window one execution stride back and applies the same buffer-update rule:

```
prev_pred = detokenize(generate(cond(prev_obs, prev_past_action)))
past      = prev_pred[:, n_exec - past_n : n_exec]          # n_exec >= past_n
```

That is **one level of unrolling** — the inner call still uses ground-truth past, so the
gap is closed at depth 1, not to convergence. The extra window (`prev_obs`,
`prev_past_action`) comes from `ZarrDatasetWithPrevWindow`, which is why the self-past
configs need a `*_with_prev_window` task group.

## New files added (nothing existing was modified)

| File | Purpose |
|---|---|
| `oat/policy/oat_policy_with_self_past.py` | `OATPolicyWithSelfPast` — subclasses `OATPolicyWithEnrichedPast`, overrides only where `past_action` comes from |
| `oat/dataset/zarr_dataset_with_prev_window.py` | `ZarrDatasetWithPrevWindow` — parent's output plus `prev_obs` / `prev_past_action` |
| `oat/config/train_oatpolicy_with_self_past_robocasa.yaml` | RoboCasa-default self-past policy config (defaults to `CloseDrawer_with_prev_window`) |
| `oat/config/train_oatpolicy_with_self_past.yaml` | LIBERO-default self-past policy config |
| `oat/config/task/policy/robocasa/CloseDrawer_with_prev_window.yaml` | `CloseDrawer_with_past` + the previous window |
| `oat/config/task/policy/robocasa/doors4_with_prev_window.yaml` | `doors4_with_past` + the previous window |
| `oat/config/task/policy/libero/libero10_with_prev_window.yaml` | `libero10_with_past` + the previous window |

To cover any other RoboCasa task, copy its `*_with_past.yaml`, change the dataset
`_target_` to `oat.dataset.zarr_dataset_with_prev_window.ZarrDatasetWithPrevWindow` and
add `n_exec_steps: ${n_action_steps}`. Nothing else changes.

## Phase 2 — Train the Self-Past policy

Same Phase-1 tokenizer checkpoint as the enriched-past guide.

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
    --config-name=train_oatpolicy_with_self_past_robocasa \
    task/policy=robocasa/CloseDrawer_with_prev_window \
    training.num_demo=200 \
    task.policy.lazy_eval=false \
    policy.action_tokenizer.checkpoint=<path_to_so3aug_tokenizer_checkpoint>
```

Multi-task doors4 (4 tasks × 200 demos → `data/robocasa/doors4_N800.zarr`):

```bash
MUJOCO_GL=egl \
PYTHONPATH=/workspace/oat_casa/oat:/workspace/oat_casa/robosuite:/workspace/oat_casa/robocasa:$PYTHONPATH \
HYDRA_FULL_ERROR=1 accelerate launch \
    --num_machines 1 --multi_gpu --num_processes [num_gpu] \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy_with_self_past_robocasa \
    task/policy=robocasa/doors4_with_prev_window \
    training.num_demo=800 \
    policy.action_tokenizer.checkpoint=output/20260702/100505_train_oattok_so3aug_doors4_N800/checkpoints/ep-0420_mse-0.002.ckpt
```

> The task group **must** be a `*_with_prev_window` one. Point this config at a
> `*_with_past` group and the policy raises
> `KeyError: OATPolicyWithSelfPast needs 'prev_obs' / 'prev_past_action'` as soon as
> warmup ends.

## Knobs

| Config key | Default | Meaning |
|---|---|---|
| `policy.self_past_p` | `1.0` | Per-sample probability of using the generated past. `1.0` = always (the intended experiment); lower mixes with ground truth, scheduled-sampling style |
| `policy.self_past_warmup_steps` | `500` | Train steps of ground-truth past first, so the policy is not conditioned on a randomly-initialised model's output. Those steps also run at baseline speed |
| `policy.self_past_temperature` | `null` | Sampling temperature of the inner generation; `null` reuses `policy.temperature` — what rollout actually uses |
| `policy.self_past_topk` | `null` | Same, for `policy.topk` |

## Notes

- **Cost.** Each step after warmup adds one obs-encoder forward over `prev_obs` plus an
  autoregressive generation of `latent_horizon` tokens — measured at ~1.47× step time
  (batch 64, bf16, single RTX 4090) on LIBERO. RoboCasa has 3 cameras and a 12-dim
  action, so expect the same ratio on a larger base cost; drop batch size
  (`dataloader.batch_size=16 val_dataloader.batch_size=16`) if you hit OOM.
- **Dataset cost.** `pad_before` grows by one execution stride (16 vs 8 with the default
  `To=2, past_n=7, n_exec=8`), i.e. one extra observation window per sample. Boundary
  padding is unchanged — still the `SequenceSampler`'s edge repetition — so the emitted
  `obs` / `action` / `past_action` are frame-for-frame identical to the `*_with_past`
  dataset.
- **Validation** also runs `forward()`, so the generation cost applies to val steps too.
- **Checkpoint compatibility.** Runs are named `train_oatpolicy_with_self_past`. The
  checkpoint's `cfg.policy._target_` is
  `oat.policy.oat_policy_with_self_past.OATPolicyWithSelfPast`, which resolves in this
  repo, so `scripts/eval_policy_sim.py` loads it exactly like an enriched-past
  checkpoint: inference is inherited unchanged (rolling `_past_buffer`, no dataset
  involved), and the baked-in `env_runner` block of a `*_with_prev_window` task group is
  identical to its `*_with_past` sibling.
- **LIBERO instead:** `--config-name=train_oatpolicy_with_self_past` (already defaults to
  `libero/libero10_with_prev_window`).
