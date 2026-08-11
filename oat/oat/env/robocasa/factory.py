"""RoboCasa task registry for single-task and multi-task training.

This mirrors ``oat.env.libero.factory`` and the ``task_name_to_suite_and_ids``
mapping in ``oat.env.libero.env``: it provides

  * a canonical, ordered list of RoboCasa tasks,
  * a stable ``task_name -> task_uid`` mapping (used both when building the
    multi-task zarr and when injecting ``task_uid`` into observations at eval
    time), and
  * ``MT_TASKS`` describing which subtasks make up each multi-task suite.

The ``task_uid`` is the only multi-task conditioning signal consumed by the
policy (it enters the state encoder as a ``type: state`` observation of shape
``[1]``), exactly like LIBERO's ``task_uid``.  Keeping a single source of truth
here guarantees the integer a task is trained with matches the integer it is
evaluated with.

NOTE: only the tasks that currently have converted zarr data
(``CoffeePressButton``, ``TurnOffMicrowave``, ``TurnOffSinkFaucet``) are part of
the ``robocasa3`` suite.  ``CloseDrawer`` is kept in the canonical list (it has
configs but no data yet) so its uid is reserved and existing single-task uids do
not shift if its data is added later.
"""

from typing import List


# Canonical, ordered list of RoboCasa tasks. The index in this list IS the
# task_uid. Append-only: never reorder, or previously trained checkpoints /
# datasets would silently change the meaning of a uid.
ROBOCASA_TASK_NAMES: List[str] = [
    "CoffeePressButton",   # uid 0
    "TurnOffMicrowave",    # uid 1
    "TurnOffSinkFaucet",   # uid 2
    "CloseDrawer",         # uid 3 (reserved; no zarr data yet)
    "TurnOnSinkFaucet",    # uid 4
    "TurnSinkSpout",       # uid 5
    "OpenSingleDoor",      # uid 6
    "CloseSingleDoor",     # uid 7
    "OpenDoubleDoor",      # uid 8
    "CloseDoubleDoor",     # uid 9
    "CupcakeCleanup",      # uid 10 (RoboCasa v1.0 LeRobot data; see note below)
    "PastryDisplay",       # uid 11 (RoboCasa v1.0 LeRobot data; see note below)
]

ROBOCASA_TASK_TO_UID = {name: i for i, name in enumerate(ROBOCASA_TASK_NAMES)}
num_robocasa_tasks = len(ROBOCASA_TASK_NAMES)

# Human-readable language instruction per task. NOT written into the zarr (the
# policy conditions on task_uid, not text) — kept here for reference and for
# future language-conditioned variants.
ROBOCASA_TASK_PROMPTS = {
    "CoffeePressButton": "press the button to brew coffee",
    "TurnOffMicrowave": "turn off the microwave",
    "TurnOffSinkFaucet": "turn off the sink faucet",
    "CloseDrawer": "close the drawer",
    "TurnOnSinkFaucet": "turn on the sink faucet",
    "TurnSinkSpout": "turn the sink spout",
    "OpenSingleDoor": "open the single door",
    "CloseSingleDoor": "close the single door",
    "OpenDoubleDoor": "open the double door",
    "CloseDoubleDoor": "close the double door",
    # Verbatim from the v1.0 LeRobot `meta/tasks.jsonl`.
    "CupcakeCleanup": (
        "Move the fresh-baked cupcake off the tray onto the counter, and place "
        "the bowl used for mixing into the sink."
    ),
    "PastryDisplay": "Place the pastries on the plates.",
}

# Multi-task suites: name -> ordered list of constituent task names.
MT_TASKS = {
    "robocasa3": [
        "CoffeePressButton",
        "TurnOffMicrowave",
        "TurnOffSinkFaucet",
    ],
    # Sink-manipulation suite (uids 4, 2, 5 — non-contiguous because uids are
    # append-only; the normalizer maps the distinct values fine).
    "sink3": [
        "TurnOnSinkFaucet",
        "TurnOffSinkFaucet",
        "TurnSinkSpout",
    ],
    # Door-manipulation suite (uids 6, 7, 8, 9). 4 tasks x 200 demos -> N800.
    "doors4": [
        "OpenSingleDoor",
        "CloseSingleDoor",
        "OpenDoubleDoor",
        "CloseDoubleDoor",
    ],
    # Baking suite (uids 10, 11). Unlike every suite above, these two tasks have
    # no RoboCasa v0.2 release: their demos come from the RoboCasa **v1.0** drop
    # in LeRobot format and are converted by
    # ``scripts/convert_robocasa_lerobot_to_zarr.py``, which permutes the v1.0
    # action ordering into v0.2's and converts the base-relative eef pose to
    # world frame. Episode counts are whatever v1.0 shipped (101 + 103 = 204),
    # not the 200/task convention used by the v0.2 suites.
    "baking2": [
        "CupcakeCleanup",
        "PastryDisplay",
    ],
}


def is_multitask(task_name: str) -> bool:
    return task_name in MT_TASKS


def get_subtasks(task_name: str) -> List[str]:
    """Return the list of constituent tasks (``[task_name]`` if single-task)."""
    if is_multitask(task_name):
        return MT_TASKS[task_name]
    else:
        return [task_name]


def get_task_uid(task_name: str) -> int:
    """Map a (single) task name to its canonical integer uid."""
    if task_name not in ROBOCASA_TASK_TO_UID:
        raise KeyError(
            f"Unknown RoboCasa task '{task_name}'. Known tasks: "
            f"{ROBOCASA_TASK_NAMES}. Add it to ROBOCASA_TASK_NAMES (append-only)."
        )
    return ROBOCASA_TASK_TO_UID[task_name]
