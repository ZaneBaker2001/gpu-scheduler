from __future__ import annotations

import numpy as np


def _visible_queue(env):
    return env.queue[: env.max_queue_visible]


def _feasible_job_indices(env) -> list[int]:
    feasible = []
    for i, job in enumerate(_visible_queue(env)):
        if job.gpus_required <= env.free_gpus:
            feasible.append(i)
    return feasible


def select_fcfs_action(env) -> int:
    feasible = _feasible_job_indices(env)
    if feasible:
        return feasible[0]
    return env.advance_action


def select_sjf_action(env) -> int:
    feasible = _feasible_job_indices(env)
    if not feasible:
        return env.advance_action

    visible = _visible_queue(env)
    best = min(feasible, key=lambda idx: visible[idx].duration)
    return best


def select_best_fit_gpu_action(env) -> int:
    feasible = _feasible_job_indices(env)
    if not feasible:
        return env.advance_action

    visible = _visible_queue(env)
    best = min(feasible, key=lambda idx: env.free_gpus - visible[idx].gpus_required)
    return best


def select_random_action(env, rng: np.random.Generator | None = None) -> int:
    if rng is None:
        rng = np.random.default_rng()

    feasible = _feasible_job_indices(env)
    if not feasible:
        return env.advance_action

    return int(rng.choice(feasible))