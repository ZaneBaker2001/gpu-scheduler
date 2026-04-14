from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Job:
    job_id: int
    submit_time: int
    gpus_required: int
    duration: int
    remaining_time: int | None = None
    start_time: int | None = None
    finish_time: int | None = None

    def __post_init__(self):
        if self.remaining_time is None:
            self.remaining_time = self.duration


def generate_job_sequence(
    n_jobs: int,
    n_gpus: int,
    seed: int | None = None,
    max_arrival_gap: int = 3,
    max_duration: int = 20,
) -> list[Job]:
    """
    Generates a synthetic stream of GPU jobs.
    This imitates mixed small/medium/large training jobs.
    """
    rng = np.random.default_rng(seed)

    jobs: list[Job] = []
    t = 0
    for job_id in range(n_jobs):
        t += int(rng.integers(0, max_arrival_gap + 1))

        # GPU demand skewed toward smaller jobs, with some larger jobs
        demand_bucket = rng.choice([1, 2, 4], p=[0.55, 0.30, 0.15])
        gpus_required = int(min(demand_bucket, n_gpus))

        duration = int(rng.integers(2, max_duration + 1))

        jobs.append(
            Job(
                job_id=job_id,
                submit_time=t,
                gpus_required=gpus_required,
                duration=duration,
            )
        )
    return jobs