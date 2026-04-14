from __future__ import annotations

from collections import deque
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from priority_high.simulation import Job, generate_job_sequence


class GPUSchedulerEnv(gym.Env):
    """
    GPU cluster scheduling environment with explicit scheduling phase.

    Actions:
      0..max_queue_visible-1 : schedule the job in that visible queue slot
      max_queue_visible      : ADVANCE_TIME

    Key difference from the previous version:
    - scheduling a job does NOT advance time
    - time advances only when the agent chooses ADVANCE_TIME

    This makes the problem closer to real schedulers, which usually keep
    placing jobs until no good placement remains.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_gpus: int = 8,
        episode_n_jobs: int = 200,
        max_queue_visible: int = 10,
        max_arrival_gap: int = 3,
        max_duration: int = 20,
        seed: int | None = None,
        max_time_factor: int = 10,
        max_schedule_ops_factor: int = 20,
    ) -> None:
        super().__init__()

        self.n_gpus = n_gpus
        self.episode_n_jobs = episode_n_jobs
        self.max_queue_visible = max_queue_visible
        self.max_arrival_gap = max_arrival_gap
        self.max_duration = max_duration
        self.seed_value = seed
        self.max_time_factor = max_time_factor
        self.max_schedule_ops_factor = max_schedule_ops_factor

        # 0..max_queue_visible-1 = queue slot
        # max_queue_visible = ADVANCE_TIME
        self.advance_action = self.max_queue_visible
        self.action_space = spaces.Discrete(self.max_queue_visible + 1)

        # Global features:
        # [free_gpus_ratio, queue_len_ratio, running_jobs_ratio, time_norm, feasible_count_ratio]
        # Per visible slot:
        # [present, feasible, gpus_req_norm, duration_norm, wait_norm]
        obs_dim = 5 + self.max_queue_visible * 5
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.rng = np.random.default_rng(seed)

        self.all_jobs: list[Job] = []
        self.future_jobs: deque[Job] = deque()
        self.queue: list[Job] = []
        self.running_jobs: list[Job] = []
        self.completed_jobs: list[Job] = []

        self.current_time: int = 0
        self.free_gpus: int = self.n_gpus

        self.total_gpu_time_used: float = 0.0
        self.total_time_elapsed: int = 0
        self.max_episode_time: int = self.episode_n_jobs * self.max_time_factor
        self.max_schedule_ops: int = self.episode_n_jobs * self.max_schedule_ops_factor
        self.schedule_ops_count: int = 0

    def _reset_state(self) -> None:
        episode_seed = int(self.rng.integers(0, 2**31 - 1))

        self.all_jobs = generate_job_sequence(
            n_jobs=self.episode_n_jobs,
            n_gpus=self.n_gpus,
            seed=episode_seed,
            max_arrival_gap=self.max_arrival_gap,
            max_duration=self.max_duration,
        )
        self.future_jobs = deque(copy.deepcopy(self.all_jobs))
        self.queue = []
        self.running_jobs = []
        self.completed_jobs = []

        self.current_time = 0
        self.free_gpus = self.n_gpus
        self.total_gpu_time_used = 0.0
        self.total_time_elapsed = 0
        self.schedule_ops_count = 0

        self._admit_arrivals()

    def _admit_arrivals(self) -> None:
        while self.future_jobs and self.future_jobs[0].submit_time <= self.current_time:
            self.queue.append(self.future_jobs.popleft())

    def _visible_queue(self) -> list[Job]:
        return self.queue[: self.max_queue_visible]

    def _feasible_visible_indices(self) -> list[int]:
        feasible = []
        for i, job in enumerate(self._visible_queue()):
            if job.gpus_required <= self.free_gpus:
                feasible.append(i)
        return feasible

    def _advance_time(self) -> dict:
        self.total_gpu_time_used += (self.n_gpus - self.free_gpus)
        self.total_time_elapsed += 1
        self.current_time += 1

        finished: list[Job] = []
        for job in self.running_jobs:
            job.remaining_time -= 1
            if job.remaining_time <= 0:
                job.finish_time = self.current_time
                finished.append(job)

        for job in finished:
            self.running_jobs.remove(job)
            self.completed_jobs.append(job)
            self.free_gpus += job.gpus_required

        self._admit_arrivals()
        return {"finished_jobs": len(finished)}

    def _launch_job(self, queue_idx: int) -> tuple[bool, Job | None]:
        visible = self._visible_queue()
        if queue_idx < 0 or queue_idx >= len(visible):
            return False, None

        job = self.queue[queue_idx]
        if job.gpus_required > self.free_gpus:
            return False, None

        job.start_time = self.current_time
        self.free_gpus -= job.gpus_required
        self.running_jobs.append(job)
        del self.queue[queue_idx]
        return True, job

    def _gpu_utilization(self) -> float:
        if self.total_time_elapsed <= 0:
            return 0.0
        return float(self.total_gpu_time_used / (self.total_time_elapsed * self.n_gpus))

    def _mean_wait(self) -> float:
        waits = [
            job.start_time - job.submit_time
            for job in self.completed_jobs
            if job.start_time is not None
        ]
        return float(np.mean(waits)) if waits else 0.0

    def _mean_slowdown(self) -> float:
        vals = []
        for job in self.completed_jobs:
            if job.start_time is None or job.finish_time is None:
                continue
            turnaround = job.finish_time - job.submit_time
            slowdown = turnaround / max(job.duration, 1)
            vals.append(slowdown)
        return float(np.mean(vals)) if vals else 0.0

    def _throughput(self) -> float:
        if self.current_time <= 0:
            return 0.0
        return float(len(self.completed_jobs) / self.current_time)

    def _build_info(self, done: bool) -> dict:
        return {
            "time": int(self.current_time),
            "free_gpus": int(self.free_gpus),
            "queue_length": int(len(self.queue)),
            "running_jobs": int(len(self.running_jobs)),
            "completed_jobs": int(len(self.completed_jobs)),
            "gpu_utilization": float(self._gpu_utilization()),
            "mean_wait": float(self._mean_wait()),
            "mean_slowdown": float(self._mean_slowdown()),
            "throughput": float(self._throughput()),
            "feasible_visible_jobs": int(len(self._feasible_visible_indices())),
            "done": bool(done),
        }

    def _get_obs(self) -> np.ndarray:
        current_time_norm = min(self.current_time / max(self.max_episode_time, 1), 1.0)
        visible = self._visible_queue()
        feasible = set(self._feasible_visible_indices())

        features = [
            self.free_gpus / self.n_gpus,
            min(len(self.queue) / max(self.max_queue_visible * 2, 1), 1.0),
            min(len(self.running_jobs) / self.n_gpus, 1.0),
            current_time_norm,
            len(feasible) / max(self.max_queue_visible, 1),
        ]

        for i in range(self.max_queue_visible):
            if i < len(visible):
                job = visible[i]
                wait = self.current_time - job.submit_time
                features.extend(
                    [
                        1.0,
                        1.0 if i in feasible else 0.0,
                        job.gpus_required / self.n_gpus,
                        job.duration / self.max_duration,
                        min(wait / max(self.max_episode_time, 1), 1.0),
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return np.asarray(features, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if seed is not None:
            self.seed_value = seed
            self.rng = np.random.default_rng(seed)

        self._reset_state()
        return self._get_obs(), self._build_info(done=False)

    def step(self, action: int):
     action = int(np.asarray(action).item())

     if action < 0 or action >= self.action_space.n:
        action = self.advance_action

     reward = 0.0
     terminated = False
     truncated = False

     self.schedule_ops_count += 1

     feasible_visible = self._feasible_visible_indices()
     had_feasible_before = len(feasible_visible) > 0
     used_gpus_before = self.n_gpus - self.free_gpus

     if action == self.advance_action:
        advance_info = self._advance_time()

        used_gpus_after = self.n_gpus - self.free_gpus
        util = used_gpus_after / self.n_gpus

        # Much softer shaping
        reward += 0.25 * util
        reward += 1.00 * advance_info["finished_jobs"]

        # Mild penalty only when agent idles despite feasible work
        if had_feasible_before and used_gpus_before < self.n_gpus:
            reward -= 0.05

        # Mild queue pressure penalty
        reward -= 0.001 * len(self.queue)

     else:
        success, launched = self._launch_job(action)
        if success and launched is not None:
            # Reward launching feasible work
            reward += 0.30
            reward += 0.05 * launched.gpus_required

            # Small bonus for older waiting jobs
            wait = self.current_time - launched.submit_time
            reward += min(0.02 * wait, 0.20)

            # Small packing bonus for using scarce GPUs
            used_after_launch = self.n_gpus - self.free_gpus
            reward += 0.10 * (used_after_launch / self.n_gpus)
        else:
            # Softer invalid-action penalty
            reward -= 0.08

     if not self.future_jobs and not self.queue and not self.running_jobs:
        terminated = True

     if self.current_time >= self.max_episode_time:
        truncated = True

     if self.schedule_ops_count >= self.max_schedule_ops:
        truncated = True

     if terminated or truncated:
        final_util = self._gpu_utilization()
        final_wait = self._mean_wait()
        final_slowdown = self._mean_slowdown()
        final_throughput = self._throughput()

        # Softer terminal objective
        reward += 8.0 * final_util
        reward += 6.0 * final_throughput
        reward -= 0.002 * final_wait
        reward -= 0.02 * max(final_slowdown - 1.0, 0.0)

     obs = self._get_obs()
     info = self._build_info(done=(terminated or truncated))
     return obs, float(reward), terminated, truncated, info

    def render(self):
        print(
            {
                "time": self.current_time,
                "free_gpus": self.free_gpus,
                "advance_action": self.advance_action,
                "queue": [
                    {
                        "slot": i,
                        "job_id": j.job_id,
                        "gpus": j.gpus_required,
                        "dur": j.duration,
                        "wait": self.current_time - j.submit_time,
                        "feasible": j.gpus_required <= self.free_gpus,
                    }
                    for i, j in enumerate(self._visible_queue())
                ],
                "running": [
                    {
                        "job_id": j.job_id,
                        "gpus": j.gpus_required,
                        "remaining": j.remaining_time,
                    }
                    for j in self.running_jobs
                ],
                "completed_jobs": len(self.completed_jobs),
            }
        )