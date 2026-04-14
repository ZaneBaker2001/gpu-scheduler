from __future__ import annotations

import os
import numpy as np
from stable_baselines3 import PPO

from priority_high.envs import GPUSchedulerEnv
from priority_low.config import TRAIN_CONFIG, ARTIFACT_DIRS


class SchedulerService:
    def __init__(self) -> None:
        self.model = None
        self.model_path = os.path.join(ARTIFACT_DIRS["models"], "ppo_gpu_scheduler.zip")

    def load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Train first with `python -m priority_medium.train`."
            )
        self.model = PPO.load(self.model_path)

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_action(self, observation: list[float]) -> int:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        obs = np.asarray(observation, dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return int(np.asarray(action).item())

    def action_name(self, action: int) -> str:
        if action == TRAIN_CONFIG["max_queue_visible"]:
            return "ADVANCE_TIME"
        return f"SCHEDULE_SLOT_{action}"

    def run_rollout(self, seed: int = 777, episode_n_jobs: int | None = None) -> dict:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        env = GPUSchedulerEnv(
            n_gpus=TRAIN_CONFIG["n_gpus"],
            episode_n_jobs=episode_n_jobs or TRAIN_CONFIG["episode_n_jobs"],
            max_queue_visible=TRAIN_CONFIG["max_queue_visible"],
            max_arrival_gap=TRAIN_CONFIG["max_arrival_gap"],
            max_duration=TRAIN_CONFIG["max_duration"],
            seed=seed,
        )

        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            action = int(np.asarray(action).item())
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        return {
            "steps": steps,
            "total_reward": float(total_reward),
            "gpu_utilization": float(info["gpu_utilization"]),
            "mean_wait": float(info["mean_wait"]),
            "mean_slowdown": float(info["mean_slowdown"]),
            "throughput": float(info["throughput"]),
            "completed_jobs": int(info["completed_jobs"]),
        }

    def config(self) -> dict:
        return {
            "n_gpus": TRAIN_CONFIG["n_gpus"],
            "episode_n_jobs": TRAIN_CONFIG["episode_n_jobs"],
            "max_queue_visible": TRAIN_CONFIG["max_queue_visible"],
            "max_arrival_gap": TRAIN_CONFIG["max_arrival_gap"],
            "max_duration": TRAIN_CONFIG["max_duration"],
        }