from __future__ import annotations

import csv
import os
from dataclasses import dataclass

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from priority_high.envs import GPUSchedulerEnv
from priority_low.config import TRAIN_CONFIG, ARTIFACT_DIRS
from priority_low.plotting import plot_training_curve


@dataclass
class EpisodeRecord:
    timestep: int
    episode_reward: float


class RewardLoggingCallback(BaseCallback):
    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.records: list[EpisodeRecord] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.records.append(
                    EpisodeRecord(
                        timestep=int(self.num_timesteps),
                        episode_reward=float(ep["r"]),
                    )
                )
        return True

    def _on_training_end(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "episode_reward"])
            for rec in self.records:
                writer.writerow([rec.timestep, rec.episode_reward])


def make_env(seed_offset: int):
    def _init():
        env = GPUSchedulerEnv(
            n_gpus=TRAIN_CONFIG["n_gpus"],
            episode_n_jobs=TRAIN_CONFIG["episode_n_jobs"],
            max_queue_visible=TRAIN_CONFIG["max_queue_visible"],
            max_arrival_gap=TRAIN_CONFIG["max_arrival_gap"],
            max_duration=TRAIN_CONFIG["max_duration"],
            seed=TRAIN_CONFIG["seed"] + seed_offset,
        )
        return Monitor(env)
    return _init


def main():
    for path in ARTIFACT_DIRS.values():
        os.makedirs(path, exist_ok=True)

    vec_env = DummyVecEnv([make_env(i) for i in range(TRAIN_CONFIG["n_envs"])])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=TRAIN_CONFIG["learning_rate"],
        n_steps=TRAIN_CONFIG["n_steps"],
        batch_size=TRAIN_CONFIG["batch_size"],
        n_epochs=TRAIN_CONFIG["n_epochs"],
        gamma=TRAIN_CONFIG["gamma"],
        gae_lambda=TRAIN_CONFIG["gae_lambda"],
        clip_range=TRAIN_CONFIG["clip_range"],
        ent_coef=TRAIN_CONFIG["ent_coef"],
        seed=TRAIN_CONFIG["seed"],
        verbose=1,
    )

    reward_csv = os.path.join(ARTIFACT_DIRS["logs"], "training_rewards.csv")
    callback = RewardLoggingCallback(reward_csv)

    model.learn(
        total_timesteps=TRAIN_CONFIG["total_timesteps"],
        callback=callback,
    )

    model_path = os.path.join(ARTIFACT_DIRS["models"], "ppo_gpu_scheduler.zip")
    model.save(model_path)

    plot_path = os.path.join(ARTIFACT_DIRS["plots"], "training_curve.png")
    plot_training_curve(reward_csv, plot_path)

    print(f"Saved model: {model_path}")
    print(f"Saved log:   {reward_csv}")
    print(f"Saved plot:  {plot_path}")


if __name__ == "__main__":
    main()