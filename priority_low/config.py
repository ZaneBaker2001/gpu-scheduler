from __future__ import annotations

import os

ARTIFACT_ROOT = "artifacts"

ARTIFACT_DIRS = {
    "models": os.path.join(ARTIFACT_ROOT, "models"),
    "logs": os.path.join(ARTIFACT_ROOT, "logs"),
    "results": os.path.join(ARTIFACT_ROOT, "results"),
    "plots": os.path.join(ARTIFACT_ROOT, "plots"),
}

TRAIN_CONFIG = {
    "seed": 1234,
    "n_gpus": 8,
    "episode_n_jobs": 200,
    "max_queue_visible": 10,
    "max_arrival_gap": 3,
    "max_duration": 20,
    "n_envs": 8,
    "total_timesteps": 1000000,
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 512,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.001,
}

EVAL_CONFIG = {
    "n_eval_episodes": 100,
}