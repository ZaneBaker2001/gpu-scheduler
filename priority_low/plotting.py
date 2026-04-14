from __future__ import annotations

import csv
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


def plot_training_curve(csv_path: str, save_path: str, smooth_window: int = 20) -> None:
    timesteps = []
    rewards = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timesteps.append(int(row["timestep"]))
            rewards.append(float(row["episode_reward"]))

    if not rewards:
        raise ValueError("Training log is empty")

    rewards_np = np.array(rewards, dtype=np.float32)
    if len(rewards_np) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smooth = np.convolve(rewards_np, kernel, mode="valid")
        smooth_x = timesteps[smooth_window - 1 :]
    else:
        smooth = rewards_np
        smooth_x = timesteps

    _ensure_dir(save_path)
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, rewards, alpha=0.35, label="Episode reward")
    plt.plot(smooth_x, smooth, label=f"Moving average ({smooth_window})")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison_bars(results: dict[str, dict[str, Any]], metric: str, title: str, ylabel: str, save_path: str) -> None:
    labels = list(results.keys())
    means = [results[k][metric] for k in labels]

    _ensure_dir(save_path)
    plt.figure(figsize=(8, 5))
    plt.bar(labels, means)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_metric_bars(results: dict[str, dict[str, Any]], metric: str, title: str, ylabel: str, save_path: str) -> None:
    labels = list(results.keys())
    means = [results[k][metric] for k in labels]

    _ensure_dir(save_path)
    plt.figure(figsize=(8, 5))
    plt.bar(labels, means)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_reward_histogram(results: dict[str, dict[str, Any]], save_path: str) -> None:
    _ensure_dir(save_path)
    plt.figure(figsize=(8, 5))
    for label, stats in results.items():
        plt.hist(stats["rewards"], bins=25, alpha=0.5, label=label)
    plt.xlabel("Episode reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()