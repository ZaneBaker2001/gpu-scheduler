from __future__ import annotations

import json
import os
import traceback
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from priority_high.envs import GPUSchedulerEnv
from priority_high.baselines import (
    select_fcfs_action,
    select_sjf_action,
    select_best_fit_gpu_action,
    select_random_action,
)
from priority_low.config import TRAIN_CONFIG, EVAL_CONFIG, ARTIFACT_DIRS
from priority_low.plotting import (
    plot_comparison_bars,
    plot_metric_bars,
    plot_reward_histogram,
)


def _make_env(seed: int) -> GPUSchedulerEnv:
    return GPUSchedulerEnv(
        n_gpus=TRAIN_CONFIG["n_gpus"],
        episode_n_jobs=TRAIN_CONFIG["episode_n_jobs"],
        max_queue_visible=TRAIN_CONFIG["max_queue_visible"],
        max_arrival_gap=TRAIN_CONFIG["max_arrival_gap"],
        max_duration=TRAIN_CONFIG["max_duration"],
        seed=seed,
    )


def run_policy_episode(
    env: GPUSchedulerEnv,
    policy_name: str,
    model: PPO | None = None,
    rng=None,
) -> dict:
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    max_steps = env.max_schedule_ops + 10

    while not done:
        if policy_name == "ppo":
            assert model is not None
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).item())
        elif policy_name == "fcfs":
            action = select_fcfs_action(env)
        elif policy_name == "sjf":
            action = select_sjf_action(env)
        elif policy_name == "best_fit_gpu":
            action = select_best_fit_gpu_action(env)
        elif policy_name == "random":
            action = select_random_action(env, rng=rng)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step_count += 1

        if step_count > max_steps:
            raise RuntimeError(
                f"Episode exceeded max_steps={max_steps}. "
                f"time={env.current_time}, queue={len(env.queue)}, "
                f"running={len(env.running_jobs)}, completed={len(env.completed_jobs)}"
            )

    return {
        "reward": float(total_reward),
        "gpu_utilization": float(info["gpu_utilization"]),
        "mean_wait": float(info["mean_wait"]),
        "mean_slowdown": float(info["mean_slowdown"]),
        "throughput": float(info["throughput"]),
        "completed_jobs": int(info["completed_jobs"]),
    }


def summarize(records: list[dict]) -> dict:
    keys = [
        "reward",
        "gpu_utilization",
        "mean_wait",
        "mean_slowdown",
        "throughput",
        "completed_jobs",
    ]
    out = {}
    for k in keys:
        vals = [r[k] for r in records]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    out["rewards"] = [r["reward"] for r in records]
    return out


def main():
    try:
        for path in ARTIFACT_DIRS.values():
            os.makedirs(path, exist_ok=True)

        model_path = os.path.join(ARTIFACT_DIRS["models"], "ppo_gpu_scheduler.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run `python3 -m priority_medium.train` first."
            )

        debug_env = _make_env(seed=123)
        check_env(debug_env, warn=True)

        model = PPO.load(model_path)

        env_action_n = debug_env.action_space.n
        model_action_n = int(model.policy.action_net.out_features)

        print(f"Env actions:   {env_action_n}")
        print(f"Model actions: {model_action_n}")

        if model_action_n != env_action_n:
            raise RuntimeError(
                "Model/env action-space mismatch. "
                f"Model outputs {model_action_n} actions, "
                f"but env expects {env_action_n}. "
                "Delete artifacts/models/ppo_gpu_scheduler.zip and retrain."
            )

        rng = np.random.default_rng(123)

        policies = ["ppo", "fcfs", "sjf", "best_fit_gpu", "random"]
        all_results = {}

        for policy in policies:
            records = []
            for ep in range(EVAL_CONFIG["n_eval_episodes"]):
                env = _make_env(seed=10_000 + ep)
                result = run_policy_episode(env, policy_name=policy, model=model, rng=rng)
                records.append(result)
            all_results[policy] = summarize(records)

        results_path = os.path.join(ARTIFACT_DIRS["results"], "evaluation.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        plot_comparison_bars(
            results=all_results,
            metric="reward_mean",
            title="Average Episode Reward",
            ylabel="Reward",
            save_path=os.path.join(ARTIFACT_DIRS["plots"], "reward_comparison.png"),
        )

        plot_metric_bars(
            results=all_results,
            metric="gpu_utilization_mean",
            title="GPU Utilization",
            ylabel="Utilization",
            save_path=os.path.join(ARTIFACT_DIRS["plots"], "gpu_utilization.png"),
        )

        plot_metric_bars(
            results=all_results,
            metric="mean_wait_mean",
            title="Average Waiting Time",
            ylabel="Time",
            save_path=os.path.join(ARTIFACT_DIRS["plots"], "mean_wait.png"),
        )

        plot_metric_bars(
            results=all_results,
            metric="mean_slowdown_mean",
            title="Average Slowdown",
            ylabel="Slowdown",
            save_path=os.path.join(ARTIFACT_DIRS["plots"], "mean_slowdown.png"),
        )

        plot_metric_bars(
            results=all_results,
            metric="throughput_mean",
            title="Throughput",
            ylabel="Jobs per unit time",
            save_path=os.path.join(ARTIFACT_DIRS["plots"], "throughput.png"),
        )

        plot_reward_histogram(
            results=all_results,
            save_path=os.path.join(ARTIFACT_DIRS["plots"], "reward_histogram.png"),
        )

        printable = {
            k: {
                "reward_mean": round(v["reward_mean"], 3),
                "gpu_utilization_mean": round(v["gpu_utilization_mean"], 3),
                "mean_wait_mean": round(v["mean_wait_mean"], 3),
                "mean_slowdown_mean": round(v["mean_slowdown_mean"], 3),
                "throughput_mean": round(v["throughput_mean"], 3),
            }
            for k, v in all_results.items()
        }
        print(json.dumps(printable, indent=2))
        print(f"Saved evaluation results to: {results_path}")

    except Exception:
        print("\n=== FULL TRACEBACK ===")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()