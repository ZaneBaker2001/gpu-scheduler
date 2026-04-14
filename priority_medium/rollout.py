from __future__ import annotations

import os
from stable_baselines3 import PPO

from priority_high.envs import GPUSchedulerEnv
from priority_low.config import TRAIN_CONFIG, ARTIFACT_DIRS


def main():
    model_path = os.path.join(ARTIFACT_DIRS["models"], "ppo_gpu_scheduler.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python -m priority_medium.train` first."
        )

    model = PPO.load(model_path)

    env = GPUSchedulerEnv(
        n_gpus=TRAIN_CONFIG["n_gpus"],
        episode_n_jobs=40,
        max_queue_visible=TRAIN_CONFIG["max_queue_visible"],
        max_arrival_gap=TRAIN_CONFIG["max_arrival_gap"],
        max_duration=TRAIN_CONFIG["max_duration"],
        seed=777,
    )

    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step_idx = 0

    print("=== PPO GPU Scheduling Rollout ===")
    while not done:
        visible = []
        for i, job in enumerate(env.queue[: env.max_queue_visible]):
            visible.append(
                {
                    "slot": i + 1,
                    "job_id": job.job_id,
                    "gpus": job.gpus_required,
                    "dur": job.duration,
                    "wait": env.current_time - job.submit_time,
                }
            )

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        print(f"\ntime={env.current_time} free_gpus={env.free_gpus}")
        print(f"visible_queue={visible}")
        print(f"chosen_action={action} ({'NOOP' if action == 0 else 'schedule queue slot ' + str(action)})")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_idx += 1
        done = terminated or truncated

    print("\n=== Final Metrics ===")
    print(f"steps:           {step_idx}")
    print(f"total_reward:    {total_reward:.3f}")
    print(f"gpu_utilization: {info['gpu_utilization']:.3f}")
    print(f"mean_wait:       {info['mean_wait']:.3f}")
    print(f"mean_slowdown:   {info['mean_slowdown']:.3f}")
    print(f"throughput:      {info['throughput']:.3f}")
    print(f"completed_jobs:  {info['completed_jobs']}")


if __name__ == "__main__":
    main()