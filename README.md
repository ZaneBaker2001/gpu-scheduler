# GPU Scheduler 

A GPU cluster for scheduling optimization.

## Tech stack


- **Gymnasium** 
- **Stable-Baselines3** 
- **NumPy** 
- **Matplotlib**
- **FastAPI**

## Quick Start 

```bash
pip3 install -r requirements.txt
python3 -m priority_medium.train
python3 -m priority_medium.evaluate
python3 -m priority_medium.rollout
python3 -m uvicorn api.app:app --reload
```

## Docker Setup 

```docker
docker build -t gpu-scheduler .
docker run --rm -it -v "$(pwd)/artifacts:/app/artifacts" gpu-scheduler python -m priority_medium.train
docker run --rm -it -v "$(pwd)/artifacts:/app/artifacts" gpu-scheduler python -m priority_medium.evaluate
docker run --rm -it -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts" gpu-scheduler python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Requests 
Sample simulation request: 

```curl
curl -X POST http://127.0.0.1:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"seed": 777, "episode_n_jobs": 40}'
```
Response:

```json
{"steps":164,
"total_reward":91.74613583330358,"gpu_utilization":0.7580645161290323,"mean_wait":19.9,
"mean_slowdown":2.9657083348153392,"throughput":0.3225806451612903,"completed_jobs":40}
```

Sample prediction request:

```curl
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"observation":[1.0,0.2,0.0,0.0,0.3,1.0,1.0,0.25,0.4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}'
```

Response:

```json
{"action":0,"action_name":"SCHEDULE_SLOT_0"}
```

## Problem

We simulate a GPU cluster with a finite number of GPUs and a queue of ML training jobs.
Each job has:
- required GPU count
- duration
- submit time

The scheduler chooses which queued job to launch whenever resources are available.

## Objective

Optimize for:
- higher GPU utilization
- lower average waiting time
- lower normalized slowdown
- higher throughput

## Baselines

- FCFS
- SJF
- Best-Fit GPU
- Random

## Directory priorities

- `priority_high/`: environment, simulator, and baseline schedulers
- `priority_medium/`: training, evaluation, and experiment runners
- `priority_low/`: config and plotting

## Directory Structure 

```
gpu_scheduler_rl/
├── requirements.txt
├── Dockerfile 
├── README.md
├── api/
│   ├── app.py 
│   ├── schemas.py
│   └── service.py
├── priority_high/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   └── gpu_scheduler_env.py
│   ├── baselines/
│   │   ├── __init__.py
│   │   └── heuristics.py
│   └── simulation/
│       ├── __init__.py
│       └── job_generator.py
├── priority_medium/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   └── rollout.py
└── priority_low/
    ├── __init__.py
    ├── config.py
    └── plotting.py
```
## Reward structure

The reward function combines step-level shaping with episode-level scheduling metrics.

Step-level rewards encourage the agent to:
- launch feasible jobs,
- use available GPUs efficiently,
- avoid invalid actions,
- avoid advancing time while schedulable jobs remain.

When time advances, the reward also reflects short-horizon system behavior such as GPU utilization, job completions, queue pressure, and cluster idleness.

At the end of the episode, the environment adds a terminal reward based on overall scheduler performance:
- higher GPU utilization,
- higher throughput,
- lower mean wait time,
- lower mean slowdown.

This design gives PPO dense learning signals during long episodes while keeping the optimization target aligned with real scheduling objectives.

## Results 

### Evaluation 

PPO achieved near-parity with the heuristic baselines on the GPU scheduling benchmark.

The below graphs illustrate the verage evaluation metrics for each policy:

![GPU Utilization](./gpu_scheduler/artifacts/plots/gpu_utilization.png)

![Mean Slowdown](./gpu_scheduler/artifacts/plots/mean_slowdown.png)

![Mean Wait](./gpu_scheduler/artifacts/plots/mean_wait.png)

![Throughput](./gpu_scheduler/artifacts/plots/throughput.png)

![Reward Comparison](./gpu_scheduler/artifacts/plots/reward_comparison.png)

PPO did not outperform the best heuristic, but it learned a competitive policy with strong overall performance. FCFS and Best-Fit GPU produced the highest utilization and throughput, while SJF achieved the best latency metrics. PPO landed between these extremes, showing that an RL scheduler can recover a balanced scheduling strategy without being manually programmed around a single dispatch rule.


### Rollout

A sample PPO rollout produced the following episode metrics:

- GPU utilization: 0.758
- Mean wait time: 19.9
- Mean slowdown: 2.97
- Throughput: 0.323 jobs per time unit
- Completed jobs: 40

This illustrates the behavior of a trained policy on a single scheduling episode. Final performance comparisons are based on averages over many evaluation episodes rather than a single rollout.
