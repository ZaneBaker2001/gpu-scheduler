from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    observation: List[float] = Field(..., description="Environment observation vector")


class PredictResponse(BaseModel):
    action: int
    action_name: str


class RolloutRequest(BaseModel):
    seed: int = 777
    episode_n_jobs: Optional[int] = None


class RolloutResponse(BaseModel):
    steps: int
    total_reward: float
    gpu_utilization: float
    mean_wait: float
    mean_slowdown: float
    throughput: float
    completed_jobs: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ConfigResponse(BaseModel):
    n_gpus: int
    episode_n_jobs: int
    max_queue_visible: int
    max_arrival_gap: int
    max_duration: int