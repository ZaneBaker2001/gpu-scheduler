from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import (
    ConfigResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RolloutRequest,
    RolloutResponse,
)
from api.service import SchedulerService

service = SchedulerService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service.load_model()
    except FileNotFoundError:
        # Service can still start, but prediction endpoints will fail until model exists.
        pass
    yield


app = FastAPI(
    title="GPU Scheduler RL API",
    version="1.0.0",
    description="FastAPI service for PPO-based GPU job scheduling",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=service.is_loaded())


@app.get("/config", response_model=ConfigResponse)
def get_config() -> ConfigResponse:
    return ConfigResponse(**service.config())


@app.post("/reload")
def reload_model() -> dict:
    try:
        service.load_model()
        return {"status": "ok", "message": "Model reloaded"}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        action = service.predict_action(req.observation)
        return PredictResponse(
            action=action,
            action_name=service.action_name(action),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/simulate", response_model=RolloutResponse)
def simulate(req: RolloutRequest) -> RolloutResponse:
    try:
        result = service.run_rollout(seed=req.seed, episode_n_jobs=req.episode_n_jobs)
        return RolloutResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc