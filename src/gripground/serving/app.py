from __future__ import annotations

import base64
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image

from gripground.data.dataset import tokenize_instruction
from gripground.models.policy import LoadedPolicy, load_policy_checkpoint
from gripground.serving.schemas import PredictRequest, PredictResponse

app = FastAPI(title="GripGround Inference API", version="0.1.0")
_LOADED: LoadedPolicy | None = None
_MODEL_PATH = Path(os.environ.get("GRIPGROUND_MODEL_PATH", "artifacts/checkpoints/best.pt"))


def _decode_image(image_base64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image_base64)
        with Image.open(BytesIO(raw)) as img:
            img = img.convert("RGB").resize((64, 64))
            return np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image payload: {exc}") from exc


@app.on_event("startup")
def startup() -> None:
    global _LOADED
    if not _MODEL_PATH.exists():
        raise RuntimeError(f"Model checkpoint not found: {_MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOADED = load_policy_checkpoint(str(_MODEL_PATH), device)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": _LOADED is not None}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    if _LOADED is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    meta = _LOADED.checkpoint_meta
    return {
        "model_type": meta.get("model_type"),
        "action_dim": meta.get("action_dim", 4),
        "hidden_dim": meta.get("hidden_dim", 128),
        "checkpoint_path": str(_MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if _LOADED is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    image = _decode_image(payload.image_base64)
    state = np.asarray(payload.state if payload.state is not None else [0.0] * 11, dtype=np.float32)
    instruction_tokens = tokenize_instruction(payload.instruction)
    t0 = time.perf_counter()
    action = _LOADED.model.predict(image, instruction_tokens, state, _LOADED.device)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    if not np.isfinite(action).all():
        raise HTTPException(status_code=500, detail="Model produced invalid action values")
    return PredictResponse(
        predicted_action=action.tolist(),
        model_version="0.1.0",
        inference_latency_ms=latency_ms,
    )

