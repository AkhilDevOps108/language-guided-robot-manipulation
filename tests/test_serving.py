import base64
import importlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from gripground.config import TrainConfig
from gripground.models.policy import LanguageConditionedPolicy
from gripground.training.train_policy import save_checkpoint


def _image_payload() -> str:
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_real_checkpoint(path: Path) -> None:
    model = LanguageConditionedPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2):
        image = torch.rand(4, 3, 64, 64)
        tokens = torch.randint(0, 16, (4, 16))
        state = torch.rand(4, 12)
        target = torch.rand(4, 4) * 2 - 1
        pred = model(image, tokens, state)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    save_checkpoint(path, model, TrainConfig(output_dir=path.parent), epoch=1, val_loss=float(loss.item()))


def test_api_success_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = tmp_path / "best.pt"
    _make_real_checkpoint(ckpt)
    monkeypatch.setenv("GRIPGROUND_MODEL_PATH", str(ckpt))
    mod = importlib.import_module("gripground.serving.app")
    mod = importlib.reload(mod)
    client = TestClient(mod.app)
    with client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True
        response = client.post(
            "/predict",
            json={
                "instruction": "move cube to target",
                "image_base64": _image_payload(),
                "state": [0.0] * 12,
            },
        )
        assert response.status_code == 200
        action = response.json()["predicted_action"]
        assert len(action) == 4
        bad = client.post("/predict", json={"instruction": "x", "image_base64": "bad"})
        assert bad.status_code == 422


def test_api_missing_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing.pt"
    monkeypatch.setenv("GRIPGROUND_MODEL_PATH", str(missing))
    mod = importlib.import_module("gripground.serving.app")
    mod = importlib.reload(mod)
    client = TestClient(mod.app)
    with pytest.raises(RuntimeError):
        with client:
            client.get("/health")
