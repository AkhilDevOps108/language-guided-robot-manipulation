from pathlib import Path

import torch

from gripground.config import TrainConfig
from gripground.models.policy import LanguageConditionedPolicy, load_policy_checkpoint
from gripground.training.train_policy import save_checkpoint


def test_checkpoint_save_load(tmp_path: Path) -> None:
    model = LanguageConditionedPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    image = torch.rand(2, 3, 64, 64)
    tokens = torch.randint(0, 16, (2, 16))
    state = torch.rand(2, 12)
    target = torch.rand(2, 4) * 2 - 1
    pred = model(image, tokens, state)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    opt.step()

    ckpt = tmp_path / "trained.pt"
    save_checkpoint(ckpt, model, TrainConfig(output_dir=tmp_path), epoch=1, val_loss=0.1)
    loaded = load_policy_checkpoint(str(ckpt), torch.device("cpu"))
    assert isinstance(loaded.model, LanguageConditionedPolicy)
