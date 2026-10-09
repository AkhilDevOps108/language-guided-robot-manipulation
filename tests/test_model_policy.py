import numpy as np
import torch

from gripground.models.policy import LanguageConditionedPolicy


def test_policy_shape_and_bounds() -> None:
    model = LanguageConditionedPolicy()
    image = torch.rand(2, 3, 64, 64)
    tokens = torch.randint(0, 20, (2, 16))
    state = torch.rand(2, 11)
    out = model(image, tokens, state)
    assert out.shape == (2, 4)
    assert torch.max(out).item() <= 1.0
    assert torch.min(out).item() >= -1.0


def test_predict_numpy() -> None:
    model = LanguageConditionedPolicy()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    state = np.zeros((11,), dtype=np.float32)
    tokens = np.zeros((16,), dtype=np.int64)
    action = model.predict(img, tokens, state, device=torch.device("cpu"))
    assert action.shape == (4,)

