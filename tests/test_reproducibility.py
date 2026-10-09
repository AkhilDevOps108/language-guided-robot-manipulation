import numpy as np
import torch

from gripground.utils.reproducibility import set_global_seed


def test_seed_reproducibility() -> None:
    set_global_seed(11)
    a = np.random.rand(4)
    b = torch.randn(4)
    set_global_seed(11)
    a2 = np.random.rand(4)
    b2 = torch.randn(4)
    assert np.allclose(a, a2)
    assert torch.allclose(b, b2)

