"""Unit tests for global seeding."""

import numpy as np
import torch

from csa_yolo.utils import set_seed


def test_seeding_makes_numpy_and_torch_repeatable() -> None:
    set_seed(123)
    a_np, a_t = np.random.rand(5), torch.rand(5)
    set_seed(123)
    b_np, b_t = np.random.rand(5), torch.rand(5)
    assert np.allclose(a_np, b_np)
    assert torch.allclose(a_t, b_t)


def test_different_seeds_differ() -> None:
    set_seed(1)
    a = np.random.rand(5)
    set_seed(2)
    b = np.random.rand(5)
    assert not np.allclose(a, b)
