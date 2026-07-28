"""Global seeding for reproducible experiments.

Every experiment in CSA-YOLO must call :func:`set_seed` before any stochastic
operation so that results are exactly reproducible across runs, which is a
hard requirement for the ablation studies and journal submission.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed every random number generator used by the framework.

    Args:
        seed: The global random seed (recorded in the experiment config).
        deterministic_torch: If True, request deterministic cuDNN/torch
            algorithms. This can reduce GPU throughput slightly but makes
            training runs repeatable; keep it True for all reported results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d (deterministic_torch=%s)", seed, deterministic_torch)
