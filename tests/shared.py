import os
from functools import wraps

import pytest

model_names = ["resnet18"]

# datasets in configs/dataset
dataset_names = [
    "lensless"  # , "whoi-plankton"
]


def skip_in_github_ci(func):
    @wraps(func)
    def run_if_not_gh_ci(*args, **kwargs):
        if os.getenv("GITHUB_ACTIONS") == "true":
            pytest.skip("Test skipped on CI to avoid disk full/time-out errors.")
        return func(*args, **kwargs)

    return run_if_not_gh_ci
