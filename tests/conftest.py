from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_master() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_pre, n_in, n_out = 10, 40, 20
    n = n_pre + n_in + n_out
    returns = rng.normal(0.0, 0.01, n)
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=n),
            "period": ["presample"] * n_pre + ["insample"] * n_in + ["outsample"] * n_out,
            "log_return": returns,
            "sq_return": returns**2,
            "abs_return": np.abs(returns),
            "P_t": np.linspace(0.1, 0.9, n),
            "N_t": -np.linspace(0.9, 0.1, n),
        }
    )
    frame["S_t"] = frame["P_t"] + frame["N_t"]
    frame["sq_return_5d"] = frame["sq_return"].rolling(5).sum().shift(-4)
    return frame
