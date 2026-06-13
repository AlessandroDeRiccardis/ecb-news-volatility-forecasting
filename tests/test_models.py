from __future__ import annotations

import numpy as np

from ecb_vol_forecasting.models import NAGarchAsym


def test_na_garch_scaling_factor_is_positive_and_news_responsive() -> None:
    returns = np.array([0.0, 0.01, -0.01, 0.005])
    p = np.array([0.0, 0.2, 0.8, 1.0])
    n = np.array([0.0, -0.2, -0.8, -1.0])
    model = NAGarchAsym(returns, P=p, N=n)
    theta_news = np.log([0.8, 0.7, 2.0, 2.0])
    scaling = model._scaling_factor(theta_news)
    assert np.isfinite(scaling).all()
    assert (scaling > 0).all()
    assert scaling[-1] > scaling[0]


def test_na_garch_rejects_invalid_a_plus_b() -> None:
    model = NAGarchAsym([0.0, 0.01], P=[0.0, 0.1], N=[0.0, -0.1])
    scaling = model._scaling_factor(np.log([0.1, 0.1, 1.0, 1.0]))
    assert np.isnan(scaling).all()
