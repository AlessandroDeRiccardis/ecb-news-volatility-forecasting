"""
models.py — model classes for the NA-GARCH / H_t volatility paper
=================================================================

Six model classes implementing Blocks 1–3 + the hockey-stick robustness
of Block 5.6:

    GARCH11     (B1.1)  — plain GARCH(1,1), wrapped from `arch` package
    GJRGARCH    (B1.2)  — GJR-GARCH(1,1), wrapped from `arch` package
    EGARCH      (B1.3)  — EGARCH(1,1), the benchmark Sadik et al. used
    NAGarchNet  (B2.1)  — NA-GARCH with net stance S_t (Sadik replication)
    NAGarchAsym (B2.2)  — NA-GARCH with asymmetric P_t, N_t
    NAGarchHt   (B3.1)  — NA-GARCH with H_t × stance interaction (MAIN MODEL)
    NAGarchHockey (B5.6) — continuous hockey-stick H_t, otherwise = B3.1

Why both GJR and EGARCH as benchmarks: Sadik et al. (2018) compared their
NA-GARCH against EGARCH only. Including EGARCH lets us replicate their
comparison directly. GJR is a stronger benchmark that captures the leverage
effect within a GARCH(1,1) framework. If NA-GARCH beats EGARCH but loses
to GJR, the paper's honest reading is: "the news effect replicates Sadik's
result against EGARCH but does not survive against GJR's asymmetric-shock
mechanism."

All classes share a common interface:
    .fit()                    → returns self; populates .params, .loglik, .aic, .nobs
    .conditional_variance()   → in-sample fitted σ²_t series
    .forecast_variance(h=1)   → out-of-sample variance forecasts at horizon h
    .residuals_standardized() → ε_t / σ_t for residual diagnostics

SPECIFICATION (Sadik et al. 2018 NA-GARCH framework, dovish/hawkish substitution)
─────────────────────────────────────────────────────────────────────────────────

Variance equation common form:
    σ²_t = scaling_factor_t · (ω + α·ε²_{t-1} + β·σ²_{t-1})

With:
    GARCH11:        scaling_factor_t = 1   (no scaling)
    GJRGARCH:       σ²_t = ω + (α + γ_lev·𝟙{ε_{t-1}<0})·ε²_{t-1} + β·σ²_{t-1}
                    (different functional form, no multiplicative scaling)
    NAGarchNet:     scaling_factor_t = a + 0.5·b·tanh(κ·S_{t-1}/2)
    NAGarchAsym:    scaling_factor_t = a + 0.5·b·[tanh(κ·P_{t-1}/2) − tanh(γ·N_{t-1}/2)]
    NAGarchHt:      scaling_factor_t = a + 0.5·(b + δ·H_{t-1})·[tanh(κ·P_{t-1}/2) − tanh(γ·N_{t-1}/2)]
    NAGarchHockey:  same as Ht but with H_hockey replacing the binary H

Where:
    H_{t-1} = 1{roll_vol_20d_{t-1} > c · long_run_mean_20d}    (binary)
    H_hockey_{t-1} = max(0, roll_vol_20d_{t-1} / long_run_mean_20d − c)

(N_t is in [-1, 0] by Sadik's convention, so −tanh(γN/2) ≥ 0.)

Constraints (enforced by L-BFGS-B bounds + parameterization):
    ω > 0, α ≥ 0, β ≥ 0, α + β < 1
    a ≥ 0, b ≥ 0, 0.5 ≤ a + b ≤ 2
    κ, γ ≥ 0
    b + δ ≥ 0  (so news scaling non-negative when H=1)
    ν ≥ 4 for Student-t  (so kurtosis exists)

ESTIMATION
----------
Maximum likelihood under either Student-t (default, primary spec) or
Gaussian innovations. Optimizer: L-BFGS-B with bounds. The threshold
parameter c in NAGarchHt and NAGarchHockey is NOT estimated jointly — it's
held fixed by the caller (grid search done outside, in estimate_in_sample.py).
At any fixed c, the H series is a precomputed 0/1 vector (or continuous
non-negative vector for hockey), so the likelihood is smooth in the other
parameters and gradient methods work fine.

USAGE
-----
    from models import GARCH11, NAGarchHt
    df = pd.read_csv("output/model_data_master.csv")
    insample = df[df["period"]=="insample"].dropna()

    m = GARCH11(insample["log_return"], dist="studentst").fit()
    print(m.aic, m.params)

    m3 = NAGarchHt(
        insample["log_return"],
        P=insample["P_t"],
        N=insample["N_t"],
        roll_vol=insample["roll_vol_20d"],
        long_run_mean=insample["long_run_mean_20d"].iloc[0],
        c=1.5,
        dist="studentst",
    ).fit()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# arch is only needed for B1.1 / B1.2; soft import so models.py stays usable
# when only NA-GARCH variants are needed.
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

# Numba is optional — if available, the variance recursion is JIT-compiled
# (10-30× speedup, dominant gain for bootstrap and OOS re-estimation).
# If not available, we fall back to a pure-numpy version with identical
# numeric behavior.
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        # No-op decorator if numba not installed
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def _wrap(f):
            return f
        return _wrap


@njit(cache=True)
def _variance_recursion(eps, scaling, omega, alpha, beta, init_var):
    """
    Standalone variance-recursion kernel:
        σ²_0 = init_var
        σ²_t = scaling[t] · (ω + α·ε²_{t-1} + β·σ²_{t-1}),  t = 1..T-1

    Extracted as a free function so numba can JIT-compile it. Identical
    numeric behavior to the pure-Python equivalent.
    """
    T = len(eps)
    sigma_sq = np.empty(T)
    sigma_sq[0] = init_var
    for t in range(1, T):
        garch = omega + alpha * eps[t - 1] * eps[t - 1] + beta * sigma_sq[t - 1]
        sigma_sq[t] = scaling[t] * garch
    return sigma_sq


# ─────────────────────────────────────────────────────────────────────────────
# LOG-LIKELIHOOD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _loglik_studentt(eps, sigma_sq, nu):
    """
    Sum of Student-t log-likelihoods of `eps` under conditional variance
    `sigma_sq` and degrees-of-freedom `nu`. Uses the standard parameterization
    where Var(eps) = sigma_sq for nu > 2 (so the student-t is rescaled by
    (nu-2)/nu internally).
    """
    z2 = eps * eps / sigma_sq
    half_nu_p1 = 0.5 * (nu + 1.0)
    log_const = (gammaln(half_nu_p1) - gammaln(0.5 * nu)
                 - 0.5 * np.log(np.pi * (nu - 2.0)))
    log_lik = (log_const
               - 0.5 * np.log(sigma_sq)
               - half_nu_p1 * np.log1p(z2 / (nu - 2.0)))
    return float(log_lik.sum())


def _loglik_normal(eps, sigma_sq):
    return float(
        (-0.5 * np.log(2.0 * np.pi * sigma_sq) - 0.5 * eps * eps / sigma_sq).sum()
    )


def _negloglik(eps, sigma_sq, dist, nu=None):
    if dist == "normal":
        return -_loglik_normal(eps, sigma_sq)
    elif dist == "studentst":
        return -_loglik_studentt(eps, sigma_sq, nu)
    raise ValueError(f"unknown dist: {dist}")


def _aic(loglik, n_params):
    return 2.0 * n_params - 2.0 * loglik


def _validate_finite(x, name):
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values; clean before fitting.")


# ─────────────────────────────────────────────────────────────────────────────
# BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ModelBase:
    """Common interface."""
    name = "base"

    def __init__(self, returns, dist="studentst"):
        if dist not in ("studentst", "normal"):
            raise ValueError("dist must be 'studentst' or 'normal'")
        self.dist = dist
        self.returns = (returns.values if hasattr(returns, "values")
                        else np.asarray(returns)).astype(float)
        _validate_finite(self.returns, "returns")
        self.dates = (returns.index if hasattr(returns, "index") else None)
        self.nobs = len(self.returns)
        self.fitted = False
        self.params: dict[str, float] = {}
        self.loglik: Optional[float] = None
        self.aic:    Optional[float] = None
        self._sigma_sq_in = None

    def fit(self):
        raise NotImplementedError

    def conditional_variance(self):
        """In-sample fitted σ²_t series, indexed to dates if available."""
        if not self.fitted:
            raise RuntimeError("call .fit() first")
        if self.dates is not None:
            return pd.Series(self._sigma_sq_in, index=self.dates,
                             name=f"sigma2_{self.name}")
        return self._sigma_sq_in

    def residuals_standardized(self):
        if not self.fitted:
            raise RuntimeError("call .fit() first")
        sigma = np.sqrt(self._sigma_sq_in)
        z = self.returns / sigma
        return (pd.Series(z, index=self.dates) if self.dates is not None else z)


# ─────────────────────────────────────────────────────────────────────────────
# B1.1 GARCH(1,1) and B1.2 GJR-GARCH(1,1) via the `arch` package
# ─────────────────────────────────────────────────────────────────────────────

class _ArchWrapper(ModelBase):
    """
    Common wrapper around arch.arch_model. Subclasses set
        _vol_kind ∈ {"GARCH", "EGARCH"}
        _o (asymmetric-shock order, 1 for GJR/EGARCH-with-leverage, else 0)
    The arch package internally scales returns to ~1; we feed it returns × 100
    (returns in percent) to keep the optimizer well-conditioned, then unscale
    estimated ω back to per-unit return where appropriate.
    """
    _vol_kind = "GARCH"
    _o = 0
    _SCALE = 100.0
    name = "arch-base"

    def fit(self):
        if not ARCH_AVAILABLE:
            raise ImportError("arch not installed. pip install arch")
        am = arch_model(
            self.returns * self._SCALE,
            mean="Zero",                      # zero-mean returns
            vol=self._vol_kind,
            p=1, o=self._o, q=1,
            dist=self.dist,
            rescale=False,                    # we already scaled
        )
        res = am.fit(disp="off")
        # arch reports parameters for percent-returns; convert ω back to
        # raw-return units. The conversion differs by volatility model:
        #   GARCH/GJR : ω is in variance units (percent²) → divide by SCALE²
        #   EGARCH    : ω is in log-variance units → subtract (1-β)·log(SCALE²)
        # α, β, γ, ν are all scale-invariant.
        self.params = {}
        beta_arch = float(res.params.get("beta[1]", 0.0))
        for k, v in res.params.items():
            if k == "omega":
                if self._vol_kind == "EGARCH":
                    self.params["omega"] = float(v) - (1.0 - beta_arch) * 2.0 * np.log(self._SCALE)
                else:
                    self.params["omega"] = float(v) / (self._SCALE * self._SCALE)
            elif k in ("alpha[1]", "beta[1]"):
                self.params[k.replace("[1]", "")] = float(v)
            elif k == "gamma[1]":
                self.params["gamma_lev"] = float(v)
            elif k == "nu":
                self.params["nu"] = float(v)
            else:
                self.params[k] = float(v)
        # Recover σ²_t in raw-return units
        self._sigma_sq_in = (np.asarray(res.conditional_volatility) ** 2) / (self._SCALE ** 2)
        # Density change-of-variable: returns scaled by SCALE means each
        # density carries an extra +log(SCALE), so loglik on raw returns is
        # arch's reported loglik + T·log(SCALE) (Jacobian).
        self.loglik = float(res.loglikelihood) + self.nobs * np.log(self._SCALE)
        # AIC under raw-return scaling
        self.aic = _aic(self.loglik, n_params=len(res.params))
        self._arch_res = res
        self.fitted = True
        return self

    def forecast_variance(self, h=1, simulations=1000, seed=42):
        """
        h-step-ahead variance forecast (returns σ²_{T+1}, …, σ²_{T+h}, in
        raw-return units).

        For GARCH/GJR with h ≥ 1: arch supplies analytic forecasts.
        For EGARCH with h > 1: no analytic form; use Monte Carlo simulation
        (default 1000 paths) under the fitted parameters.
        """
        if not self.fitted:
            raise RuntimeError("call .fit() first")
        if self._vol_kind == "EGARCH" and h > 1:
            f = self._arch_res.forecast(
                horizon=h, method="simulation", simulations=simulations,
                rng=np.random.default_rng(seed).standard_normal, reindex=False,
            )
        else:
            f = self._arch_res.forecast(horizon=h, reindex=False)
        # arch returns variance forecasts in percent² units
        return np.asarray(f.variance.iloc[-1]) / (self._SCALE ** 2)


class GARCH11(_ArchWrapper):
    """B1.1 — plain GARCH(1,1)."""
    name = "GARCH(1,1)"
    _vol_kind = "GARCH"
    _o = 0


class GJRGARCH(_ArchWrapper):
    """B1.2 — GJR-GARCH(1,1) with leverage term."""
    name = "GJR-GARCH(1,1)"
    _vol_kind = "GARCH"
    _o = 1


class EGARCH(_ArchWrapper):
    """
    B1.3 — EGARCH(1,1) of Nelson (1991).

    log(σ²_t) = ω + α·(|z_{t-1}| - E|z_{t-1}|) + γ·z_{t-1} + β·log(σ²_{t-1})

    Where z_t = ε_t/σ_t is the standardized residual. Asymmetry is captured
    by γ (leverage effect: γ < 0 means negative shocks raise vol more than
    positive ones). Sadik et al. (2018) used this as their primary
    benchmark; we include it so we can replicate their comparison directly.
    """
    name = "EGARCH(1,1)"
    _vol_kind = "EGARCH"
    _o = 1


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM NA-GARCH FAMILY
# ─────────────────────────────────────────────────────────────────────────────

class _NAGarchBase(ModelBase):
    """
    Shared machinery for the NA-GARCH-style models. Subclasses override:
        _scaling_factor(theta_news, t_lag)   — vector of shape (T,) of f_t
        _theta_news_dim                      — number of news params
        _theta_news_init / _theta_news_bounds
        _theta_news_names
    where t_lag is the lagged set of regressors used in the variance equation.
    """
    _theta_news_dim = 0

    def __init__(self, returns, dist="studentst"):
        super().__init__(returns, dist=dist)

    # ── Subclasses must implement these three ──────────────────────────────
    def _theta_news_init(self):
        raise NotImplementedError

    def _theta_news_bounds(self):
        raise NotImplementedError

    def _theta_news_names(self):
        raise NotImplementedError

    def _scaling_factor(self, theta_news):
        """Return the (T,) vector of scaling-factor values evaluated at theta_news."""
        raise NotImplementedError

    # ── Variance recursion ─────────────────────────────────────────────────
    def _conditional_variance(self, omega, alpha, beta, scaling):
        """
        Compute σ²_t for t=0..T-1 given ω, α, β and a precomputed scaling
        vector (length T). σ²_0 is initialized to the unconditional GARCH
        variance.

        Delegates the inner recursion to the numba-jitted free function
        `_variance_recursion` for ~15–30× speedup on the hot path.
        """
        # Initial variance = unconditional GARCH variance (scalar precompute)
        denom = 1.0 - alpha - beta
        if denom > 0:
            init_var = omega / max(denom, 1e-8)
        else:
            init_var = float(np.var(self.returns))
        return _variance_recursion(
            self.returns, scaling, float(omega), float(alpha), float(beta), float(init_var)
        )

    # ── Negative log-likelihood ────────────────────────────────────────────
    def _negloglik(self, theta):
        """theta packs: log_omega, raw_alpha, raw_beta, [theta_news...], [log_nu_minus_4]."""
        # GARCH params (constrained via transforms)
        log_omega = theta[0]
        raw_a, raw_b = theta[1], theta[2]
        omega = float(np.exp(log_omega))
        # α, β ≥ 0 and α + β < 1 via softmax-style transform:
        #   sum   = sigmoid(raw_a)·0.999  (so sum ∈ (0, 0.999))
        #   share = sigmoid(raw_b)        (so β/sum ∈ (0,1), β = share·sum)
        sigm_a = 1.0 / (1.0 + np.exp(-raw_a))
        sigm_b = 1.0 / (1.0 + np.exp(-raw_b))
        sum_ab = 0.999 * sigm_a
        beta   = sigm_b * sum_ab
        alpha  = sum_ab - beta

        # News block
        theta_news = theta[3 : 3 + self._theta_news_dim]
        scaling = self._scaling_factor(theta_news)
        if not np.all(np.isfinite(scaling)) or np.any(scaling < 0):
            return 1e10

        # Distribution params
        if self.dist == "studentst":
            log_nu_m4 = theta[-1]
            nu = 4.0 + float(np.exp(log_nu_m4))   # ν ≥ 4
        else:
            nu = None

        # Variance recursion
        try:
            sigma_sq = self._conditional_variance(omega, alpha, beta, scaling)
        except Exception:
            return 1e10
        if not np.all(np.isfinite(sigma_sq)) or np.any(sigma_sq <= 0):
            return 1e10

        return _negloglik(self.returns, sigma_sq, self.dist, nu=nu)

    # ── Convert between unconstrained θ and natural params ────────────────
    def _unpack(self, theta):
        log_omega = theta[0]
        raw_a, raw_b = theta[1], theta[2]
        omega = float(np.exp(log_omega))
        sigm_a = 1.0 / (1.0 + np.exp(-raw_a))
        sigm_b = 1.0 / (1.0 + np.exp(-raw_b))
        sum_ab = 0.999 * sigm_a
        beta   = sigm_b * sum_ab
        alpha  = sum_ab - beta
        theta_news = theta[3 : 3 + self._theta_news_dim]
        params = {"omega": omega, "alpha": alpha, "beta": beta}
        params.update(self._unpack_news(theta_news))
        if self.dist == "studentst":
            params["nu"] = 4.0 + float(np.exp(theta[-1]))
        return params

    def _unpack_news(self, theta_news):
        raise NotImplementedError

    # ── Initial guess (in unconstrained space) ─────────────────────────────
    def _theta_init(self):
        # Sensible GARCH starting values: ω = small, α = 0.05, β = 0.9
        log_omega0 = np.log(np.var(self.returns) * 0.05)
        # α + β = 0.95 → sum_ab = 0.95 → sigm_a such that 0.999·sigm = 0.95
        sigm_a0 = 0.95 / 0.999
        raw_a0  = np.log(sigm_a0 / (1.0 - sigm_a0))
        # β/sum = 0.9/0.95 ≈ 0.947
        sigm_b0 = 0.947
        raw_b0  = np.log(sigm_b0 / (1.0 - sigm_b0))
        theta = [log_omega0, raw_a0, raw_b0]
        theta.extend(self._theta_news_init())
        if self.dist == "studentst":
            # ν = 8 → log(ν-4) = log(4)
            theta.append(np.log(4.0))
        return np.array(theta)

    def _theta_bounds(self):
        b = [(-30.0, 5.0), (-10.0, 10.0), (-10.0, 10.0)]   # log_omega, raw_a, raw_b
        b.extend(self._theta_news_bounds())
        if self.dist == "studentst":
            b.append((-5.0, 5.0))   # log(ν-4)
        return b

    # ── Forecasting ───────────────────────────────────────────────────────
    def _scaling_factor_forecast(self, theta_news):
        """
        Subclasses override to return the SCALAR scaling factor at the
        forecast horizon (using the model's stored last unlagged values).
        For h>1 we hold this constant — exogenous regressors don't move
        without new info.
        """
        raise NotImplementedError

    def forecast_variance(self, h=1):
        """
        h-step-ahead variance forecasts σ²_{T+1}, ..., σ²_{T+h}.

        Multi-step propagation uses E[ε²_{t+k}] = σ²_{t+k} for k > 0,
        which gives σ²_{t+k} = scaling_factor · (ω + (α+β)·σ²_{t+k-1})
        for k > 1. Exogenous regressors held constant at their last
        observed values.

        Returns: numpy array of shape (h,).
        """
        if not self.fitted:
            raise RuntimeError("call .fit() first")
        omega = self.params["omega"]
        alpha = self.params["alpha"]
        beta  = self.params["beta"]

        scale = self._scaling_factor_forecast(
            self._theta_opt[3 : 3 + self._theta_news_dim]
        )
        if not np.isfinite(scale) or scale < 0:
            scale = 1.0     # defensive fallback

        eps_T   = self.returns[-1]
        sig_T   = self._sigma_sq_in[-1]

        forecasts = np.empty(h)
        for k in range(h):
            if k == 0:
                garch = omega + alpha * eps_T * eps_T + beta * sig_T
            else:
                garch = omega + (alpha + beta) * forecasts[k - 1]
            forecasts[k] = scale * garch
        return forecasts

    # ── fit ───────────────────────────────────────────────────────────────
    def fit(self, theta_init=None, n_restarts=3, warm_start=None):
        """
        L-BFGS-B optimization. Tries multiple random restarts to mitigate
        local minima; returns the best one.

        warm_start: optional dict with 'theta' key — a θ vector from a
        nested model to use as a strong starting point (in addition to the
        default and random restarts). Used by NAGarchHt to warm-start from
        NAGarchAsym's optimum (since B3.1 reduces to B2.2 at δ=0).
        """
        candidates = []
        # Default starting point
        candidates.append(self._theta_init() if theta_init is None
                          else np.asarray(theta_init))
        # Warm-start from nested model
        if warm_start is not None and "theta" in warm_start:
            candidates.append(np.asarray(warm_start["theta"]))
        # Random perturbations of the default
        for r in range(n_restarts):
            rng = np.random.default_rng((r + 1) * 17)
            candidates.append(candidates[0] + rng.normal(scale=0.5, size=candidates[0].shape))

        best_nll = np.inf
        best_theta = None
        for x0 in candidates:
            res = minimize(
                self._negloglik,
                x0=x0,
                method="L-BFGS-B",
                bounds=self._theta_bounds(),
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if res.fun < best_nll and np.isfinite(res.fun):
                best_nll = res.fun
                best_theta = res.x
        if best_theta is None:
            raise RuntimeError(f"{self.name} optimization failed to converge.")

        self.params = self._unpack(best_theta)
        self.loglik = -float(best_nll)
        self.aic = _aic(self.loglik, n_params=len(best_theta))
        # Cache fitted variance for diagnostics
        scaling = self._scaling_factor(best_theta[3 : 3 + self._theta_news_dim])
        self._sigma_sq_in = self._conditional_variance(
            self.params["omega"], self.params["alpha"], self.params["beta"], scaling
        )
        self._theta_opt = best_theta
        self.fitted = True
        return self


# ─── B2.1 NA-GARCH-net (single net stance S_t) ────────────────────────────
class NAGarchNet(_NAGarchBase):
    """
    σ²_t = [a + 0.5·b·tanh(κ·S_{t-1}/2)] · (ω + α·ε²_{t-1} + β·σ²_{t-1})

    Sadik replication baseline using NET stance S_t = P_t + N_t (so
    S_t ∈ [-1, 1]).
    """
    name = "NA-GARCH-net (B2.1)"
    _theta_news_dim = 3   # log(a), log(b), log(κ); enforce 0.5 ≤ a+b ≤ 2 via penalty

    def __init__(self, returns, S, dist="studentst"):
        super().__init__(returns, dist=dist)
        S_arr = np.asarray(S.values if hasattr(S, "values") else S, dtype=float)
        self.S = np.r_[0.0, S_arr[:-1]]    # lagged: self.S[t] = S[t-1]
        self.S_last = float(S_arr[-1])     # unlagged last value for forecasting
        _validate_finite(self.S, "S")

    def _theta_news_init(self):
        return [np.log(0.8), np.log(0.8), np.log(4.0)]   # a, b ≈ Sadik defaults

    def _theta_news_bounds(self):
        return [(-5.0, 2.0)] * 3

    def _theta_news_names(self):
        return ["a", "b", "kappa"]

    def _scaling_factor(self, theta_news):
        a = float(np.exp(theta_news[0]))
        b = float(np.exp(theta_news[1]))
        kappa = float(np.exp(theta_news[2]))
        # 0.5 ≤ a+b ≤ 2 enforced by clipping to bounds during return; if violated
        # we add a soft penalty by clamping (the optimizer's bounds usually keep
        # it in range)
        ab = a + b
        if ab < 0.5 or ab > 2.0:
            # signal infeasible to the caller via NaN scaling (handled in negloglik)
            return np.full(self.nobs, np.nan)
        return a + 0.5 * b * np.tanh(0.5 * kappa * self.S)

    def _unpack_news(self, theta_news):
        return {
            "a":     float(np.exp(theta_news[0])),
            "b":     float(np.exp(theta_news[1])),
            "kappa": float(np.exp(theta_news[2])),
        }

    def _scaling_factor_forecast(self, theta_news):
        a = float(np.exp(theta_news[0]))
        b = float(np.exp(theta_news[1]))
        kappa = float(np.exp(theta_news[2]))
        return a + 0.5 * b * np.tanh(0.5 * kappa * self.S_last)


# ─── B2.2 NA-GARCH-asym (P_t and N_t separately) ──────────────────────────
class NAGarchAsym(_NAGarchBase):
    """
    σ²_t = [a + 0.5·b·(tanh(κ·P_{t-1}/2) − tanh(γ·N_{t-1}/2))] · GARCH_term

    Asymmetric NA-GARCH with separate dovish (P_t ∈ [0,1]) and hawkish
    (N_t ∈ [-1,0]) terms. Smales-style asymmetry.
    """
    name = "NA-GARCH-asym (B2.2)"
    _theta_news_dim = 4   # log(a), log(b), log(κ), log(γ)

    def __init__(self, returns, P, N, dist="studentst"):
        super().__init__(returns, dist=dist)
        P_arr = np.asarray(P.values if hasattr(P, "values") else P, dtype=float)
        N_arr = np.asarray(N.values if hasattr(N, "values") else N, dtype=float)
        self.P = np.r_[0.0, P_arr[:-1]]
        self.N = np.r_[0.0, N_arr[:-1]]
        self.P_last = float(P_arr[-1])
        self.N_last = float(N_arr[-1])
        _validate_finite(self.P, "P")
        _validate_finite(self.N, "N")

    def _theta_news_init(self):
        return [np.log(0.8), np.log(0.8), np.log(4.0), np.log(4.0)]

    def _theta_news_bounds(self):
        return [(-5.0, 2.0)] * 4

    def _theta_news_names(self):
        return ["a", "b", "kappa", "gamma"]

    def _scaling_factor(self, theta_news):
        a, b = np.exp(theta_news[0]), np.exp(theta_news[1])
        kappa, gamma = np.exp(theta_news[2]), np.exp(theta_news[3])
        if a + b < 0.5 or a + b > 2.0:
            return np.full(self.nobs, np.nan)
        return a + 0.5 * b * (np.tanh(0.5 * kappa * self.P)
                              - np.tanh(0.5 * gamma * self.N))

    def _unpack_news(self, theta_news):
        return {
            "a":     float(np.exp(theta_news[0])),
            "b":     float(np.exp(theta_news[1])),
            "kappa": float(np.exp(theta_news[2])),
            "gamma": float(np.exp(theta_news[3])),
        }

    def _scaling_factor_forecast(self, theta_news):
        a = float(np.exp(theta_news[0]))
        b = float(np.exp(theta_news[1]))
        kappa = float(np.exp(theta_news[2]))
        gamma = float(np.exp(theta_news[3]))
        return a + 0.5 * b * (np.tanh(0.5 * kappa * self.P_last)
                              - np.tanh(0.5 * gamma * self.N_last))


# ─── B3.1 NA-GARCH with binary H_t (MAIN MODEL) ──────────────────────────
class NAGarchHt(_NAGarchBase):
    """
    σ²_t = [a + 0.5·(b + δ·H_{t-1})·(tanh(κ·P_{t-1}/2) − tanh(γ·N_{t-1}/2))]
           · (ω + α·ε²_{t-1} + β·σ²_{t-1})

    where H_{t-1} = 1{roll_vol_20d_{t-1} > c·long_run_mean_20d}.

    `c` is FIXED at construction (grid-searched outside, in
    estimate_in_sample.py). At fixed c, the H series is a precomputed 0/1
    vector and the likelihood is smooth in (ω, α, β, a, b, κ, γ, δ, ν) so
    L-BFGS-B handles inner MLE fine.

    Constraint b + δ ≥ 0 (so news scaling stays non-negative when H=1) is
    enforced via a barrier added to the negative log-likelihood.
    """
    name = "NA-GARCH-Ht (B3.1)"
    _theta_news_dim = 5   # log(a), log(b), log(κ), log(γ), δ_raw

    def __init__(self, returns, P, N, roll_vol, long_run_mean, c,
                 dist="studentst"):
        super().__init__(returns, dist=dist)
        P_arr = np.asarray(P.values if hasattr(P, "values") else P, dtype=float)
        N_arr = np.asarray(N.values if hasattr(N, "values") else N, dtype=float)
        rv_arr = np.asarray(roll_vol.values if hasattr(roll_vol, "values") else roll_vol,
                            dtype=float)
        self.P = np.r_[0.0, P_arr[:-1]]                          # lagged
        self.N = np.r_[0.0, N_arr[:-1]]
        self.P_last = float(P_arr[-1])                           # unlagged last
        self.N_last = float(N_arr[-1])
        rv_lag = np.r_[np.nan, rv_arr[:-1]]                      # H uses rv_{t-1}
        self.long_run_mean = float(long_run_mean)
        self.c = float(c)
        self.H = np.where(np.isfinite(rv_lag),
                          (rv_lag > self.c * self.long_run_mean).astype(float), 0.0)
        # H_last is the H that would apply on the FIRST forecast day (t=T+1):
        # uses rv_T (the last in-sample rolling vol).
        last_rv = rv_arr[-1] if np.isfinite(rv_arr[-1]) else 0.0
        self.H_last = float(last_rv > self.c * self.long_run_mean)
        self.last_rv = float(last_rv)
        _validate_finite(self.P, "P")
        _validate_finite(self.N, "N")

    def _theta_news_init(self):
        # Start at δ = 0 (no regime effect), other news-params at moderate values
        return [np.log(0.8), np.log(0.8), np.log(4.0), np.log(4.0), 0.0]

    def _theta_news_bounds(self):
        return [(-5.0, 2.0)] * 4 + [(-5.0, 5.0)]   # δ unconstrained-ish

    def _theta_news_names(self):
        return ["a", "b", "kappa", "gamma", "delta"]

    def _scaling_factor(self, theta_news):
        a, b = np.exp(theta_news[0]), np.exp(theta_news[1])
        kappa, gamma = np.exp(theta_news[2]), np.exp(theta_news[3])
        delta = float(theta_news[4])
        if a + b < 0.5 or a + b > 2.0:
            return np.full(self.nobs, np.nan)
        # Barrier: b + δ must be ≥ 0 (otherwise scaling can go negative on H=1 days)
        if b + delta < 0:
            return np.full(self.nobs, np.nan)
        b_eff = b + delta * self.H        # vector of length T
        f = a + 0.5 * b_eff * (np.tanh(0.5 * kappa * self.P)
                               - np.tanh(0.5 * gamma * self.N))
        return f

    def _unpack_news(self, theta_news):
        return {
            "a":     float(np.exp(theta_news[0])),
            "b":     float(np.exp(theta_news[1])),
            "kappa": float(np.exp(theta_news[2])),
            "gamma": float(np.exp(theta_news[3])),
            "delta": float(theta_news[4]),
            "c":     float(self.c),
        }

    def _scaling_factor_forecast(self, theta_news):
        a = float(np.exp(theta_news[0]))
        b = float(np.exp(theta_news[1]))
        kappa = float(np.exp(theta_news[2]))
        gamma = float(np.exp(theta_news[3]))
        delta = float(theta_news[4])
        b_eff = b + delta * self.H_last
        return a + 0.5 * b_eff * (np.tanh(0.5 * kappa * self.P_last)
                                  - np.tanh(0.5 * gamma * self.N_last))


# ─── B5.6 NA-GARCH with continuous "hockey stick" H_t ─────────────────────
class NAGarchHockey(NAGarchHt):
    """
    Same as NAGarchHt but with H replaced by a continuous "hockey-stick":
        H_hockey_{t-1} = max(0, roll_vol_{t-1} / LRM − c)

    Below the threshold, H = 0 (regime off, identical to binary). Above it,
    H grows linearly with how far rolling vol exceeds c × LRM.
    """
    name = "NA-GARCH-Hockey (B5.6)"

    def __init__(self, returns, P, N, roll_vol, long_run_mean, c,
                 dist="studentst"):
        super().__init__(returns, P, N, roll_vol, long_run_mean, c, dist=dist)
        rv_arr = np.asarray(roll_vol.values if hasattr(roll_vol, "values") else roll_vol,
                            dtype=float)
        rv_lag = np.r_[np.nan, rv_arr[:-1]]
        ratio = np.where(np.isfinite(rv_lag), rv_lag / self.long_run_mean, 0.0)
        self.H = np.maximum(0.0, ratio - self.c)
        # Hockey-stick H_last for σ²_{T+1}
        last_rv = rv_arr[-1] if np.isfinite(rv_arr[-1]) else 0.0
        self.H_last = float(max(0.0, last_rv / self.long_run_mean - self.c))
