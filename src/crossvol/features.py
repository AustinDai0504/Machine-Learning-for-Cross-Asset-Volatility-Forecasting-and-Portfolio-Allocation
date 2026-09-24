"""Features available at the origin close; labels begin after execution delay."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_variance(returns: pd.Series, horizon: int, execution_lag: int, annualization=252):
    """For origin t, mean squared returns t+lag+1 through t+lag+horizon."""
    if horizon < 1 or execution_lag < 0:
        raise ValueError("Positive horizon and nonnegative execution lag required")
    future = pd.concat(
        [
            returns.shift(-step).pow(2)
            for step in range(execution_lag + 1, execution_lag + horizon + 1)
        ],
        axis=1,
    )
    return future.sum(axis=1, min_count=horizon) * annualization / horizon


def build_features(prices: pd.DataFrame, config: dict):
    cfg = config["research"]
    annual = cfg["annualization"]
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    returns = np.log(wide).diff()
    rv21 = returns.pow(2).rolling(21).mean() * annual
    ret5 = returns.rolling(5).sum()
    panels = {}
    groups = None
    for ticker in config["data"]["tickers"]:
        r = returns[ticker]
        x = pd.DataFrame(index=wide.index)
        for window in (5, 21, 63):
            x[f"log_rv{window}"] = np.log((r.pow(2).rolling(window).mean() * annual).clip(1e-8))
        ewma = (
            r.pow(2).ewm(alpha=1 - cfg["ewma_lambda"], adjust=False, min_periods=63).mean() * annual
        )
        x["log_ewma"] = np.log(ewma.clip(1e-8))
        risk_cols = list(x.columns)
        for window in (1, 5, 21, 63):
            x[f"return_{window}"] = r.rolling(window).sum()
        x["abs_return_1"] = r.abs()
        x["log_downside21"] = np.log(
            (r.clip(upper=0).pow(2).rolling(21).mean() * annual).clip(1e-8)
        )
        x["rv5_rv63_ratio"] = np.exp(x.log_rv5 - x.log_rv63)
        x["vol_of_vol21"] = np.sqrt(rv21[ticker]).rolling(21).std()
        x["drawdown63"] = wide[ticker] / wide[ticker].rolling(63).max() - 1
        asset = prices.loc[prices.ticker == ticker].set_index("date").reindex(wide.index)
        range_var = np.log(asset.high / asset.low).pow(2) / (4 * np.log(2))
        x["log_parkinson21"] = np.log((range_var.rolling(21).mean() * annual).clip(1e-8))
        no_cross_cols = list(x.columns)
        for other in config["data"]["tickers"]:
            # Keep one schema across assets, but zero self channels: duplicated own
            # predictors would change Ridge's effective regularization in ablation.
            x[f"cross_{other}_log_rv21"] = (
                0.0 if other == ticker else np.log(rv21[other].clip(1e-8))
            )
            x[f"cross_{other}_return5"] = 0.0 if other == ticker else ret5[other]
            x[f"cross_{other}_corr63"] = (
                0.0 if other == ticker else r.rolling(63).corr(returns[other])
            )
        full_cols = list(x.columns)
        groups = {"risk_only": risk_cols, "no_cross": no_cross_cols, "full": full_cols}
        x["hv_var"] = rv21[ticker]
        x["ewma_var"] = ewma
        x["target_var"] = forward_variance(r, cfg["horizon"], cfg["execution_lag"], annual)
        x["label_end"] = pd.Series(wide.index, index=wide.index).shift(
            -(cfg["horizon"] + cfg["execution_lag"])
        )
        panels[ticker] = x.replace([np.inf, -np.inf], np.nan).dropna(subset=full_cols)
    return panels, groups
