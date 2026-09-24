"""Forecast evaluation and paired, date-synchronized circular block inference.

Bootstrap intervals are conditional on the fitted walk-forward predictions and
the chosen universe; they do not incorporate model-selection uncertainty.
Forecast inference compares daily equal-asset losses. All assets and methods
share each sampled date, preserving contemporaneous cross-asset dependence.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "model",
    "feature_set",
    "ticker",
    "qlike",
    "vol_rmse",
    "vol_mae",
    "calibration",
    "n_obs",
    "n_dates",
]
COMPARISON_MODELS = ("Ridge", "LightGBM")
BASELINE = "EWMA"


def qlike(target_var: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
    """QLIKE variance loss, with zero denoting a perfect forecast.

    Both arguments must be finite and strictly positive. Rejecting invalid
    observations prevents missing losses from silently changing the sample.
    """
    target = np.asarray(target_var, dtype=float)
    prediction = np.asarray(pred_var, dtype=float)
    if not np.all(np.isfinite(target)) or not np.all(target > 0):
        raise ValueError("target_var must be finite and strictly positive")
    if not np.all(np.isfinite(prediction)) or not np.all(prediction > 0):
        raise ValueError("pred_var must be finite and strictly positive")
    ratio = target / prediction
    # log1p improves numerical accuracy near a perfect forecast.
    distance = ratio - 1.0
    return distance - np.log1p(distance)


def _check_forecasts(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "model", "feature_set", "target_var", "pred_var"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    keys = ["date", "ticker", "model", "feature_set"]
    if frame[keys].isna().any().any():
        raise ValueError("Forecast keys cannot contain missing values")
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate date/ticker/model/feature_set forecasts")
    frame["_qlike"] = qlike(frame["target_var"], frame["pred_var"])
    return frame


def forecast_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return per-asset and pooled scores for every model/feature set.

    Volatility RMSE and MAE use annualized volatility in decimal units.
    Calibration is mean realized variance / mean predicted variance, ideal 1.
    Pooled metrics average assets within each date, then dates equally. Thus
    more available assets on a date cannot give that date greater weight.
    ``ticker == 'Pooled'`` identifies the aggregate rows.
    """
    if predictions.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    frame = _check_forecasts(predictions)
    error = np.sqrt(frame["pred_var"]) - np.sqrt(frame["target_var"])
    frame["_squared_error"] = error**2
    frame["_absolute_error"] = error.abs()
    aggregate_columns = ["_qlike", "_squared_error", "_absolute_error", "target_var", "pred_var"]
    rows = []
    for (model, feature_set), sample in frame.groupby(["model", "feature_set"], sort=True):
        populations = list(sample.groupby("ticker", sort=True)) + [("Pooled", sample)]
        for ticker, population in populations:
            averages = population.groupby("date")[aggregate_columns].mean().mean()
            rows.append(
                {
                    "model": model,
                    "feature_set": feature_set,
                    "ticker": ticker,
                    "qlike": averages["_qlike"],
                    "vol_rmse": np.sqrt(averages["_squared_error"]),
                    "vol_mae": averages["_absolute_error"],
                    "calibration": averages["target_var"] / averages["pred_var"],
                    "n_obs": len(population),
                    "n_dates": population["date"].nunique(),
                }
            )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm family-wise adjustment, preserving the input comparison order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("p_values must be a finite one-dimensional array")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    result = np.empty_like(values)
    result[order] = adjusted
    return result


def _circular_batches(
    n_dates: int,
    block_length: int,
    n_bootstrap: int,
    rng: np.random.Generator,
    batch_size: int = 128,
) -> Iterator[np.ndarray]:
    """Yield complete sampled date paths; blocks can wrap around sample end."""
    n_blocks = (n_dates + block_length - 1) // block_length
    offsets = np.arange(block_length)
    for first in range(0, n_bootstrap, batch_size):
        count = min(batch_size, n_bootstrap - first)
        starts = rng.integers(0, n_dates, size=(count, n_blocks))
        indices = (starts[:, :, None] + offsets) % n_dates
        yield indices.reshape(count, -1)[:, :n_dates]


def _daily_loss_differences(predictions: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.loc[
        ((predictions["model"] == BASELINE) & (predictions["feature_set"] == "baseline"))
        | (predictions["model"].isin(COMPARISON_MODELS) & (predictions["feature_set"] == "full"))
    ]
    if selected.empty:
        raise ValueError("Inference requires EWMA baseline and full Ridge/LightGBM predictions")
    selected = _check_forecasts(selected)
    models = [BASELINE, *COMPARISON_MODELS]
    dates = sorted(selected["date"].unique())
    tickers = sorted(selected["ticker"].unique())
    complete_index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    losses = selected.pivot(index=["date", "ticker"], columns="model", values="_qlike")
    losses = losses.reindex(index=complete_index, columns=models)
    if losses.isna().any().any():
        raise ValueError(
            "Inference requires identical complete dates/assets across all three models"
        )
    targets = selected.pivot(index=["date", "ticker"], columns="model", values="target_var")[models]
    if not np.allclose(
        targets.to_numpy(), targets[BASELINE].to_numpy()[:, None], rtol=1e-12, atol=0
    ):
        raise ValueError("Models must share identical realized targets for paired inference")
    differences = losses[list(COMPARISON_MODELS)].subtract(losses[BASELINE], axis=0)
    return differences.groupby(level="date", sort=True).mean()


def _annualized_sharpe(returns: np.ndarray, annualization: int, axis: int) -> np.ndarray:
    mean = np.mean(returns, axis=axis)
    volatility = np.std(returns, axis=axis, ddof=1)
    return np.divide(
        np.sqrt(annualization) * mean,
        volatility,
        out=np.full_like(mean, np.nan, dtype=float),
        where=volatility > 1e-14,
    )


def _paired_returns(daily: pd.DataFrame, base_cost: float) -> pd.DataFrame:
    required = {"date", "model", "cost_bps", "net_return"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Missing portfolio columns: {sorted(missing)}")
    models = [BASELINE, *COMPARISON_MODELS]
    sample = daily.loc[
        daily["model"].isin(models) & np.isclose(daily["cost_bps"], base_cost)
    ].copy()
    # The all-cash NAV anchor is a timestamp, not an earned return interval.
    # Exclude it consistently with portfolio metrics; synthetic standalone
    # return inputs without the explicit marker retain all their observations.
    if "is_initial_anchor" in sample:
        if not sample["is_initial_anchor"].isin([True, False]).all():
            raise ValueError("is_initial_anchor must contain nonmissing Boolean flags")
        sample = sample.loc[~sample["is_initial_anchor"].astype(bool)].copy()
    sample["date"] = pd.to_datetime(sample["date"])
    if sample.empty or sample[["date", "model"]].isna().any().any():
        raise ValueError("Portfolio inference requires nonempty dated returns at the base cost")
    if sample.duplicated(["date", "model"]).any():
        raise ValueError("Duplicate portfolio model/date returns at the base cost")
    returns = sample.pivot(index="date", columns="model", values="net_return").reindex(
        columns=models
    )
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError(
            "Portfolio inference requires identical finite dates across all three models"
        )
    return returns.sort_index()


def run_inference(
    predictions: pd.DataFrame,
    daily: pd.DataFrame,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """Compare full Ridge and LightGBM against EWMA on a paired sample.

    Forecast: mean QLIKE difference (negative is better), percentile 95% CI,
    two-sided centered-null bootstrap p-value with a finite simulation correction,
    and Holm adjustment across the two candidate models separately per block size.

    Strategy: net base-cost annualized Sharpe difference (positive is better),
    percentile 95% CI only. No naive bootstrap p-value is reported for Sharpe.
    Initial all-cash anchors are excluded from returns. Resampling already
    realized net returns does not rebuild trades at joins.
    """
    inference = config.get("inference", {})
    n_bootstrap = int(inference.get("n_bootstrap", 2000))
    block_lengths = list(inference.get("block_lengths", [10, 20, 60]))
    seed = int(config.get("research", {}).get("seed", 42))
    annualization = int(config.get("research", {}).get("annualization", 252))
    base_cost = float(config.get("portfolio", {}).get("base_cost_bps", 5))
    if n_bootstrap < 2 or not block_lengths or annualization <= 0:
        raise ValueError(
            "Need at least two bootstrap replicates, positive annualization, and block lengths"
        )
    if any(int(block) != block or block < 1 for block in block_lengths):
        raise ValueError("block_lengths must contain positive integers")
    if len(set(block_lengths)) != len(block_lengths):
        raise ValueError("block_lengths must be unique")

    differences = _daily_loss_differences(predictions)
    returns = _paired_returns(daily, base_cost)
    if len(differences) < 2 or len(returns) < 2:
        raise ValueError("Inference requires at least two synchronized dates")
    loss_values = differences.to_numpy()
    return_values = returns.to_numpy()
    observed_losses = loss_values.mean(axis=0)
    observed_sharpes = _annualized_sharpe(return_values, annualization, axis=0)
    if not np.isfinite(observed_sharpes).all():
        raise ValueError("Sharpe inference is undefined for a strategy with zero return variance")
    observed_sharpe_differences = observed_sharpes[1:] - observed_sharpes[0]
    forecast_rows, strategy_rows = [], []

    for block in block_lengths:
        block = int(block)
        # Independent stable streams make output invariant to block-size ordering.
        forecast_rng = np.random.default_rng(np.random.SeedSequence([seed, block, 0]))
        strategy_rng = np.random.default_rng(np.random.SeedSequence([seed, block, 1]))
        sampled_losses = np.concatenate(
            [
                loss_values[indices].mean(axis=1)
                for indices in _circular_batches(len(differences), block, n_bootstrap, forecast_rng)
            ]
        )
        loss_intervals = np.quantile(sampled_losses, [0.025, 0.975], axis=0)
        null_statistics = sampled_losses - observed_losses
        p_values = (1 + (np.abs(null_statistics) >= np.abs(observed_losses)).sum(axis=0)) / (
            n_bootstrap + 1
        )
        p_holm = holm_adjust(p_values)

        sampled_sharpes = []
        for indices in _circular_batches(len(returns), block, n_bootstrap, strategy_rng):
            sharpes = _annualized_sharpe(return_values[indices], annualization, axis=1)
            sampled_sharpes.append(sharpes[:, 1:] - sharpes[:, [0]])
        sampled_sharpes = np.concatenate(sampled_sharpes)
        valid_replicates = np.isfinite(sampled_sharpes).sum(axis=0)
        if np.any(valid_replicates < 0.9 * n_bootstrap):
            raise ValueError("Too many degenerate bootstrap paths for reliable Sharpe intervals")
        sharpe_intervals = np.nanquantile(sampled_sharpes, [0.025, 0.975], axis=0)

        for index, model in enumerate(COMPARISON_MODELS):
            metadata = {
                "model": model,
                "baseline": BASELINE,
                "block_length": block,
                "n_bootstrap": n_bootstrap,
            }
            forecast_rows.append(
                {
                    **metadata,
                    "metric": "qlike_difference",
                    "n_dates": len(differences),
                    "estimate": observed_losses[index],
                    "ci_low": loss_intervals[0, index],
                    "ci_high": loss_intervals[1, index],
                    "p_value": p_values[index],
                    "p_holm": p_holm[index],
                }
            )
            strategy_rows.append(
                {
                    **metadata,
                    "metric": "sharpe_difference",
                    "cost_bps": base_cost,
                    "n_dates": len(returns),
                    "estimate": observed_sharpe_differences[index],
                    "ci_low": sharpe_intervals[0, index],
                    "ci_high": sharpe_intervals[1, index],
                    "valid_replicates": valid_replicates[index],
                }
            )
    return {
        "forecast_inference": pd.DataFrame(forecast_rows),
        "strategy_inference": pd.DataFrame(strategy_rows),
    }
