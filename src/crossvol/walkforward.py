"""Purged, nested, annual expanding-window volatility forecasting.

An observation's timestamp is its information/forecast origin.  ``label_end``
is the last close needed to observe its target, so filtering origin timestamps
alone would leak information across validation and test boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


VAR_FLOOR = 1e-8
VAR_CAP = 4.0
FEATURE_SETS = ("risk_only", "no_cross", "full")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    next_boundary: pd.Timestamp


@dataclass
class _Fitted:
    estimator: Any
    smearing: float

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        # Clip in log space first to avoid overflow before applying common bounds.
        predicted_log = np.asarray(self.estimator.predict(x), dtype=float)
        return np.clip(
            np.exp(np.clip(predicted_log, -50, 50)) * self.smearing,
            VAR_FLOOR,
            VAR_CAP,
        )


def _qlike(target: np.ndarray, prediction: np.ndarray) -> float:
    ratio = np.asarray(target, dtype=float) / np.asarray(prediction, dtype=float)
    return float(np.mean(ratio - np.log(ratio) - 1))


def _fit(
    model: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    columns: list[str],
    seed: int,
) -> _Fitted:
    if model == "Ridge":
        estimator = make_pipeline(StandardScaler(), Ridge(**params))
    elif model == "LightGBM":
        # Lazy loading gives a useful error on systems missing the OpenMP runtime.
        try:
            from lightgbm import LGBMRegressor
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "LightGBM could not load. Install the project dependencies and the "
                "OpenMP runtime (macOS: brew install libomp)."
            ) from exc
        estimator = LGBMRegressor(
            **params,
            objective="regression",
            learning_rate=0.05,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            importance_type="gain",
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown fitted model: {model}")
    x = train.loc[:, columns]
    log_y = np.log(train["target_var"].clip(lower=VAR_FLOOR).to_numpy())
    estimator.fit(x, log_y)
    residual = log_y - estimator.predict(x)
    # Duan's retransformation factor uses only this fit's training residuals.
    # It estimates arithmetic conditional variance from log-variance predictions.
    smearing = float(np.mean(np.exp(np.clip(residual, -50, 50))))
    return _Fitted(estimator, smearing)


def _inner_splits(
    outer_train: pd.DataFrame,
    outer_boundary: pd.Timestamp,
    validation_size: int,
    validation_splits: int,
    min_train: int,
) -> list[_Split]:
    """Build full-size validation blocks backward, purging label overlap.

    Selecting the preceding block only after filtering on its label endpoint
    leaves a gap between validation blocks.  Each block therefore retains the
    requested number of observations without sharing future target returns.
    """
    splits = []
    boundary = pd.Timestamp(outer_boundary)
    for _ in range(validation_splits):
        eligible = outer_train.loc[
            (outer_train.index < boundary) & (outer_train["label_end"] < boundary)
        ]
        if len(eligible) < validation_size:
            raise ValueError("Insufficient history for the requested validation blocks")
        validation = eligible.iloc[-validation_size:]
        validation_start = validation.index[0]
        train = outer_train.loc[
            (outer_train.index < validation_start) & (outer_train["label_end"] < validation_start)
        ]
        if len(train) < min_train:
            raise ValueError(
                f"Only {len(train)} training origins before validation "
                f"{validation_start.date()}; min_train={min_train}. "
                "Use earlier data, a later oos_start, or smaller validation blocks."
            )
        splits.append(_Split(train, validation, boundary))
        boundary = validation_start
    return list(reversed(splits))


def _prepare_panels(
    panels: dict[str, pd.DataFrame], groups: dict[str, list[str]]
) -> dict[str, pd.DataFrame]:
    if not panels:
        raise ValueError("panels must contain at least one asset")
    missing_groups = set(FEATURE_SETS) - set(groups)
    if missing_groups:
        raise ValueError(f"Missing feature sets: {sorted(missing_groups)}")
    columns = list(dict.fromkeys(c for group in FEATURE_SETS for c in groups[group]))
    prohibited = {"target_var", "label_end", "date", "ticker"} & set(columns)
    if prohibited:
        raise ValueError(f"Target/metadata columns cannot be predictors: {sorted(prohibited)}")
    for group in FEATURE_SETS:
        if not groups[group] or len(groups[group]) != len(set(groups[group])):
            raise ValueError(f"Feature set {group} must be nonempty without duplicates")
    required = columns + ["target_var", "label_end", "hv_var", "ewma_var"]
    prepared = {}
    for ticker, panel in panels.items():
        if not isinstance(panel.index, pd.DatetimeIndex):
            raise ValueError(f"{ticker}: panel must have a DatetimeIndex")
        if panel.index.has_duplicates:
            raise ValueError(f"{ticker}: duplicate forecast origin dates")
        missing = set(required) - set(panel.columns)
        if missing:
            raise ValueError(f"{ticker}: missing columns {sorted(missing)}")
        frame = panel.copy().sort_index()
        frame["label_end"] = pd.to_datetime(frame["label_end"])
        numeric = columns + ["target_var", "hv_var", "ewma_var"]
        valid = np.isfinite(frame[numeric].to_numpy(dtype=float)).all(axis=1)
        valid &= frame["label_end"].notna().to_numpy()
        valid &= (frame[["target_var", "hv_var", "ewma_var"]] > 0).all(axis=1).to_numpy()
        frame = frame.loc[valid]
        if frame.empty:
            raise ValueError(f"{ticker}: no complete, positive-variance observations")
        if (frame["label_end"] <= frame.index).any():
            raise ValueError(f"{ticker}: label_end must follow the forecast origin")
        prepared[ticker] = frame
    return prepared


def run_forecasts(
    panels: dict[str, pd.DataFrame],
    groups: dict[str, list[str]],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Fit all models and return predictions plus reproducible audit tables.

    All assets, model families, and feature ablations share exactly the same
    labeled out-of-sample origin dates.  Each model is refitted once per calendar
    year using only labels observable strictly before its first test origin.
    Hyperparameters minimize mean QLIKE across the inner validation blocks.
    Test labels never participate in candidate selection or retransformation.

    ``folds`` contains one row per fitted asset/model/feature-set/year; baseline
    forecasts require no fit. ``tuning`` contains every candidate/split, including
    endpoint audits. ``importance`` is training-derived and is not causal evidence.
    """
    settings = config["research"]
    validation_size = int(settings["validation_size"])
    n_splits = int(settings["validation_splits"])
    min_train = int(settings["min_train"])
    if min(validation_size, n_splits, min_train) < 1:
        raise ValueError("Validation sizes, split count, and min_train must be positive")
    seed = int(settings.get("seed", 42))
    candidates = {
        "Ridge": [{"alpha": float(alpha)} for alpha in settings["ridge_alphas"]],
        "LightGBM": [dict(params) for params in settings["lgbm_grid"]],
    }
    if any(not grid for grid in candidates.values()):
        raise ValueError("Each model must have at least one candidate")
    prepared = _prepare_panels(panels, groups)
    common = next(iter(prepared.values())).index
    for frame in prepared.values():
        common = common.intersection(frame.index)
    common = common[common >= pd.Timestamp(settings["oos_start"])].sort_values()
    if common.empty:
        raise ValueError("No common labeled observations on or after oos_start")

    predictions: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    tuning_records: list[dict[str, Any]] = []
    importance_records: list[dict[str, Any]] = []
    for year in sorted(common.year.unique()):
        test_dates = common[common.year == year]
        test_start = test_dates[0]
        LOGGER.info(
            "Forecast fold %s: %s through %s (%s assets, %s common origins)",
            year,
            test_start.date(),
            test_dates[-1].date(),
            len(prepared),
            len(test_dates),
        )
        for ticker, frame in prepared.items():
            outer_train = frame.loc[(frame.index < test_start) & (frame["label_end"] < test_start)]
            test = frame.loc[test_dates]
            splits = _inner_splits(outer_train, test_start, validation_size, n_splits, min_train)

            def add_predictions(model: str, feature_set: str, values: np.ndarray) -> None:
                predictions.append(
                    pd.DataFrame(
                        {
                            "date": test_dates,
                            "ticker": ticker,
                            "model": model,
                            "feature_set": feature_set,
                            "pred_var": np.clip(values, VAR_FLOOR, VAR_CAP),
                            "target_var": test["target_var"].to_numpy(),
                            "label_end": test["label_end"].to_numpy(),
                            "fold": int(year),
                        }
                    )
                )

            add_predictions("HV21", "baseline", test["hv_var"].to_numpy())
            add_predictions("EWMA", "baseline", test["ewma_var"].to_numpy())
            for model, grid in candidates.items():
                for feature_set in FEATURE_SETS:
                    columns = groups[feature_set]
                    candidate_scores = []
                    audit_rows = []
                    for candidate, params in enumerate(grid):
                        scores = []
                        for split_number, split in enumerate(splits, start=1):
                            fitted = _fit(model, params, split.train, columns, seed)
                            validation_pred = fitted.predict(split.validation[columns])
                            score = _qlike(
                                split.validation["target_var"].to_numpy(), validation_pred
                            )
                            scores.append(score)
                            audit_rows.append(
                                {
                                    "ticker": ticker,
                                    "model": model,
                                    "feature_set": feature_set,
                                    "fold": int(year),
                                    "candidate": candidate,
                                    "params": json.dumps(params, sort_keys=True),
                                    "split": split_number,
                                    "train_start": split.train.index[0],
                                    "train_end": split.train.index[-1],
                                    "train_max_label_end": split.train["label_end"].max(),
                                    "validation_start": split.validation.index[0],
                                    "validation_end": split.validation.index[-1],
                                    "validation_max_label_end": split.validation["label_end"].max(),
                                    "next_boundary": split.next_boundary,
                                    "n_train": len(split.train),
                                    "n_validation": len(split.validation),
                                    "smearing": fitted.smearing,
                                    "qlike": score,
                                }
                            )
                        candidate_scores.append(float(np.mean(scores)))
                    if not np.isfinite(candidate_scores).all():
                        raise ValueError(f"Nonfinite validation score: {ticker}/{model}/{year}")
                    # np.argmin makes ties deterministic in configured candidate order.
                    best_candidate = int(np.argmin(candidate_scores))
                    best_params = grid[best_candidate]
                    for row in audit_rows:
                        row["mean_qlike"] = candidate_scores[row["candidate"]]
                        row["selected"] = row["candidate"] == best_candidate
                    tuning_records.extend(audit_rows)
                    fitted = _fit(model, best_params, outer_train, columns, seed)
                    add_predictions(model, feature_set, fitted.predict(test[columns]))
                    fold_records.append(
                        {
                            "ticker": ticker,
                            "model": model,
                            "feature_set": feature_set,
                            "fold": int(year),
                            "train_start": outer_train.index[0],
                            "train_end": outer_train.index[-1],
                            "train_max_label_end": outer_train["label_end"].max(),
                            "test_start": test_start,
                            "test_end": test_dates[-1],
                            "n_train": len(outer_train),
                            "n_test": len(test),
                            "n_features": len(columns),
                            "candidate": best_candidate,
                            "params": json.dumps(best_params, sort_keys=True),
                            "validation_qlike": candidate_scores[best_candidate],
                            "smearing": fitted.smearing,
                        }
                    )
                    if model == "Ridge":
                        coefficients = np.asarray(fitted.estimator[-1].coef_)
                        importance = np.abs(coefficients)
                        kind = "absolute_standardized_coefficient"
                    else:
                        importance = np.asarray(fitted.estimator.feature_importances_, dtype=float)
                        coefficients = np.full(len(columns), np.nan)
                        kind = "training_split_gain"
                    total = float(importance.sum())
                    for feature, value, coefficient in zip(columns, importance, coefficients):
                        importance_records.append(
                            {
                                "ticker": ticker,
                                "model": model,
                                "feature_set": feature_set,
                                "fold": int(year),
                                "feature": feature,
                                "importance": float(value),
                                "normalized_importance": float(value / total) if total else 0.0,
                                "coefficient": float(coefficient),
                                "importance_type": kind,
                            }
                        )
    prediction_frame = pd.concat(predictions, ignore_index=True).sort_values(
        ["date", "ticker", "model", "feature_set"], ignore_index=True
    )
    return {
        "predictions": prediction_frame,
        "folds": pd.DataFrame(fold_records),
        "tuning": pd.DataFrame(tuning_records),
        "importance": pd.DataFrame(importance_records),
    }
