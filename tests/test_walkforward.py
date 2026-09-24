"""Synthetic end-to-end checks for forecast timing and nested selection."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from crossvol.walkforward import VAR_CAP, VAR_FLOOR, _fit, run_forecasts


@pytest.fixture(scope="module")
def experiment():
    rng = np.random.default_rng(31415)
    dates = pd.bdate_range("2017-01-02", periods=820)
    x = np.cumsum(rng.normal(0, 0.05, size=len(dates)))
    panels = {}
    for i, ticker in enumerate(("AAA", "BBB")):
        risk = x + rng.normal(0, 0.1, size=len(dates))
        trend = np.sin(np.arange(len(dates)) / 35 + i)
        cross = np.cos(np.arange(len(dates)) / 70)
        panel = pd.DataFrame(
            {
                "risk": risk,
                "trend": trend,
                "cross_risk": cross,
                "target_var": np.exp(
                    -4 + 0.4 * risk + 0.1 * trend + rng.normal(0, 0.3, len(dates))
                ),
                "label_end": pd.Series(dates, index=dates).shift(-6),
                "hv_var": np.exp(-4 + 0.3 * risk),
                "ewma_var": np.exp(-4 + 0.35 * risk),
            },
            index=dates,
        )
        panel.loc[dates[-6:], "target_var"] = np.nan
        panels[ticker] = panel
    # A date available to only one asset must disappear from every OOS variant.
    panels["BBB"].loc[dates[-12], "cross_risk"] = np.nan
    panels["AAA"].loc[dates[-11], "hv_var"] = 1e-12
    panels["AAA"].loc[dates[-11], "ewma_var"] = 5.0
    groups = {
        "risk_only": ["risk"],
        "no_cross": ["risk", "trend"],
        "full": ["risk", "trend", "cross_risk"],
    }
    config = {
        "research": {
            "oos_start": "2019-11-01",
            "min_train": 100,
            "validation_size": 25,
            "validation_splits": 2,
            "seed": 42,
            "ridge_alphas": [0.1, 10.0],
            "lgbm_grid": [
                {"n_estimators": 8, "num_leaves": 3, "min_child_samples": 10, "reg_lambda": 1.0},
                {"n_estimators": 12, "num_leaves": 3, "min_child_samples": 20, "reg_lambda": 5.0},
            ],
        }
    }
    return panels, groups, config


@pytest.fixture(scope="module")
def result(experiment):
    return run_forecasts(*experiment)


def test_all_assets_models_and_ablations_share_complete_oos_dates(experiment, result):
    panels, _, config = experiment
    predictions = result["predictions"]
    expected = panels["AAA"].index
    expected = expected[expected >= pd.Timestamp(config["research"]["oos_start"])]
    expected = expected[~expected.isin(panels["AAA"].index[-6:])]
    expected = expected[expected != panels["BBB"].index[-12]]
    by_method = predictions.groupby(["ticker", "model", "feature_set"])
    assert len(by_method) == 16  # two assets * (two baselines + six ML variants)
    for _, group in by_method:
        assert group["date"].tolist() == expected.tolist()
    assert not predictions.duplicated(["date", "ticker", "model", "feature_set"]).any()
    assert predictions["pred_var"].between(VAR_FLOOR, VAR_CAP).all()
    bounded = predictions.loc[
        (predictions["ticker"] == "AAA") & (predictions["date"] == panels["AAA"].index[-11])
    ].set_index("model")
    assert bounded.loc["HV21", "pred_var"] == VAR_FLOOR
    assert bounded.loc["EWMA", "pred_var"] == VAR_CAP


def test_all_train_and_validation_labels_finish_before_next_boundary(result):
    folds = result["folds"]
    tuning = result["tuning"]
    assert set(folds["fold"]) == {2019, 2020}
    assert (folds["train_max_label_end"] < folds["test_start"]).all()
    assert (tuning["train_max_label_end"] < tuning["validation_start"]).all()
    assert (tuning["validation_max_label_end"] < tuning["next_boundary"]).all()
    assert (tuning["n_validation"] == 25).all()
    assert (tuning["n_train"] >= 100).all()
    for _, group in folds.groupby(["ticker", "model", "feature_set"]):
        assert group.sort_values("fold")["n_train"].is_monotonic_increasing
    for _, group in tuning.groupby(["ticker", "model", "feature_set", "fold"]):
        means = group.groupby("candidate")["qlike"].mean()
        selected = group.loc[group["selected"], "candidate"].unique()
        assert selected.tolist() == [int(means.idxmin())]
        assert np.allclose(group["mean_qlike"], group["candidate"].map(means))


def test_outer_test_and_overlapping_pretest_labels_cannot_change_forecasts(experiment, result):
    panels, groups, config = experiment
    # Keep just the first annual holdout so later annual refits cannot legitimately
    # use the modified observations from the preceding test year.
    first_boundary = result["folds"]["test_start"].min()
    modified = {}
    original = {}
    for ticker, panel in panels.items():
        original[ticker] = panel.loc[panel.index.year <= 2019].copy()
        changed = original[ticker].copy()
        unobservable = changed["label_end"] >= first_boundary
        # Includes pretest origins whose five-day outcomes cross the boundary.
        assert ((changed.index < first_boundary) & unobservable).any()
        changed.loc[unobservable, "target_var"] *= 100
        modified[ticker] = changed
    before = run_forecasts(original, groups, config)
    after = run_forecasts(modified, groups, config)
    pd.testing.assert_frame_equal(
        before["predictions"].drop(columns="target_var"),
        after["predictions"].drop(columns="target_var"),
    )
    pd.testing.assert_frame_equal(before["tuning"], after["tuning"])
    pd.testing.assert_frame_equal(before["folds"], after["folds"])


def test_repeated_runs_are_deterministic(experiment, result):
    again = run_forecasts(*experiment)
    for table in ("predictions", "folds", "tuning", "importance"):
        pd.testing.assert_frame_equal(result[table], again[table])


def test_standardization_and_smearing_use_fitted_training_sample_only(experiment):
    panels, _, _ = experiment
    train = panels["AAA"].iloc[:100]
    fitted = _fit("Ridge", {"alpha": 10.0}, train, ["risk", "trend"], 42)
    scaler = fitted.estimator[0]
    np.testing.assert_allclose(scaler.mean_, train[["risk", "trend"]].mean())
    residual = np.log(train["target_var"]) - fitted.estimator.predict(train[["risk", "trend"]])
    assert fitted.smearing == pytest.approx(np.exp(residual).mean())
    means_before = scaler.mean_.copy()
    fitted.predict(pd.DataFrame({"risk": [1000.0], "trend": [-1000.0]}))
    np.testing.assert_array_equal(scaler.mean_, means_before)


def test_fail_clearly_when_inner_training_history_is_insufficient(experiment):
    panels, groups, config = experiment
    config = deepcopy(config)
    config["research"]["min_train"] = 10_000
    with pytest.raises(ValueError, match="min_train"):
        run_forecasts(panels, groups, config)


def test_reject_target_columns_in_predictors(experiment):
    panels, groups, config = experiment
    groups = deepcopy(groups)
    groups["full"].append("target_var")
    with pytest.raises(ValueError, match="cannot be predictors"):
        run_forecasts(panels, groups, config)
