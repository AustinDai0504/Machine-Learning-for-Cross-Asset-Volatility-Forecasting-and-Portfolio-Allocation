"""Tests of paired inference, metric weighting, and sample integrity."""

import numpy as np
import pandas as pd
import pytest

from crossvol.statistics import forecast_metrics, holm_adjust, qlike, run_inference


def _sample(n_dates=160):
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rng = np.random.default_rng(17)
    rows = []
    # A persistent loss process and common market shock exercise paired sampling.
    common = 1 + 0.3 * np.sin(np.arange(n_dates) / 11)
    for ticker, multiplier in [("SPY", 1.0), ("TLT", 0.7), ("GLD", 0.8)]:
        target = 0.04 * multiplier * common
        for model, forecast_ratio, feature_set in [
            ("EWMA", 2.0, "baseline"),
            ("Ridge", 1.0, "full"),
            ("LightGBM", 2.0, "full"),
        ]:
            rows.extend(
                {
                    "date": date,
                    "ticker": ticker,
                    "model": model,
                    "feature_set": feature_set,
                    "target_var": realized,
                    "pred_var": realized * forecast_ratio,
                }
                for date, realized in zip(dates, target)
            )
    returns = rng.normal(0.0002, 0.01, n_dates)
    daily = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "model": model,
                    "cost_bps": 5,
                    "net_return": returns + alpha,
                }
            )
            for model, alpha in [("EWMA", 0), ("Ridge", 0.0005), ("LightGBM", 0)]
        ],
        ignore_index=True,
    )
    config = {
        "research": {"annualization": 252, "seed": 42},
        "portfolio": {"base_cost_bps": 5},
        "inference": {"n_bootstrap": 240, "block_lengths": [10, 20, 60]},
    }
    return pd.DataFrame(rows), daily, config


def test_qlike_is_scale_invariant_and_penalizes_miscalibration():
    target = np.array([0.01, 0.04, 0.2])
    np.testing.assert_allclose(qlike(target, target), 0)
    np.testing.assert_allclose(qlike(target, 2 * target), qlike(100 * target, 200 * target))
    assert (qlike(target, 2 * target) > 0).all()
    with pytest.raises(ValueError, match="strictly positive"):
        qlike(target, np.array([0.1, 0, 0.2]))


def test_pooled_metrics_weight_dates_equally_and_report_variance_calibration():
    # Day one has two assets and day two has one: dates must still weigh equally.
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "ticker": ["SPY", "TLT", "SPY"],
            "model": "Ridge",
            "feature_set": "full",
            "target_var": [0.04, 0.04, 0.16],
            "pred_var": [0.04, 0.04, 0.04],
        }
    )
    metrics = forecast_metrics(frame)
    pooled = metrics.set_index("ticker").loc["Pooled"]
    assert pooled["qlike"] == pytest.approx((4 - np.log(4) - 1) / 2)
    assert pooled["vol_rmse"] == pytest.approx(np.sqrt(0.2**2 / 2))
    assert pooled["vol_mae"] == pytest.approx(0.1)
    assert pooled["calibration"] == pytest.approx(2.5)
    assert pooled["n_obs"] == 3
    assert pooled["n_dates"] == 2
    with pytest.raises(ValueError, match="Duplicate"):
        forecast_metrics(pd.concat([frame, frame.iloc[[0]]]))


def test_identical_strategies_and_forecasts_have_exactly_zero_paired_intervals():
    predictions, daily, config = _sample()
    result = run_inference(predictions, daily, config)
    forecast = result["forecast_inference"]
    identical = forecast.loc[forecast["model"] == "LightGBM"]
    np.testing.assert_allclose(identical[["estimate", "ci_low", "ci_high"]], 0)
    np.testing.assert_allclose(identical[["p_value", "p_holm"]], 1)
    strategy = result["strategy_inference"]
    identical = strategy.loc[strategy["model"] == "LightGBM"]
    np.testing.assert_allclose(identical[["estimate", "ci_low", "ci_high"]], 0)
    assert "p_value" not in strategy.columns
    assert (strategy["valid_replicates"] == 240).all()


def test_known_improvement_has_correct_sign_and_is_reproducible_under_reordering():
    predictions, daily, config = _sample()
    first = run_inference(predictions, daily, config)
    second = run_inference(
        predictions.sample(frac=1, random_state=4), daily.sample(frac=1, random_state=9), config
    )
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key])
    ridge_forecast = first["forecast_inference"].query("model == 'Ridge'")
    assert (ridge_forecast["ci_high"] < 0).all()
    assert (ridge_forecast["p_holm"] < 0.05).all()
    ridge_strategy = first["strategy_inference"].query("model == 'Ridge'")
    assert (ridge_strategy["ci_low"] > 0).all()


def test_inference_rejects_unpaired_dates_and_inconsistent_targets():
    predictions, daily, config = _sample()
    with pytest.raises(ValueError, match="identical complete dates/assets"):
        run_inference(predictions.iloc[1:], daily, config)
    with pytest.raises(ValueError, match="identical finite dates"):
        run_inference(predictions, daily.iloc[1:], config)
    predictions.loc[0, "target_var"] *= 1.2
    with pytest.raises(ValueError, match="identical realized targets"):
        run_inference(predictions, daily, config)


def test_holm_adjusts_a_family_in_input_order():
    np.testing.assert_allclose(holm_adjust(np.array([0.04, 0.01, 0.03])), [0.06, 0.03, 0.06])
    np.testing.assert_allclose(holm_adjust(np.array([0.001, 0.7])), [0.002, 0.7])


def test_replicating_an_asset_does_not_spuriously_narrow_paired_intervals():
    predictions, daily, config = _sample()
    one_asset = predictions.loc[predictions["ticker"] == "SPY"].copy()
    # Give forecasts time-varying error so the nonzero interval is informative.
    dates = sorted(one_asset["date"].unique())
    distortion = dict(zip(dates, 1.2 + 0.3 * np.sin(np.arange(len(dates)) / 10)))
    mask = one_asset["model"] == "Ridge"
    one_asset.loc[mask, "pred_var"] *= one_asset.loc[mask, "date"].map(distortion)
    repeated_assets = pd.concat(
        [one_asset.assign(ticker=ticker) for ticker in ["SPY", "COPY1", "COPY2"]]
    )
    single = run_inference(one_asset, daily, config)["forecast_inference"]
    repeated = run_inference(repeated_assets, daily, config)["forecast_inference"]
    np.testing.assert_allclose(
        single[["estimate", "ci_low", "ci_high"]],
        repeated[["estimate", "ci_low", "ci_high"]],
        atol=1e-14,
    )
    ridge = single.query("model == 'Ridge'")
    assert (ridge["ci_high"] > ridge["ci_low"]).all()


def test_cash_anchor_is_excluded_consistently_with_reported_sharpe():
    predictions, daily, config = _sample()
    original = run_inference(predictions, daily, config)["strategy_inference"]
    anchor_date = daily["date"].min() - pd.Timedelta(days=1)
    anchors = pd.DataFrame(
        {
            "date": anchor_date,
            "model": ["EWMA", "Ridge", "LightGBM"],
            "cost_bps": 5,
            "net_return": 0.0,
            "is_initial_anchor": True,
        }
    )
    with_anchor = pd.concat([anchors, daily.assign(is_initial_anchor=False)], ignore_index=True)
    result = run_inference(predictions, with_anchor, config)["strategy_inference"]
    pd.testing.assert_frame_equal(original, result)
    moments = daily.groupby("model")["net_return"].agg(["mean", "std"])
    sharpes = moments["mean"] / moments["std"] * np.sqrt(252)
    for row in result.itertuples():
        assert row.estimate == pytest.approx(sharpes[row.model] - sharpes["EWMA"])
        assert row.n_dates == 160
