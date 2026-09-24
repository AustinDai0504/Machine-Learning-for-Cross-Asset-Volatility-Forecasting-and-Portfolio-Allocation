"""Accounting and timing invariants, using synthetic prices without network I/O."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from crossvol.portfolio import run_portfolios


def _inputs(price_rows, origin_indices=(1,), cost_bps=(0,), borrow_bps=0, trend_window=1):
    values = np.asarray(price_rows, dtype=float)
    if values.ndim == 1:
        values = np.repeat(values[:, None], 3, axis=1)
    dates = pd.bdate_range("2020-01-01", periods=len(values))
    tickers = ["SPY", "TLT", "GLD"]
    prices = (
        pd.DataFrame(values, index=dates, columns=tickers)
        .rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="close")
    )
    predictions = pd.DataFrame(
        [
            {
                "date": dates[i],
                "ticker": ticker,
                "model": model,
                "feature_set": feature_set,
                "pred_var": 0.01,
                "label_end": dates[i + 6],
            }
            for i in origin_indices
            for ticker in tickers
            for model, feature_set in [("EWMA", "baseline"), ("Ridge", "full")]
        ]
    )
    config = {
        "data": {"tickers": tickers},
        "research": {"annualization": 252, "execution_lag": 1, "horizon": 5},
        "portfolio": {
            "trend_window": trend_window,
            "target_vol": 0.10,
            "vol_floor": 0.05,
            "gross_cap": 1.0,
            "asset_cap": 0.6,
            "cost_bps": list(cost_bps),
            "short_borrow_bps": borrow_bps,
            "cash_rate": 0.0,
        },
    }
    return prices, predictions, config, dates


def _daily(result, model="EWMA", cost=0):
    return result["daily"].query("model == @model and cost_bps == @cost").set_index("date")


def test_execute_after_signal_and_earn_only_the_following_return():
    prices, predictions, config, dates = _inputs([100, 101, 202, 222.2, 222.2, 222.2, 222.2, 222.2])
    result = run_portfolios(prices, predictions, config)
    daily = _daily(result)
    assert daily.index.tolist() == dates[1:].tolist()
    assert daily["is_initial_anchor"].tolist() == [True] + [False] * (len(daily) - 1)
    assert daily.loc[dates[1], "nav"] == 1  # all-cash anchor
    assert daily.loc[dates[2], "net_return"] == 0  # doubling before execution is not earned
    assert daily.loc[dates[2], "is_rebalance"]
    assert daily.loc[dates[3], "net_return"] == pytest.approx(0.10)
    assert daily.iloc[-1]["nav"] == pytest.approx(1.10)
    assert daily.iloc[-1]["is_liquidation"]
    assert daily.iloc[-1]["gross_exposure"] == 0


def test_exact_self_financing_initial_and_final_transaction_costs():
    prices, predictions, config, dates = _inputs(
        [99, 100, 100, 100, 100, 100, 100, 100], cost_bps=(50,)
    )
    result = run_portfolios(prices, predictions, config)
    daily = _daily(result, cost=50)
    c = 0.005
    assert daily.loc[dates[2], "nav"] == pytest.approx(1 / (1 + c))
    assert daily.iloc[-1]["nav"] == pytest.approx((1 - c) / (1 + c))
    assert daily.loc[dates[2], "cash"] == pytest.approx(0, abs=1e-14)
    assert daily.loc[dates[2], "turnover"] == pytest.approx(1 / (1 + c))
    assert daily.iloc[-1]["turnover"] == pytest.approx(1)
    assert (daily["transaction_cost_dollars"] > 0).sum() == 2
    np.testing.assert_allclose(
        daily["net_return"],
        daily["gross_return"] - daily["transaction_cost_return"] - daily["borrow_cost"],
        atol=1e-14,
    )
    weights = result["weights"].query("model == 'EWMA' and is_rebalance")
    np.testing.assert_allclose(weights["weight"], weights["target_weight"], atol=1e-14)


def test_rebalance_turnover_uses_drifted_holdings():
    rows = [[90, 90, 90], [95, 95, 95], [100, 100, 100], [100, 100, 100]] + [[120, 100, 100]] * 6
    prices, predictions, config, dates = _inputs(rows, origin_indices=(2, 3), trend_window=2)
    result = run_portfolios(prices, predictions, config)
    daily = _daily(result)
    # The target stays 1/3 each, but the first asset's price creates turnover.
    assert daily.loc[dates[4], "turnover"] == pytest.approx(1 / 12)
    assert daily.loc[dates[4], "nav"] == pytest.approx(16 / 15)
    weights = result["weights"].query("model == 'EWMA' and date == @dates[4]")
    np.testing.assert_allclose(weights["weight"], 1 / 3)


def test_tail_holds_shares_without_repeating_last_target():
    rows = [[90, 90, 90], [95, 95, 95], [100, 100, 100], [100, 100, 100]] + [[120, 100, 100]] * 5
    prices, predictions, config, dates = _inputs(rows, origin_indices=(2,), trend_window=2)
    result = run_portfolios(prices, predictions, config)
    daily = _daily(result)
    assert daily["is_rebalance"].sum() == 1
    assert (daily.loc[dates[4] : dates[7], "turnover"] == 0).all()
    weights = result["weights"].query("model == 'EWMA' and date == @dates[4]").set_index("ticker")
    assert weights.loc["SPY", "weight"] == pytest.approx(0.375)
    assert daily.iloc[-1]["nav"] == pytest.approx(16 / 15)


def test_borrow_uses_previous_close_short_notional_and_all_five_held_days():
    prices, predictions, config, dates = _inputs(
        [101, 100, 100, 100, 100, 100, 100, 100], borrow_bps=252
    )
    daily = _daily(run_portfolios(prices, predictions, config))
    assert daily.loc[dates[2], "borrow_cost_dollars"] == 0
    np.testing.assert_allclose(daily.loc[dates[3] :, "borrow_cost_dollars"], 0.0001)
    assert daily.iloc[-1]["nav"] == pytest.approx(0.9995)
    assert daily.iloc[-1]["gross_exposure"] == 0


def test_all_models_costs_share_dates_and_rebalance_caps_hold_after_costs():
    rows = [[90, 110, 80], [100, 100, 100]] + [[100, 100, 100]] * 6
    prices, predictions, config, dates = _inputs(rows, cost_bps=(0, 5, 100))
    predictions.loc[predictions["ticker"] == "SPY", "pred_var"] = 0.000001
    predictions.loc[predictions["ticker"] == "GLD", "pred_var"] = 4.0
    result = run_portfolios(prices, predictions, config)
    for _, daily in result["daily"].groupby(["model", "cost_bps"]):
        assert daily["date"].tolist() == dates[1:].tolist()
    active = result["weights"].loc[result["weights"]["is_rebalance"]]
    assert (active["weight"].abs() <= 0.6 + 1e-12).all()
    gross = active.groupby(["date", "model", "cost_bps"])["weight"].apply(lambda x: x.abs().sum())
    assert (gross <= 1 + 1e-12).all()
    np.testing.assert_allclose(active["weight"], active["target_weight"], atol=1e-13)
    np.testing.assert_allclose(
        result["daily"]["net_return"],
        result["daily"]["gross_return"]
        - result["daily"]["transaction_cost_return"]
        - result["daily"]["borrow_cost"],
        atol=1e-14,
    )


def test_incomplete_forecast_grid_fails_instead_of_changing_comparison_sample():
    prices, predictions, config, _ = _inputs([99, 100, 100, 100, 100, 100, 100, 100])
    with pytest.raises(ValueError, match="identical forecast dates"):
        run_portfolios(prices, predictions.iloc[:-1], config)


def test_label_horizon_mismatch_is_rejected():
    prices, predictions, config, dates = _inputs([99, 100, 100, 100, 100, 100, 100, 100])
    predictions["label_end"] = dates[-2]
    with pytest.raises(ValueError, match="label_end must equal"):
        run_portfolios(prices, predictions, config)


def test_future_price_change_cannot_change_earlier_positions_or_nav():
    prices, predictions, config, dates = _inputs(
        [99, 100, 100, 100, 100, 100, 100, 100], cost_bps=(5,)
    )
    original = run_portfolios(prices, predictions, config)
    changed_prices = prices.copy()
    changed_prices.loc[changed_prices["date"] >= dates[5], "close"] *= 1.5
    changed = run_portfolios(changed_prices, predictions, deepcopy(config))
    pd.testing.assert_frame_equal(
        original["daily"].loc[original["daily"]["date"] < dates[5]].reset_index(drop=True),
        changed["daily"].loc[changed["daily"]["date"] < dates[5]].reset_index(drop=True),
    )


def test_metrics_exclude_initial_anchor_from_annualization():
    prices, predictions, config, _ = _inputs([99, 100, 100, 101, 102, 103, 104, 105])
    result = run_portfolios(prices, predictions, config)
    row = result["metrics"].query("model == 'EWMA'").iloc[0]
    daily = _daily(result)
    assert row["n_periods"] == 6  # entry interval plus five held returns
    assert row["cagr"] == pytest.approx(1.05 ** (252 / 6) - 1)
    assert row["ann_vol"] == pytest.approx(daily["net_return"].iloc[1:].std(ddof=1) * np.sqrt(252))
