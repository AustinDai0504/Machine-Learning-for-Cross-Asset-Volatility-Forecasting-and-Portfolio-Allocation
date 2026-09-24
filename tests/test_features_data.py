import numpy as np
import pandas as pd
import pytest

from crossvol.data import validate_prices
from crossvol.features import build_features, forward_variance


def synthetic_prices(n=220):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2010-01-01", periods=n)
    frames = []
    for ticker in ("SPY", "TLT", "GLD"):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        frames.append(
            pd.DataFrame(
                dict(
                    date=dates,
                    ticker=ticker,
                    open=close,
                    high=close * 1.01,
                    low=close * 0.99,
                    close=close,
                    volume=100,
                )
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_forward_label_exact_execution_delay_and_unavailable_tail():
    r = pd.Series(np.arange(10, dtype=float))
    y = forward_variance(r, horizon=3, execution_lag=1, annualization=1)
    assert y.iloc[0] == pytest.approx((2**2 + 3**2 + 4**2) / 3)
    assert y.iloc[-4:].isna().all()


def test_features_cannot_see_mutated_future():
    prices = synthetic_prices()
    config = {
        "data": {"tickers": ["SPY", "TLT", "GLD"]},
        "research": {"annualization": 252, "horizon": 5, "execution_lag": 1, "ewma_lambda": 0.94},
    }
    original, groups = build_features(prices, config)
    for ticker, panel in original.items():
        own_cross = [c for c in groups["full"] if c.startswith(f"cross_{ticker}_")]
        assert (panel[own_cross] == 0).all().all()
    boundary = sorted(prices.date.unique())[160]
    changed = prices.copy()
    mask = changed.date > sorted(prices.date.unique())[161]
    changed.loc[mask, ["open", "high", "low", "close"]] *= 1.7
    altered, _ = build_features(changed, config)
    for ticker in original:
        pd.testing.assert_frame_equal(
            original[ticker].loc[:boundary, groups["full"]],
            altered[ticker].loc[:boundary, groups["full"]],
        )
        assert (
            original[ticker].loc[boundary, "target_var"]
            != altered[ticker].loc[boundary, "target_var"]
        )


def test_bad_prices_are_rejected_instead_of_filled():
    prices = synthetic_prices()
    validate_prices(prices, ["SPY", "TLT", "GLD"])
    with pytest.raises(ValueError, match="Misaligned"):
        validate_prices(prices.iloc[1:], ["SPY", "TLT", "GLD"])
    with pytest.raises(ValueError, match="duplicate"):
        validate_prices(pd.concat([prices, prices.iloc[[0]]]), ["SPY", "TLT", "GLD"])


def test_cached_snapshot_tampering_is_rejected(tmp_path):
    import json
    from crossvol.data import load_prices, sha256

    config = {
        "data": {
            "tickers": ["SPY", "TLT", "GLD"],
            "start": "2010-01-01",
            "end": "2011-01-01",
            "cache": "prices.csv",
        }
    }
    path = tmp_path / "prices.csv"
    synthetic_prices().to_csv(path, index=False)
    metadata = {
        "request": {k: config["data"][k] for k in ["tickers", "start", "end"]},
        "sha256": sha256(path),
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    load_prices(config, tmp_path, offline=True)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_prices(config, tmp_path, offline=True)
