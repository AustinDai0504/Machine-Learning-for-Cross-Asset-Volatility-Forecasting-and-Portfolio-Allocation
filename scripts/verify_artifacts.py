"""Independent numerical audits of a completed local run; no network or fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts"
    manifest = json.loads((output / "run_manifest.json").read_text())
    for filename, expected in manifest["source_hashes"].items():
        assert digest(root / "src/crossvol" / filename) == expected, f"Source changed: {filename}"
    for name, expected in manifest["table_hashes"].items():
        assert digest(output / "tables" / f"{name}.csv") == expected, f"Table changed: {name}"
    assert digest(root / manifest["data"]["price_file"]) == manifest["data"]["sha256"]
    tables = {p.stem: pd.read_csv(p) for p in (output / "tables").glob("*.csv")}
    p = tables["predictions"]
    assert not p.duplicated(["date", "ticker", "model", "feature_set"]).any()
    n_variants = p[["ticker", "model", "feature_set"]].drop_duplicates().shape[0]
    assert p.groupby("date").size().eq(n_variants).all()
    assert p.date.nunique() == manifest["n_origins"]
    assert np.isfinite(p[["pred_var", "target_var"]]).all().all()
    assert p.pred_var.between(1e-8, 4).all() and p.target_var.gt(0).all()
    folds = tables["folds"]
    assert (folds.train_max_label_end < folds.test_start).all()
    tune = tables["tuning"]
    assert (tune.train_max_label_end < tune.validation_start).all()
    assert (tune.validation_max_label_end < tune.next_boundary).all()

    daily = tables["daily"]
    weights = tables["weights"]
    for (model, cost), sub in daily.groupby(["model", "cost_bps"]):
        assert sub.date.is_monotonic_increasing
        assert sub.is_initial_anchor.sum() == 1 and sub.is_initial_anchor.iloc[0]
        assert sub.is_liquidation.sum() == 1 and sub.is_liquidation.iloc[-1]
        np.testing.assert_allclose(np.cumprod(1 + sub.net_return), sub.nav, rtol=1e-12)
        np.testing.assert_allclose(
            sub.net_return,
            sub.gross_return - sub.transaction_cost_return - sub.borrow_cost,
            atol=2e-15,
        )
        np.testing.assert_allclose(sub.transaction_cost, sub.turnover * cost / 10000, atol=2e-15)
        w = weights[(weights.model == model) & (weights.cost_bps == cost)]
        positions = w.groupby("date").position_value.sum()
        np.testing.assert_allclose(positions.to_numpy() + sub.cash.to_numpy(), sub.nav, atol=2e-14)
        trades = w[w.is_rebalance]
        assert trades.weight.abs().max() <= manifest["config"]["portfolio"]["asset_cap"] + 1e-12
        assert (
            trades.groupby("date").weight.apply(lambda x: x.abs().sum()).max()
            <= manifest["config"]["portfolio"]["gross_cap"] + 1e-12
        )
        assert w[w.is_liquidation].weight.eq(0).all()

    fm = tables["forecast_metrics"]
    pooled = fm[(fm.ticker == "Pooled") & fm.feature_set.isin(["baseline", "full"])].set_index(
        "model"
    )
    pm = tables["portfolio_metrics"]
    pm = pm[pm.cost_bps == manifest["config"]["portfolio"]["base_cost_bps"]].set_index("model")
    for row in tables["forecast_inference"].itertuples():
        np.testing.assert_allclose(
            row.estimate, pooled.loc[row.model, "qlike"] - pooled.loc["EWMA", "qlike"], atol=1e-12
        )
    for row in tables["strategy_inference"].itertuples():
        np.testing.assert_allclose(
            row.estimate, pm.loc[row.model, "sharpe"] - pm.loc["EWMA", "sharpe"], atol=1e-12
        )
    print(
        f"Artifact audit passed: {len(p):,} forecasts, {len(folds)} folds, "
        f"{len(pm)} portfolio methods, source/data/table hashes and accounting identities verified."
    )


if __name__ == "__main__":
    main()
