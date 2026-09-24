"""Close-to-close, self-financing portfolio evaluation.

A forecast observed at close t becomes a target at close t + execution_lag.
Holdings first earn the following close-to-close return. Trades rebalance to
weights of *post-transaction-cost* NAV, solving the self-financing equation
instead of subtracting a fee after allocating all capital. Caps apply when a
target is executed; holdings are allowed to drift between forecast dates.

The daily table includes an initial all-cash anchor and ends with liquidation
at the final forecast's label_end. No targets are repeated in the unlabeled
tail. Cash is a single account: sale proceeds are available, the configured
cash rate applies symmetrically to its balance, and short borrow is a separate
charge. This deliberately omits margin segregation and broker-specific rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _post_cost_nav_ratio(
    current_weights: np.ndarray, target_weights: np.ndarray, cost_rate: float
) -> float:
    """Solve z = 1 - c * sum(abs(w_target * z - w_current)).

    Here current weights use pretrade NAV and target weights use posttrade NAV.
    The fixed-point contraction has constant c * target gross < 1; the returned
    solution satisfies the equation to floating-point precision. A liquidation
    has target weights zero and therefore converges in one update.
    """
    if cost_rate == 0:
        return 1.0
    if cost_rate * np.abs(target_weights).sum() >= 1:
        raise ValueError("Transaction cost times target gross must be below one")
    if cost_rate * np.abs(current_weights).sum() >= 1:
        raise ValueError("Transaction costs would exhaust portfolio NAV")
    ratio = 1.0
    for _ in range(1000):
        updated = 1.0 - cost_rate * np.abs(target_weights * ratio - current_weights).sum()
        if abs(updated - ratio) <= 2e-15:
            return float(updated)
        ratio = updated
    raise RuntimeError("Self-financing transaction-cost solution did not converge")


def _cap_weights(weights: np.ndarray, asset_cap: float, gross_cap: float) -> np.ndarray:
    capped = np.clip(weights, -asset_cap, asset_cap)
    gross = np.abs(capped).sum()
    if gross > gross_cap:
        capped *= gross_cap / gross
    return capped


def _portfolio_metrics(daily: pd.DataFrame, annualization: int) -> dict:
    """Exclude the initial cash anchor from return moments and elapsed time."""
    returns = daily["net_return"].iloc[1:]
    periods = len(returns)
    standard_deviation = float(returns.std(ddof=1)) if periods > 1 else 0.0
    ann_vol = standard_deviation * np.sqrt(annualization)
    sharpe = (
        float(returns.mean() / standard_deviation * np.sqrt(annualization))
        if standard_deviation > 1e-14
        else np.nan
    )
    nav = daily["nav"]
    transaction = float(daily["transaction_cost_return"].sum())
    borrow = float(daily["borrow_cost"].sum())
    return {
        "model": daily["model"].iloc[0],
        "cost_bps": daily["cost_bps"].iloc[0],
        "cagr": float(nav.iloc[-1] ** (annualization / periods) - 1),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
        "annual_turnover": float(daily["turnover"].sum() * annualization / periods),
        # Additive daily return drag, not terminal wealth lost to costs.
        "total_cost": transaction + borrow,
        "total_transaction_cost": transaction,
        "total_borrow_cost": borrow,
        # Exposure held at the start of each return interval.
        "avg_gross": float(daily["gross_exposure"].iloc[:-1].mean()),
        "final_nav": float(nav.iloc[-1]),
        "n_periods": periods,
        "start": daily["date"].iloc[0],
        "end": daily["date"].iloc[-1],
    }


def run_portfolios(
    prices: pd.DataFrame, predictions: pd.DataFrame, config: dict
) -> dict[str, pd.DataFrame]:
    """Evaluate forecast-sized trend portfolios on one identical price calendar.

    ``daily`` provides close NAV and post-close exposure. ``turnover`` and
    ``transaction_cost`` are dollars traded/paid divided by pretrade NAV;
    ``transaction_cost_return`` and ``borrow_cost`` divide charges by prior
    close NAV, so net_return = gross_return - transaction_cost_return -
    borrow_cost. ``total_cost`` in metrics sums these daily return drags.

    ``weights`` contains actual post-close weights and target weights only
    on execution/liquidation dates (NaN otherwise). ``origin`` identifies the
    forecast responsible for each execution. Common sample violations raise
    instead of silently dropping a model, asset, or date.
    """
    portfolio_config = config["portfolio"]
    research_config = config["research"]
    tickers = list(config["data"]["tickers"])
    n_assets = len(tickers)
    annualization = int(research_config.get("annualization", 252))
    execution_lag = int(research_config.get("execution_lag", 1))
    horizon = int(research_config.get("horizon", 5))
    trend_window = int(portfolio_config.get("trend_window", 126))
    target_vol = float(portfolio_config.get("target_vol", 0.10))
    vol_floor = float(portfolio_config.get("vol_floor", 0.05))
    asset_cap = float(portfolio_config.get("asset_cap", 0.60))
    gross_cap = float(portfolio_config.get("gross_cap", 1.0))
    costs = list(dict.fromkeys(float(x) for x in portfolio_config["cost_bps"]))
    borrow_rate = float(portfolio_config.get("short_borrow_bps", 50)) / 10000
    cash_rate = float(portfolio_config.get("cash_rate", 0.0))
    if n_assets == 0 or len(set(tickers)) != n_assets:
        raise ValueError("Portfolio tickers must be a nonempty, unique list")
    if min(annualization, execution_lag, horizon, trend_window) <= 0:
        raise ValueError("Calendar, horizon, lag, and trend parameters must be positive")
    if min(vol_floor, asset_cap, gross_cap) <= 0 or target_vol < 0:
        raise ValueError("Risk limits must be positive and target volatility nonnegative")
    if not costs or any(not np.isfinite(c) or c < 0 for c in costs) or borrow_rate < 0:
        raise ValueError("Trading and borrow costs must be nonnegative")

    price_data = prices.loc[prices["ticker"].isin(tickers), ["date", "ticker", "close"]].copy()
    price_data["date"] = pd.to_datetime(price_data["date"])
    if price_data.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate price date/ticker observations")
    closes = price_data.pivot(index="date", columns="ticker", values="close").sort_index()
    closes = closes.reindex(columns=tickers)
    if closes.empty or not np.isfinite(closes.to_numpy()).all() or (closes <= 0).any().any():
        raise ValueError(
            "Every portfolio asset needs a positive close on the common price calendar"
        )

    selected = predictions.loc[
        ((predictions["model"].isin(["HV21", "EWMA"])) & (predictions["feature_set"] == "baseline"))
        | (
            (predictions["model"].isin(["Ridge", "LightGBM"]))
            & (predictions["feature_set"] == "full")
        )
    ].copy()
    if selected.empty:
        raise ValueError("No baseline or full-feature portfolio forecasts supplied")
    selected["date"] = pd.to_datetime(selected["date"])
    selected["label_end"] = pd.to_datetime(selected["label_end"])
    if selected.duplicated(["date", "ticker", "model"]).any():
        raise ValueError("Duplicate portfolio forecasts")
    if set(selected["ticker"]) != set(tickers):
        raise ValueError("Portfolio forecast assets must match configured assets")
    if not np.isfinite(selected["pred_var"]).all() or (selected["pred_var"] <= 0).any():
        raise ValueError("Forecast variance must be finite and strictly positive")
    model_order = [m for m in ["HV21", "EWMA", "Ridge", "LightGBM"] if m in set(selected["model"])]
    origins = pd.DatetimeIndex(sorted(selected["date"].unique()))
    complete_index = pd.MultiIndex.from_product(
        [origins, tickers, model_order], names=["date", "ticker", "model"]
    )
    actual_index = pd.MultiIndex.from_frame(selected[["date", "ticker", "model"]])
    if len(actual_index) != len(complete_index) or len(complete_index.difference(actual_index)):
        raise ValueError("All portfolio models and assets must share identical forecast dates")
    if (
        selected["label_end"].isna().any()
        or (selected.groupby("date")["label_end"].nunique() != 1).any()
    ):
        raise ValueError("Each forecast origin must share one nonmissing label_end")

    calendar = closes.index
    origin_locations = calendar.get_indexer(origins)
    if (origin_locations < trend_window).any():
        raise ValueError("Forecast dates must exist in prices and have complete trend history")
    execution_locations = origin_locations + execution_lag
    end_locations = execution_locations + horizon
    if (end_locations >= len(calendar)).any():
        raise ValueError("Prices do not cover execution and the full forecast label horizon")
    label_ends = selected.groupby("date")["label_end"].first().reindex(origins)
    if not np.array_equal(label_ends.to_numpy(), calendar[end_locations].to_numpy()):
        raise ValueError("label_end must equal origin + execution_lag + horizon trading days")
    start_location = int(execution_locations[0] - 1)
    final_location = int(end_locations[-1])
    sample_dates = calendar[start_location : final_location + 1]
    sample_prices = closes.iloc[start_location : final_location + 1].to_numpy(dtype=float)
    returns = np.zeros_like(sample_prices)
    returns[1:] = sample_prices[1:] / sample_prices[:-1] - 1
    all_closes = closes.to_numpy(dtype=float)
    trend = np.sign(all_closes[origin_locations] / all_closes[origin_locations - trend_window] - 1)
    execution_map = {int(loc - start_location): i for i, loc in enumerate(execution_locations)}
    desired: dict[str, np.ndarray] = {}
    for model in model_order:
        variances = selected.loc[selected["model"] == model].pivot(
            index="date", columns="ticker", values="pred_var"
        )
        variances = variances.reindex(index=origins, columns=tickers).to_numpy(dtype=float)
        weights = (
            trend * target_vol / (np.sqrt(n_assets) * np.maximum(np.sqrt(variances), vol_floor))
        )
        desired[model] = np.vstack([_cap_weights(w, asset_cap, gross_cap) for w in weights])
    desired["EqualWeightTrend"] = np.vstack(
        [_cap_weights(direction / n_assets, asset_cap, gross_cap) for direction in trend]
    )

    daily_records: list[dict] = []
    weight_records: list[dict] = []
    metrics_records: list[dict] = []
    for model, target_path in desired.items():
        for cost_bps in costs:
            cost_rate = cost_bps / 10000
            cash = 1.0
            position_values = np.zeros(n_assets)
            nav = 1.0
            scenario_records: list[dict] = []
            for step, date in enumerate(sample_dates):
                previous_nav = nav
                price_pnl = float(np.dot(position_values, returns[step]))
                borrow_dollars = (
                    borrow_rate / annualization * float(np.maximum(-position_values, 0).sum())
                    if step
                    else 0.0
                )
                cash_interest = cash * cash_rate / annualization if step else 0.0
                position_values = position_values * (1 + returns[step])
                cash += cash_interest - borrow_dollars
                pretrade_nav = float(cash + position_values.sum())
                if pretrade_nav <= 0 or not np.isfinite(pretrade_nav):
                    raise ValueError(f"Portfolio {model} at {date} has nonpositive/nonfinite NAV")
                is_liquidation = step == len(sample_dates) - 1
                is_rebalance = step in execution_map and not is_liquidation
                target = np.full(n_assets, np.nan)
                origin = pd.NaT
                trade_dollars = 0.0
                transaction_dollars = 0.0
                if is_rebalance or is_liquidation:
                    if is_liquidation:
                        target = np.zeros(n_assets)
                    else:
                        origin_index = execution_map[step]
                        target = target_path[origin_index]
                        origin = origins[origin_index]
                    ratio = _post_cost_nav_ratio(position_values / pretrade_nav, target, cost_rate)
                    new_positions = target * (pretrade_nav * ratio)
                    trades = new_positions - position_values
                    trade_dollars = float(np.abs(trades).sum())
                    transaction_dollars = cost_rate * trade_dollars
                    cash -= float(trades.sum()) + transaction_dollars
                    position_values = new_positions
                nav = float(cash + position_values.sum())
                if nav <= 0 or not np.isfinite(nav):
                    raise ValueError(f"Portfolio {model} at {date} exhausted its NAV after trading")
                actual_weights = position_values / nav
                record = {
                    "date": date,
                    "model": model,
                    "cost_bps": cost_bps,
                    "net_return": nav / previous_nav - 1,
                    "gross_return": (price_pnl + cash_interest) / previous_nav,
                    "nav": nav,
                    "pretrade_nav": pretrade_nav,
                    "cash": cash,
                    "gross_exposure": float(np.abs(actual_weights).sum()),
                    "turnover": trade_dollars / pretrade_nav,
                    "transaction_cost": transaction_dollars / pretrade_nav,
                    "transaction_cost_return": transaction_dollars / previous_nav,
                    "borrow_cost": borrow_dollars / previous_nav,
                    "transaction_cost_dollars": transaction_dollars,
                    "borrow_cost_dollars": borrow_dollars,
                    "is_rebalance": is_rebalance,
                    "is_liquidation": is_liquidation,
                    "is_initial_anchor": step == 0,
                    "origin": origin,
                }
                scenario_records.append(record)
                for asset_index, ticker in enumerate(tickers):
                    weight_records.append(
                        {
                            "date": date,
                            "model": model,
                            "cost_bps": cost_bps,
                            "ticker": ticker,
                            "weight": actual_weights[asset_index],
                            "target_weight": target[asset_index],
                            "position_value": position_values[asset_index],
                            "nav": nav,
                            "is_rebalance": is_rebalance,
                            "is_liquidation": is_liquidation,
                            "origin": origin,
                        }
                    )
            scenario = pd.DataFrame(scenario_records)
            metrics_records.append(_portfolio_metrics(scenario, annualization))
            daily_records.extend(scenario_records)
    return {
        "daily": pd.DataFrame(daily_records),
        "metrics": pd.DataFrame(metrics_records),
        "weights": pd.DataFrame(weight_records),
    }
