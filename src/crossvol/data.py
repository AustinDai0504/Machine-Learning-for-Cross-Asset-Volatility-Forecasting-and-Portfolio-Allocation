"""Explicit adjusted-price ingestion, validation and immutable-cache provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_prices(prices: pd.DataFrame, tickers: list[str]) -> dict:
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    if not set(required).issubset(prices):
        raise ValueError(f"Prices require columns {required}")
    if prices.empty or prices.duplicated(["date", "ticker"]).any():
        raise ValueError("Empty prices or duplicate date/ticker observations")
    if prices.date.isna().any() or set(prices.ticker) != set(tickers):
        raise ValueError("Invalid dates or ticker universe")
    values = prices[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :4] <= 0).any():
        raise ValueError("Nonfinite prices/volume or nonpositive OHLC")
    if (prices.volume < 0).any():
        raise ValueError("Negative volume")
    tol = 1e-6
    if (prices.high + tol < prices[["open", "close", "low"]].max(axis=1)).any() or (
        prices.low - tol > prices[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("Inconsistent OHLC bounds")
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    if wide.isna().any().any():
        raise ValueError("Misaligned trading calendars: no silent filling or row deletion allowed")
    if len(wide) < 2:
        raise ValueError("Need at least two trading dates")
    log_returns = np.log(wide).diff()
    if (log_returns.abs() > 0.5).any().any():
        raise ValueError("Daily absolute log return >50%: inspect corporate-action/data error")
    return {
        "n_rows": len(prices),
        "n_dates": len(wide),
        "start": str(wide.index.min().date()),
        "end": str(wide.index.max().date()),
        "tickers": tickers,
        "missing_cells": int(wide.isna().sum().sum()),
        "duplicate_rows": 0,
        "zero_volume_rows": int((prices.volume == 0).sum()),
        "max_abs_daily_log_return": float(log_returns.abs().max().max()),
        "calendar_gaps_over_4_days": int((wide.index.to_series().diff().dt.days > 4).sum()),
    }


def load_prices(config: dict, root: Path, refresh: bool = False, offline: bool = False):
    settings = config["data"]
    path = root / settings["cache"]
    metadata_path = path.with_suffix(".metadata.json")
    request = {k: settings[k] for k in ["tickers", "start", "end"]}
    if refresh and offline:
        raise ValueError("--refresh and --offline cannot be combined")
    if path.exists() and not refresh:
        if not metadata_path.exists():
            raise ValueError(
                "Cache provenance missing. Use --refresh to fetch a verified snapshot."
            )
        metadata = json.loads(metadata_path.read_text())
        if metadata["request"] != request or metadata["sha256"] != sha256(path):
            raise ValueError("Cache request/hash mismatch. Inspect cache or explicitly --refresh.")
        prices = pd.read_csv(path, parse_dates=["date"])
    else:
        if offline:
            raise FileNotFoundError(
                "Offline cache missing; run crossvol download once with network access"
            )
        import yfinance as yf

        frames = []
        for ticker in settings["tickers"]:
            frame = yf.download(
                ticker,
                start=settings["start"],
                end=settings["end"],
                interval="1d",
                auto_adjust=True,
                back_adjust=False,
                repair=False,
                actions=False,
                threads=False,
                progress=False,
                multi_level_index=False,
                timeout=30,
            )
            if frame is None or frame.empty:
                raise RuntimeError(
                    f"No real data returned for {ticker}; synthetic fallback is prohibited"
                )
            frame.index = pd.to_datetime(frame.index).tz_localize(None)
            frame = frame.rename(columns=str.lower).rename_axis("date").reset_index()
            frame["ticker"] = ticker
            frames.append(frame[["date", "ticker", "open", "high", "low", "close", "volume"]])
        prices = pd.concat(frames).sort_values(["date", "ticker"]).reset_index(drop=True)
        validate_prices(prices, settings["tickers"])
        if prices.date.min() > pd.Timestamp(settings["start"]) + pd.Timedelta(days=7):
            raise ValueError("Downloaded history starts later than requested")
        if prices.date.max() < pd.Timestamp(settings["end"]) - pd.Timedelta(days=7):
            raise ValueError("Downloaded history ends earlier than requested")
        path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(path, index=False, float_format="%.12g")
        metadata = {
            "source": "Yahoo Finance via yfinance",
            "yfinance_version": yf.__version__,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "request": request,
            "adjustment": "auto_adjust=True: split/dividend adjusted OHLC; repair=False",
            "sha256": sha256(path),
            "price_file": str(path.relative_to(root)),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))
        # All runs consume the exact serialized snapshot, including the initial run.
        prices = pd.read_csv(path, parse_dates=["date"])
    metadata["quality"] = validate_prices(prices, settings["tickers"])
    return prices, metadata
