"""One-command research run with provenance-checked forecast reuse."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .data import load_prices
from .features import build_features

LOGGER = logging.getLogger(__name__)


def _fingerprint(config, data_hash):
    h = hashlib.sha256(
        json.dumps({"config": config, "data_hash": data_hash}, sort_keys=True).encode()
    )
    for name in ("data.py", "features.py", "walkforward.py"):
        h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


def _source_hashes():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }


def load_tables(directory):
    tables = {}
    for path in sorted(directory.glob("*.csv")):
        frame = pd.read_csv(path)
        for col in [
            "date",
            "label_end",
            "train_start",
            "train_end",
            "train_max_label_end",
            "test_start",
            "test_end",
            "validation_start",
            "validation_end",
            "validation_max_label_end",
            "next_boundary",
            "origin",
        ]:
            if col in frame:
                frame[col] = pd.to_datetime(frame[col])
        tables[path.stem] = frame
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "run", "report"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument(
        "--report", default="README.md", help="Markdown file containing the research section"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Require a locally verified price snapshot"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Explicitly replace cached Yahoo price snapshot"
    )
    parser.add_argument(
        "--reuse-forecasts",
        action="store_true",
        help="Reuse only if code/config/data fingerprint matches",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path.cwd()
    output = root / args.output
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if args.command == "report":
        from .plots import generate_figures
        from .report import write_report

        manifest = json.loads(manifest_path.read_text())
        config = manifest["config"]
        prices, data_manifest = load_prices(config, root, offline=True)
        if data_manifest["sha256"] != manifest["data"]["sha256"]:
            raise ValueError("Report data snapshot does not match saved run")
        tables = load_tables(table_dir)
        figs = generate_figures(prices, tables, config, output / "figures")
        write_report(tables, manifest, figs, root / args.report, output)
        LOGGER.info("Regenerated %s", args.report)
        return
    config = yaml.safe_load((root / args.config).read_text())
    started = time.perf_counter()
    prices, data_manifest = load_prices(config, root, refresh=args.refresh, offline=args.offline)
    (output / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2))
    LOGGER.info("Verified data: %s", data_manifest["quality"])
    if args.command == "download":
        return
    from .plots import generate_figures
    from .portfolio import run_portfolios
    from .report import write_report
    from .statistics import forecast_metrics, run_inference
    from .walkforward import run_forecasts

    panels, groups = build_features(prices, config)
    fingerprint = _fingerprint(config, data_manifest["sha256"])
    if args.reuse_forecasts:
        if not manifest_path.exists():
            raise ValueError("No completed run manifest to verify; rerun without --reuse-forecasts")
        old = json.loads(manifest_path.read_text())
        if old["forecast_fingerprint"] != fingerprint:
            raise ValueError("Forecast code/config/data changed; rerun without --reuse-forecasts")
        tables = load_tables(table_dir)
        tables = {key: tables[key] for key in ["predictions", "folds", "tuning", "importance"]}
        for key, table in tables.items():
            file_hash = hashlib.sha256((table_dir / f"{key}.csv").read_bytes()).hexdigest()
            if file_hash != old["table_hashes"][key]:
                raise ValueError(f"Cached {key} table changed since completed run")
    else:
        tables = run_forecasts(panels, groups, config)
    LOGGER.info("Running drift-aware portfolios and paired inference")
    portfolio = run_portfolios(prices, tables["predictions"], config)
    tables.update(
        {
            "daily": portfolio["daily"],
            "portfolio_metrics": portfolio["metrics"],
            "weights": portfolio["weights"],
        }
    )
    tables["forecast_metrics"] = forecast_metrics(tables["predictions"])
    tables.update(run_inference(tables["predictions"], tables["daily"], config))
    p = tables["predictions"]
    year_parts = []
    for year, part in p.groupby(p.date.dt.year):
        score = forecast_metrics(part)
        score["year"] = year
        year_parts.append(score)
    tables["yearly_forecast_metrics"] = pd.concat(year_parts, ignore_index=True)
    table_hashes = {}
    for key, table in tables.items():
        table.to_csv(table_dir / f"{key}.csv", index=False)
        table_hashes[key] = hashlib.sha256((table_dir / f"{key}.csv").read_bytes()).hexdigest()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "data": data_manifest,
        "forecast_fingerprint": fingerprint,
        "source_hashes": _source_hashes(),
        "table_hashes": table_hashes,
        "features": groups,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "lightgbm",
                "matplotlib",
                "yfinance",
            ]
        },
        "elapsed_seconds": time.perf_counter() - started,
        "n_origins": int(p.date.nunique()),
        "n_predictions": len(p),
        "oos_origin_start": str(p.date.min().date()),
        "oos_origin_end": str(p.date.max().date()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    LOGGER.info("Generating figures and research report")
    figures = generate_figures(prices, tables, config, output / "figures")
    write_report(tables, manifest, figures, root / args.report, output)
    LOGGER.info("Complete: %s; %.1f seconds", args.report, time.perf_counter() - started)


if __name__ == "__main__":
    main()
