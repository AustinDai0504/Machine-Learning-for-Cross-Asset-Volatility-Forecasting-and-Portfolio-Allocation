"""Publication-friendly static figures generated from saved research outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

COLORS = {
    "HV21": "#8497A5",
    "EWMA": "#EA9A3B",
    "Ridge": "#197B8D",
    "LightGBM": "#8F5399",
    "EqualWeightTrend": "#A7AF5C",
}
ORDER = list(COLORS)


def generate_figures(prices, tables, config, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "#FAFBFD",
            "axes.facecolor": "#FAFBFD",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.17,
            "font.size": 10,
            "savefig.dpi": 180,
            "figure.constrained_layout.use": True,
        }
    )
    files = []

    def save(fig, name):
        fig.savefig(output / f"{name}.png", bbox_inches="tight")
        fig.savefig(output / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)
        files.append(name)

    close = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    (close / close.iloc[0] * 100).plot(ax=axes[0], linewidth=1.3)
    axes[0].set(
        title="Cross-asset market context | 2005-2025", ylabel="Adjusted price (start = 100)"
    )
    rv = np.sqrt(np.log(close).diff().pow(2).rolling(21).mean() * 252)
    rv.plot(ax=axes[1], linewidth=1, legend=False)
    axes[1].set(ylabel="21-day annualized volatility", xlabel="")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    save(fig, "01_market_context")

    p = tables["predictions"]
    primary = p[p.feature_set.isin(["baseline", "full"])].copy()
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, ticker in zip(axes, config["data"]["tickers"]):
        sub = primary[(primary.ticker == ticker) & (primary.date >= "2019-01-01")]
        actual = sub.drop_duplicates("date").set_index("date").target_var.pow(0.5)
        ax.plot(
            actual.index,
            actual,
            color="#BEC5CD",
            linewidth=0.7,
            alpha=0.85,
            label="Forward 5-day proxy",
        )
        for model in ["EWMA", "Ridge", "LightGBM"]:
            series = sub[sub.model == model].set_index("date").pred_var.pow(0.5)
            ax.plot(series.index, series, color=COLORS[model], linewidth=0.95, label=model)
        ax.set(ylabel=ticker)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    axes[0].set_title("Out-of-sample volatility forecasts | date = information origin")
    axes[0].legend(ncol=4, fontsize=9)
    save(fig, "02_forecasts")

    metrics = tables["forecast_metrics"]
    main = metrics[metrics.feature_set.isin(["baseline", "full"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for metric, ax, title in zip(
        ["qlike", "vol_rmse"], axes, ["QLIKE (lower is better)", "Volatility RMSE (annualized)"]
    ):
        pivot = main.pivot(index="ticker", columns="model", values=metric).reindex(
            columns=ORDER[:4]
        )
        pivot.plot.bar(ax=ax, color=[COLORS[m] for m in pivot], rot=0)
        ax.set(title=title, xlabel="", ylabel="")
        ax.legend(fontsize=9)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    save(fig, "03_forecast_metrics")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ablation_values = metrics[metrics.model.isin(["Ridge", "LightGBM"])].qlike
    color_min, color_max = ablation_values.min(), ablation_values.max()
    for ax, model in zip(axes, ["Ridge", "LightGBM"]):
        mat = metrics[metrics.model == model].pivot(
            index="ticker", columns="feature_set", values="qlike"
        )
        mat = mat.reindex(columns=["risk_only", "no_cross", "full"])
        im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=color_min, vmax=color_max)
        ax.set_xticks(range(3), ["Risk only", "+ Own dynamics", "+ Cross asset"])
        ax.set_yticks(range(len(mat)), mat.index)
        ax.set_title(f"{model}: feature ablation (QLIKE)")
        ax.grid(False)
        for (i, j), val in np.ndenumerate(mat.to_numpy()):
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                color="white" if val > color_min + 0.64 * (color_max - color_min) else "#12243C",
            )
        fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "04_feature_ablation")

    base_cost = config["portfolio"]["base_cost_bps"]
    daily = tables["daily"]
    d = daily[daily.cost_bps == base_cost]
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]}
    )
    for model in ORDER:
        series = d[d.model == model].set_index("date").nav
        axes[0].plot(series.index, series, color=COLORS[model], label=model, linewidth=1.6)
        dd = series / series.cummax().clip(lower=1) - 1
        axes[1].plot(series.index, dd, color=COLORS[model], linewidth=1)
    axes[0].set(
        title=f"Identical trend rule, different position sizing | {base_cost} bps one-way costs",
        ylabel="Net wealth (start = 1)",
    )
    axes[0].legend(ncol=5, fontsize=9)
    axes[1].set(ylabel="Drawdown", xlabel="")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    save(fig, "05_portfolio_performance")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    pm = tables["portfolio_metrics"]
    for model in ORDER:
        sub = pm[pm.model == model].sort_values("cost_bps")
        axes[0].plot(sub.cost_bps, sub.sharpe, "o-", color=COLORS[model], label=model)
        axes[1].plot(sub.cost_bps, sub.cagr, "o-", color=COLORS[model], label=model)
    axes[0].set(
        title="Transaction-cost sensitivity",
        xlabel="One-way cost (bps)",
        ylabel="Net Sharpe (cash benchmark = 0)",
    )
    axes[1].set(xlabel="One-way cost (bps)", ylabel="Net CAGR")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    axes[0].legend(fontsize=9)
    for ax in axes:
        ax.set_xticks(config["portfolio"]["cost_bps"])
    save(fig, "06_cost_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in zip(
        axes,
        ["forecast_inference", "strategy_inference"],
        [
            "QLIKE difference vs EWMA (negative is better)",
            "Net Sharpe difference vs EWMA (positive is better)",
        ],
    ):
        inf = tables[key].sort_values(["model", "block_length"])
        for i, row in enumerate(inf.itertuples()):
            # Percentile intervals need not contain the point estimate; plot endpoints directly.
            ax.plot([row.ci_low, row.ci_high], [i, i], color=COLORS[row.model], linewidth=2)
            ax.plot(row.estimate, i, "o", color=COLORS[row.model])
        ax.set_yticks(
            range(len(inf)), [f"{r.model} / block {r.block_length}" for r in inf.itertuples()]
        )
        ax.axvline(0, color="#7A8795", linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
    fig.suptitle("Paired circular block bootstrap | 95% percentile intervals", fontsize=13)
    save(fig, "07_bootstrap_intervals")

    primary["year"] = primary.date.dt.year
    ratio = primary.target_var / primary.pred_var
    primary["loss"] = ratio - np.log(ratio) - 1
    yearly = primary.groupby(["year", "model"]).loss.mean().unstack()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for model in ["HV21", "Ridge", "LightGBM"]:
        ax.plot(
            yearly.index,
            (yearly[model] / yearly.EWMA - 1) * 100,
            "o-",
            color=COLORS[model],
            label=model,
        )
    ax.axhline(0, color=COLORS["EWMA"], linestyle="--", label="EWMA")
    ax.set(
        title="Forecast robustness across calendar years",
        ylabel="QLIKE relative to EWMA (%)\nNegative is better",
        xlabel="Out-of-sample year",
    )
    ax.legend(ncol=4)
    save(fig, "08_yearly_robustness")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    w = tables["weights"]
    for ax, model in zip(axes, ["Ridge", "LightGBM"]):
        sub = w[(w.model == model) & (w.cost_bps == base_cost)]
        wide = sub.pivot(index="date", columns="ticker", values="weight")
        wide.plot(ax=ax, linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set(
            title=f"{model}: end-of-day positions after trading",
            ylabel="Fraction of NAV",
            xlabel="",
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_ylim(-0.65, 0.65)
        ax.legend(ncol=3, fontsize=9, loc="upper left")
        ax.tick_params(axis="x", labelrotation=0)
    save(fig, "09_positions")
    return files
