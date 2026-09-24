"""Refresh README's research section while preserving authored documentation."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPORT_START = "<!-- BEGIN GENERATED RESEARCH -->"
REPORT_END = "<!-- END GENERATED RESEARCH -->"


def _write_research_section(path: Path, research_text: str):
    """Replace only the marked block; never overwrite surrounding README content."""
    block = f"{REPORT_START}\n\n{research_text.strip()}\n\n{REPORT_END}"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing.count(REPORT_START) != 1 or existing.count(REPORT_END) != 1:
            raise ValueError(
                "Expected exactly one generated research marker pair in the Markdown file"
            )
        start = existing.index(REPORT_START)
        end = existing.index(REPORT_END)
        if end < start:
            raise ValueError("Generated research markers are out of order")
        text = existing[:start] + block + existing[end + len(REPORT_END) :]
    else:
        text = (
            "# Machine Learning for Cross-Asset Volatility Forecasting and Portfolio Allocation\n\n"
            + block
            + "\n"
        )
    path.write_text(text, encoding="utf-8")


def _table(frame, decimals=4):
    return frame.to_markdown(index=False, floatfmt=f".{decimals}f")


def write_report(tables, manifest, figures, report_path: Path, output: Path):
    config = manifest["config"]
    research = config["research"]
    portfolio = config["portfolio"]
    inf_cfg = config["inference"]
    annual = research["annualization"]
    horizon = research["horizon"]
    lag = research["execution_lag"]
    n_assets = len(config["data"]["tickers"])
    metrics = tables["forecast_metrics"]
    main = metrics[metrics.feature_set.isin(["baseline", "full"])]
    pooled = main[main.ticker == "Pooled"].set_index("model")
    base_cost = portfolio["base_cost_bps"]
    pm = tables["portfolio_metrics"]
    base = pm[pm.cost_bps == base_cost].set_index("model")
    fi = tables["forecast_inference"]
    si = tables["strategy_inference"]
    primary_fi = fi[fi.block_length == inf_cfg["primary_block"]].set_index("model")
    primary_si = si[si.block_length == inf_cfg["primary_block"]].set_index("model")
    folds = tables["folds"]
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def link(path):
        return Path(os.path.relpath(path, report_path.parent)).as_posix()

    def fig(name, alt):
        return f"![{alt}]({link(output / 'figures' / f'{name}.png')})"

    conclusions = []
    for model in ["Ridge", "LightGBM"]:
        change = (pooled.loc[model, "qlike"] / pooled.loc["EWMA", "qlike"] - 1) * 100
        test = primary_fi.loc[model]
        conclusions.append(
            f"- **{model} full**：汇总 QLIKE 为 {pooled.loc[model, 'qlike']:.4f}，相对 EWMA "
            f"{'降低' if change < 0 else '增加'} {abs(change):.2f}%；主 block 检验的 Holm p={test.p_holm:.4f}。"
            f"{base_cost} bps 下净 Sharpe 为 {base.loc[model, 'sharpe']:.3f}，相对 EWMA 的差异为 "
            f"{primary_si.loc[model, 'estimate']:+.3f}，95% CI "
            f"[{primary_si.loc[model, 'ci_low']:.3f}, {primary_si.loc[model, 'ci_high']:.3f}]。"
        )
    ablation = metrics[
        (metrics.ticker == "Pooled") & metrics.model.isin(["Ridge", "LightGBM"])
    ].copy()
    ablation_notes = []
    for model in ["Ridge", "LightGBM"]:
        a = ablation[ablation.model == model].set_index("feature_set")
        delta = (a.loc["full", "qlike"] / a.loc["no_cross", "qlike"] - 1) * 100
        best = a.qlike.idxmin()
        ablation_notes.append(
            f"{model} 的 full 相对 no_cross QLIKE 变化为 {delta:+.2f}%，三组中 {best} 的点估计最低。"
        )
    improvement_count = sum((primary_fi.estimate < 0) & (primary_fi.p_holm < 0.05))
    if improvement_count:
        finding = f"在预先固定的主比较中，{improvement_count} 个 ML 模型相对 EWMA 的 QLIKE 改善通过 5% Holm 校正检验。"
    else:
        finding = (
            "在预先固定的主比较中，没有 ML 模型同时满足 QLIKE 优于 EWMA 且通过 5% Holm 校正检验。"
        )
    strategy_parts = []
    for model, row in primary_si.iterrows():
        if row.ci_high < 0:
            strategy_parts.append(
                f"{model} 的 Sharpe 差异区间完全低于零，支持其在本实验条件下劣于 EWMA。"
            )
        elif row.ci_low > 0:
            strategy_parts.append(
                f"{model} 的 Sharpe 差异区间完全高于零，支持其在本实验条件下优于 EWMA。"
            )
        else:
            strategy_parts.append(f"{model} 的 Sharpe 差异区间覆盖零，不能证明其优于 EWMA。")
    strategy_finding = " ".join(strategy_parts)
    headline = base.reset_index()[
        ["model", "cagr", "ann_vol", "sharpe", "max_drawdown", "annual_turnover", "avg_gross"]
    ].copy()
    for col in ["cagr", "ann_vol", "max_drawdown", "avg_gross"]:
        headline[col] = headline[col].map(lambda v: f"{v:.2%}")
    headline = headline.rename(
        columns={
            "cagr": "CAGR",
            "ann_vol": "年化波动",
            "sharpe": "净 Sharpe",
            "max_drawdown": "最大回撤",
            "annual_turnover": "年化单边换手倍数",
            "avg_gross": "平均总敞口",
        }
    )
    source_dir = Path(__file__).resolve().parent
    quality = manifest["data"]["quality"]
    first_fold = folds[folds.fold == folds.fold.min()].iloc[0]
    daily = tables["daily"]
    periods = [
        ("2012–2016", "2012-01-01", "2016-12-31"),
        ("2017–2021", "2017-01-01", "2021-12-31"),
        ("2022–2025", "2022-01-01", "2025-12-31"),
    ]
    regime_rows = []
    for name, start, end in periods:
        subset = daily[(daily.cost_bps == base_cost) & (daily.date >= start) & (daily.date <= end)]
        if "is_initial_anchor" in subset:
            subset = subset[~subset.is_initial_anchor]
        for model, g in subset.groupby("model"):
            r = g.net_return
            if len(r) > 1:
                regime_rows.append(
                    {
                        "时期": name,
                        "model": model,
                        "净Sharpe": r.mean() / r.std(ddof=1) * research["annualization"] ** 0.5,
                    }
                )
    regimes = (
        pd.DataFrame(regime_rows)
        .pivot(index="model", columns="时期", values="净Sharpe")
        .reset_index()
    )
    feature_rows = pd.DataFrame(
        [
            {"组别": "risk_only", "有效特征数": 4, "内容": "log RV(5/21/63), log EWMA"},
            {
                "组别": "no_cross",
                "有效特征数": 14,
                "内容": "risk_only + 本资产收益、下行风险、波动变化、回撤、Parkinson 区间波动",
            },
            {
                "组别": "full",
                "有效特征数": 20,
                "内容": "no_cross + 另外两类资产各3项：log RV21、5日收益、63日相关性",
            },
        ]
    )
    text = f"""# Machine Learning for Cross-Asset Volatility Forecasting and Portfolio Allocation

**独立量化研究项目｜已实现并完成历史样本外实验｜研究版 v0.1**

本报告由 `crossvol run` 从实际 CSV 结果自动生成；修改数据或参数后应完整重跑，不手工填写绩效。生成时间：{manifest["created_utc"]}。研究目的：检验机器学习能否改善跨资产波动率预测，以及预测是否能改善相同趋势信号下的仓位配置。所有结果均为历史模拟。

## 1. 核心结果与研究判断

{chr(10).join(conclusions)}

{finding} {strategy_finding}

本项目的可展示价值是可审计的样本外研究链条：真实数据、严格时间边界、基准对照、有限调参、消融、真实持仓会计和依赖结构下的推断。预测误差、组合收益与统计显著性分别报告；不预设复杂模型一定获胜。

## 2. 数据、范围与可复现性

| 项目 | 实际设置 |
|---|---|
| 标的 | SPY（美国股票）、TLT（长期美国国债）、GLD（黄金） |
| 原始数据 | Yahoo Finance 经 yfinance 下载的日频 OHLCV；auto_adjust=True，repair=False |
| 价格范围 | {quality["start"]} 至 {quality["end"]}，{quality["n_dates"]:,} 个交易日、{quality["n_rows"]:,} 行 |
| 预测样本外 origin | {manifest["oos_origin_start"]} 至 {manifest["oos_origin_end"]}，{manifest["n_origins"]:,} 个共同 origin |
| 组合回测日期 | {pd.to_datetime(base.iloc[0]["start"]).date()} 至 {pd.to_datetime(base.iloc[0]["end"]).date()}（起始全现金锚点不进入收益矩） |
| 外层 / 内层 | {folds.fold.nunique()} 个年度外层折；每折 {research["validation_splits"]} 个 {research["validation_size"]} 日验证块 |
| 数据下载时间 UTC | {manifest["data"]["retrieved_utc"]} |
| 随机种子 / 年化 | {research["seed"]} / {research["annualization"]} |
| 数据质量 | 共同交易日缺失 {quality["missing_cells"]}，重复 {quality["duplicate_rows"]}，零成交量 {quality["zero_volume_rows"]} |

数据包含 {quality["calendar_gaps_over_4_days"]} 个超过4个自然日的相邻日期间隔；不将周末、节假日和停市自动填成零收益。校验还检查 OHLC 正值、价格上下界、共同日期以及异常日收益。没有用模拟数据替代真实行情。

数据 SHA-256：`{manifest["data"]["sha256"]}`。完整下载参数见 [data_manifest.json]({link(output / "data_manifest.json")})；配置、包版本、源码及结果哈希见 [run_manifest.json]({link(output / "run_manifest.json")})。缓存读取先校验哈希，线上数据修订必须显式 `--refresh`；首次运行也从落盘快照重新读入，避免序列化前后差异。

价格用拆股和分红复权后的序列计算收益。复权价格是总回报近似，不是当日可直接成交的报价；空头的分红负担相应体现在总回报方向中。Yahoo 快照不是历史时点版本数据库，不能声称消除了所有供应商修订问题。数据接口参数以 [yfinance 官方说明](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html) 为准。

{fig("01_market_context", "跨资产市场背景与历史波动率")}

## 3. 目标变量与执行时间：先定义可用信息

令 $P_{{i,t}}$ 为资产 $i$ 在交易日 $t$ 的复权收盘价，$r_{{i,t}}=\\log(P_{{i,t}}/P_{{i,t-1}})$。在 $t$ 日收盘后形成特征和预测；$t+{lag}$ 收盘执行；新仓位从 $t+{lag + 1}$ 日的 close-to-close 收益开始获利或亏损。

默认 $h={research["horizon"]}$、成交延迟 $d={research["execution_lag"]}$：

$$y_{{i,t}} = \\frac{{{annual}}}{{{horizon}}}\\sum_{{k={lag + 1}}}^{{{lag + horizon}}}r_{{i,t+k}}^2,\\qquad \\sigma^{{realized}}_{{i,t}}=\\sqrt{{y_{{i,t}}}}.$$

这是未来{horizon}个交易日的**平方日收益波动率代理**，没有减去样本均值；不是高频 realized variance，也不是可直接观测的潜在条件方差。标签最后需要 $t+{lag + horizon}$ 的价格，因此 `label_end=t+{lag + horizon}`，最后{lag + horizon}个 origin 无完整标签，全部从预测评分中排除。

```mermaid
flowchart LR
  A[Close t: features and forecast] --> B[Close t+{lag}: execute target]
  B --> C[Returns t+{lag + 1} through t+{lag + horizon}]
  C --> D[Close t+{lag + horizon}: target fully observable]
```

特征只使用 origin 及此前数据。用特征时间戳过滤训练集还不够；训练时同时要求 `label_end < next_boundary`。组合用简单收益 $P_t/P_{{t-1}}-1$ 做精确净值递推，不把对数收益直接当作资金收益。

## 4. 特征、模型与有限调参

{_table(feature_rows, 0)}

no_cross 的具体扩展项：1/5/21/63日累计对数收益、当日绝对收益、21日下行平方收益均值的对数、RV5/RV63、21日历史波动率序列的21日标准差、63日回撤、21日 Parkinson 高低价区间波动率的对数。特征定义均在 [features.py]({link(source_dir / "features.py")})。

full 在代码中使用{len(manifest["features"]["full"])}列统一 schema；其中本资产的3个 cross 通道固定为零，每个模型有{len(manifest["features"]["full"]) - 3}项有效特征。这样避免重复本资产特征改变 Ridge 的有效 L2 惩罚，使 full/no_cross 更清楚地比较其它资产的信息。full 与 no_cross 各自仅在内层选择超参数；比较衡量整个受控训练流程的差异，不能把差异解释为某单一特征的因果贡献。

| 模型 | 拟合目标与设置 | 超参数候选 |
|---|---|---|
| HV21 | 最近21日平方收益均值 ×{annual} | 固定21日，无样本外调参 |
| EWMA | $v_t={research["ewma_lambda"]:.2f}v_{{t-1}}+{1 - research["ewma_lambda"]:.2f}r_t^2$，年化后预测未来方差；至少63日预热 | 固定 λ={research["ewma_lambda"]} |
| Ridge | StandardScaler + L2 线性回归，拟合 log 方差 | α={research["ridge_alphas"]} |
| LightGBM | 浅树梯度提升，平方误差拟合 log 方差，learning_rate=0.05 | {len(research["lgbm_grid"])}个固定组合，见下表 |

{_table(pd.DataFrame(research["lgbm_grid"]))}

每个资产分别拟合；“跨资产”指输入其它资产风险状态，不是把三只 ETF 的样本行随机混合。两类 ML 使用相同的候选数及验证时间预算，不做大规模搜索，不用测试集早停。Ridge 的缩放器仅在每个训练折拟合。LightGBM 固定随机种子、单线程、`deterministic=True`、`force_col_wise=True`，但跨平台或库版本仍可能存在差异，参见 [LightGBM 官方参数](https://lightgbm.readthedocs.io/en/stable/Parameters.html)。

为把 log 预测转换为算术方差，使用训练残差 smearing 因子：

$$\\widehat y_t=\\exp(\\widehat f(X_t))\\times\\frac1n\\sum_{{j\\in train}}\\exp(\\log y_j-\\widehat f(X_j)).$$

因子从当前拟合的训练残差估计，不读取验证或测试标签。全局 smearing 只是偏差修正近似，无法保证所有条件状态下的方差校准；树模型的训练内残差可能低估样本外残差分布。所有预测统一截断在 $[10^{{-8}},4]$ 年化方差内；这些固定数值保护不根据测试结果选择。

## 5. 嵌套 walk-forward 与防泄漏审计

每年年初进行一次模型重估和内层调参，该年预测使用固定参数和当日可得特征。历史训练集逐年扩展。以前年度的测试标签在实际时间上完整可得后，可以进入后续年度训练；这符合在线研究的时间顺序。首个外层训练 origin 为 {pd.to_datetime(first_fold.train_start).date()} 至 {pd.to_datetime(first_fold.train_end).date()}，其最大标签终点为 {pd.to_datetime(first_fold.train_max_label_end).date()}，严格早于测试开始 {pd.to_datetime(first_fold.test_start).date()}。

1. 对测试年度边界 $T$，仅允许 origin $<T$ 且 `label_end < T` 的训练样本。
2. 从可用历史中向前构造{research["validation_splits"]}个{research["validation_size"]}日验证块；每个训练/验证边界同样按标签终点清除重叠，验证标签还必须早于下一块/测试边界。默认等价于边界前{lag + horizon}个 origin 被剔除。
3. 每个候选在全部验证块上计算 QLIKE，取等权平均最低者；不查看外层测试成绩。
4. 用选中配置重新拟合所有当时可用训练数据，预测该年度；保存每个候选、训练区间、验证区间、所选参数及 smearing。
5. 所有模型、资产和消融组使用完全相同的有效预测日期。累计生成 {manifest["n_predictions"]:,} 条预测、{len(folds)} 个 ML 外层拟合及 {len(tables["tuning"])} 条候选/验证审计记录。

`folds.csv` / `tuning.csv` 可逐行检查 `train_max_label_end < validation_start/test_start`。这种设计与 [scikit-learn 时间序列验证的时间顺序及 gap 原则](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) 一致，本项目另外显式使用标签终点做 purge。

当前样本外定义是**相对每次模型拟合的时间样本外**。研究者已查看整个实验输出，2023–2025 年也属于滚动实验，不声称是永久锁定、从未看过的最终 holdout。继续尝试新方法会增加研究者选择偏差；后续确认应另锁定新的未来区间。

## 6. 预测评分与实际结果

主评分使用方差的 QLIKE：

$$L(y,\\widehat y)=\\frac{{y}}{{\\widehat y}}-\\log\\left(\\frac{{y}}{{\\widehat y}}\\right)-1.$$

越低越好；同时报告年化波动率 RMSE/MAE 以及 `mean(target variance)/mean(predicted variance)` 校准比，理想为1。Pooled 对同一天资产等权，再对日期等权；不把三倍资产行数误当成独立样本量。选择 QLIKE 的动机来自 [Patton (2011) 关于噪声波动率代理下预测比较的研究](https://public.econ.duke.edu/~ap172/Patton_robust_JoE_forthcoming.pdf)；其稳健性依赖代理和条件假设，不能自动解释成真实潜在波动率的无偏排名。

{_table(main[["ticker", "model", "qlike", "vol_rmse", "vol_mae", "calibration"]])}

RMSE/MAE 以小数年化波动率为单位，0.01=1个百分点。预测图仅截取2019年以后便于阅读；指标使用完整样本外区间。

{fig("02_forecasts", "2019年后逐资产预测与未来波动率代理")}

{fig("03_forecast_metrics", "按资产比较预测误差")}

## 7. 特征消融与年度稳定性

{_table(ablation[["model", "feature_set", "qlike", "vol_rmse", "calibration"]])}

{" ".join(ablation_notes)} 这些是完整样本外结果的描述性比较；没有据此替换 full 主模型，也未为全部消融比较提供多重检验后的显著性结论。训练期特征重要性保存在 `importance.csv`：Ridge 为标准化系数绝对值，LightGBM 为 gain；均不是因果解释，也不是额外的样本外证据。

{fig("04_feature_ablation", "按模型与资产的特征消融结果")}

{fig("08_yearly_robustness", "逐年QLIKE相对EWMA的变化")}

年度图与 [yearly_forecast_metrics.csv]({link(output / "tables" / "yearly_forecast_metrics.csv")}) 用于检查平均值是否由少数年份驱动。疫情、通胀及利率变化属于解释背景，未作为事后筛选“好年份”的规则。

## 8. 趋势策略与仓位会计

固定趋势方向：$s_{{i,t}}=\\operatorname{{sign}}(P_{{i,t}}/P_{{i,t-{portfolio["trend_window"]}}}-1)$。该规则只隔离波动率预测的仓位作用，不训练收益方向模型。它借鉴时间序列趋势思想，但不是 [Moskowitz, Ooi & Pedersen (2012)](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum) 的12个月期货策略复现。

$$\\widetilde w_{{i,t}}=s_{{i,t}}\\frac{{{portfolio["target_vol"]:.2f}}}{{\\sqrt{{{n_assets}}}\\max(\\widehat\\sigma_{{i,t}},{portfolio["vol_floor"]:.2f})}}.$$

先将每个资产权重限制为 ±{portfolio["asset_cap"]:.0%}，再按比例缩放使总绝对权重不超过 {portfolio["gross_cap"]:.0%}。10% 是忽略相关性的等风险预算参数，截断和相关性意味着实际组合波动率**不保证为10%**。上限在成交时施加；两次成交之间及最后尾部持仓可因价格漂移越界。EqualWeightTrend 使用同一趋势方向，方向非零时各资产绝对权重1/3，方向恰为零则空仓，作为不用动态波动率预测的控制组。

这是用五日平均方差预测作为每日仓位的平滑风险估计，不声称它已经被验证为下一日条件方差预测。每天按实际持仓更新损益，再成交；交易量来自目标与**漂移后**持仓之差。为避免“先用全额资金买入、再扣费”造成隐性杠杆，用以下自融资方程求扣费后 NAV 比例 $z$：

$$z=1-c\\sum_i|z w_i^*-w_i^-|,$$

其中 $w^-$ 用交易前 NAV 计量，$w^*$ 用交易后 NAV 计量，$c=\\mathrm{{bps}}/10000$。首次入场及最终平仓都收费，单边换手为全部买卖绝对金额之和 / 交易前 NAV，不再除以2。持仓做空时，每年 {portfolio["short_borrow_bps"]} bps 借券费按前一日空头名义额每日计提。组合现金收益假设为0，未额外采用真实无风险利率序列；本报告 Sharpe 是以零现金收益为基准的扣费后 Sharpe。

预测 origin 最后一天为 {manifest["oos_origin_end"]}，最后目标延迟一天执行，之后不人为生成新预测、不重复调仓，持仓自然漂移至最后标签终点后平仓。因此尾部仍有价格收益，但没有不完整标签的预测评分。

净收益包含总回报价格损益、借券费及交易费。默认成本情景为 {portfolio["cost_bps"]} bps；**0 bps 只代表零交易费，仍收借券费**。未另计资金规模相关市场冲击、税费、ETF 冲击成本非线性、卖空可得性和保证金分账。复权收盘价与常数成本是研究简化，不代表成交仿真。

## 9. 组合表现、成本与阶段结果

基准情景：{base_cost} bps 单边交易费 + {portfolio["short_borrow_bps"]} bps 年化借券费。

{_table(headline, 3)}

CAGR 按{annual}交易日年化；Sharpe 为日均净收益 / 日收益标准差 ×√{annual}；最大回撤相对历史净值高点；年化单边换手倍数包含入场和平仓。`total_cost` 在 CSV 中为每日归一化费用拖累的加总，不等于终值财富损失。完整情景见 [portfolio_metrics.csv]({link(output / "tables" / "portfolio_metrics.csv")})。

{fig("05_portfolio_performance", "扣费后策略净值与最大回撤路径")}

{fig("06_cost_sensitivity", "交易费敏感性")}

下表为固定连续区间的净 Sharpe 描述性拆分；沿用完整回测持仓，区间边界不额外清仓，不是重新调参的子策略。

{_table(regimes, 3)}

{fig("09_positions", "实际收盘后持仓权重")}

## 10. 保留时间依赖的 block-bootstrap 推断

重叠5日标签使损失存在序列相关；多个资产同日也相关。本项目使用**配对 circular block bootstrap**：对共同日期抽取连续区块并允许末尾环绕，所有资产与模型共享同一抽样日期，先在资产间等权，再比较模型，不能把每个资产独立重采样。方法背景可见 [arch 的时间序列 bootstrap 说明](https://arch.readthedocs.io/en/stable/bootstrap/timeseries-bootstraps.html)，本仓库直接用 NumPy 实现。

固定 {inf_cfg["n_bootstrap"]:,} 次重采样，block 长度 {inf_cfg["block_lengths"]}，预设主长度 {inf_cfg["primary_block"]}。波动率比较统计量是 `mean(QLIKE_model − QLIKE_EWMA)`，负值更好。报告95% percentile CI；检验用居中到零假设的 bootstrap 分布计算双侧 p，并在同一长度下对 Ridge/LightGBM 两项比较做 Holm 校正。不同 block 是稳健性分析，不能事后选择显著的一档。CI 本身不是同时置信区间。

{_table(fi[["model", "block_length", "estimate", "ci_low", "ci_high", "p_value", "p_holm"]])}

策略推断在 {base_cost} bps 净收益的相同日期上配对抽样，排除起始人为全现金锚点，重算 Sharpe_model − Sharpe_EWMA。报告 percentile CI；不把普通 bootstrap 的正负比例包装为正式显著性 p 值。

{_table(si[["model", "block_length", "estimate", "ci_low", "ci_high"]])}

{fig("07_bootstrap_intervals", "配对区块重采样预测损失和Sharpe差异置信区间")}

区块推断只近似处理一定长度内的依赖；全样本结构变化可能破坏平稳性。区间以本次已拟合预测、既定数据和规则为条件，不重新训练模型，不覆盖模型选择、数据供应商误差及所有研究者尝试的不确定性。Holm 只覆盖上述预设两模型主比较，未覆盖消融、年份及其它潜在尝试。

## 11. 结论、局限与下一步

{finding} {strategy_finding} {" ".join(ablation_notes)}

预测误差较低并不必然增加策略 Sharpe。以本次结果为例，{base_cost} bps 下 EWMA / Ridge / LightGBM 的年化单边换手分别为 {base.loc["EWMA", "annual_turnover"]:.2f} / {base.loc["Ridge", "annual_turnover"]:.2f} / {base.loc["LightGBM", "annual_turnover"]:.2f} 倍；更复杂的预测未自动带来较低交易频率。0 bps 情景仍可检查不含交易费的相对表现，从而避免把所有差异都归因于成本。固定趋势方向、风险预算截断、相关性变化、预测与持有期的匹配也会影响经济结果；这些机制在本实验中没有被单独做因果识别。比较模型时应同时看 QLIKE、校准、分资产/分年稳定性、回撤和成本后收益，不依据净值最高的一条曲线宣布成功。

本实验的主要边界如下：

- 仅三个事后确定的高流动性存续 ETF，无法推广到整个股票、债券及商品市场；TLT 代表长期国债风险，不是所有 Treasury 久期。
- Yahoo 复权日线缺乏历史时点版本与日内成交细节；日收益平方标签噪声大。Parkinson 特征受跳跃和隔夜信息未纳入区间的影响。
- 参数每年更新一次，强突变期可能滞后；log 回归 + 全局 smearing 的均值修正不保证条件校准。没有额外加入 GARCH/HAR 作为更强结构基准，HV21/EWMA 仅是本版基准集合。
- 现金收益假设0，净 Sharpe 未扣真实逐日无风险利率；借券费固定，不含保证金、可借约束、冲击与容量。该简化会影响收益水平及不同现金/空头比例策略之间的比较。
- 10% 是仓位预算参数；没有显式估计组合协方差，也没有承诺实现某一目标波动。交易费和隔夜执行模型不能替代真实订单仿真。
- 没有永久锁定的独立最终 holdout；所有现有样本外图表都已公开检查。Bootstrap 不是对未来盈利的保证。

后续有价值的扩展是：在新的未见区间先冻结实验计划；加入不同久期债券和更多资产；用有许可证、可追溯的历史版本数据及日内 realized variance；比较 HAR/GARCH；使用训练期残差的时间样本外校准；加入现金利率、真实借券与冲击估计；再检验协方差感知的组合风险配置。扩展应逐项验证，不能持续搜索直至找到显著结果。

## 12. 工程验证与运行方法

测试覆盖以下失效模式：未来价格扰动改变过去特征、标签错位、标签边界重叠、测试标签影响模型选择、缩放器使用测试数据、模型随机性、样本不一致、成交日期提前、忽略漂移换手、入场/平仓漏费、卖空费用漏计、净值不自融资、把资产复制当作更多独立样本、Bootstrap 不配对、以及 Sharpe 锚点口径不一致。网络下载不进入 CI 的单元测试。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
# macOS LightGBM 若缺少 OpenMP: brew install libomp
crossvol run                   # 首次下载并完整运行
crossvol run --offline         # 使用经过 SHA-256 校验的同一数据快照
crossvol run --offline --reuse-forecasts  # 仅当代码/配置/数据和预测文件哈希匹配
crossvol report                # 从已有结果重建图表及此报告
pytest -q
ruff check src tests scripts
python scripts/verify_artifacts.py
```

原始行情和大体积逐日输出保存在本地并默认不入 Git；汇总表、PNG/SVG 图、配置、清单和本报告可作为 GitHub 展示材料。数据源的使用和再分发条款独立于本仓库代码许可证；复现下载可能受到服务限流与数据修订影响。固定本地快照可以精确重跑本次数据，重新下载只保证方法可复现，不能保证未来供应商返回的价格完全一致。

关键入口：[特征]({link(source_dir / "features.py")}) · [Walk-forward]({link(source_dir / "walkforward.py")}) · [组合会计]({link(source_dir / "portfolio.py")}) · [统计推断]({link(source_dir / "statistics.py")}) · [运行入口]({link(source_dir / "cli.py")})。
"""
    # The document has one top-level title; the full report is an anchored section.
    body = text.split("\n", 1)[1].lstrip()
    body = "\n".join("#" + line if line.startswith("## ") else line for line in body.splitlines())
    _write_research_section(
        report_path, '<a id="research-report"></a>\n\n## 中文研究报告\n\n' + body
    )
