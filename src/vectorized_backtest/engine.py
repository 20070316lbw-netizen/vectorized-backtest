"""组合净值与持仓的向量化演算

只负责"给定调仓日的目标权重, 组合怎么按买入持有的方式逐日漂移", 不负责
仓位怎么算出来(portfolio.py 的事)、也不负责算完之后怎么评估(metrics.py
的事)。非调仓日不交易、不重新计算目标权重, 组合就按调仓日那天的权重
换算出份额, 一直持有到下一个调仓日, 期间的市值随价格自然漂移。

核心是一段"分段买入持有"的复利: 每个调仓区间内, 每只票相对区间起点的
价格比率(P_t / P_起点)可以整段一次性算(groupby + transform), 乘以
起点权重再逐日横截面求和就是这一区间"相对区间起点"的组合盈亏比例, 这
一步不需要逐日循环。区间与区间之间, 下一区间的起始资产 = 上一区间的
期末资产, 这个"资产接力"只发生在调仓日这个小数组上(比如 5 年数据也就
几十个调仓日), 用 cumprod 一次性算完——整个引擎里唯一算得上"链式依赖"
的地方就是这几十个元素的 cumprod, 不是遍历全部交易日。

多空组合的净值不能通过"逐日加总每只票的持仓市值"得到(dollar-neutral
的书, 多空市值加总后只剩下净盈亏、丢了本金这一项), 必须用"本金 ×
(1 + 累计盈亏比例)"这个口径, 这里的 nav 就是这么算的; 持仓市值
(daily_value)、份额(shares)是另一套东西, 只用来报告敞口, 不能拿去
加总当 nav 用。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from vectorized_backtest.rebalance import rebalance_block


@dataclass
class BacktestResult:
    """回测结果。

    Attributes:
        nav: 逐日组合净值, pd.Series, 索引是日期, 第一个调仓日当天等于
            initial_capital(口径见模块顶部说明, 用累计盈亏比例算, 不是
            持仓市值加总)。
        shares: 每个调仓日决定的持仓份额, [date, ticker] MultiIndex 的
            pd.Series(date 只包含调仓日), 只保留权重不为 0 的持仓, 在
            下一次调仓之前份额不变(可能为负, 代表空头)。
        daily_value: 每天每只持仓票的市值(份额 x 当天价格), [date, ticker]
            MultiIndex 的 pd.Series, 覆盖全部交易日(不只调仓日), 只保留
            权重不为 0 的持仓; 报告敞口用, 不能加总当 nav 用。
    """

    nav: pd.Series
    shares: pd.Series
    daily_value: pd.Series


def run(
    weights: pd.Series,
    prices: pd.DataFrame,
    freq: int,
    *,
    initial_capital: float = 1.0,
    price_col: str = "adj_close",
    date_level: str = "date",
    ticker_level: str = "ticker",
    log_rebalances: bool = True,
) -> BacktestResult:
    """按"调仓日定权重、期间买入持有"的方式跑一遍组合净值演算。

    Args:
        weights: 调仓日的目标权重(比如 portfolio.quantile_long_short 或
            portfolio.mean_variance 的输出), [date, ticker] MultiIndex,
            只需要包含调仓日那些日期; 多空组合权重可以正可以负、不要求
            按行求和为 1。
        prices: [date, ticker] MultiIndex 价格表, 必须覆盖 weights 里
            出现的调仓日、以及之后到下次调仓为止的所有交易日; 只用到
            weights 里出现过的那些 ticker(其余的会被忽略)。
        freq: 调仓间隔(交易日数), 必须跟算 weights 时用的调仓频率一致,
            用来确定每天属于哪个调仓区间(见 rebalance.rebalance_block)。
        initial_capital: 起始资产, 默认 1.0(净值当成"每 1 元本金"看)。
        price_col: 见 returns.daily_returns。
        date_level: 索引里日期所在的 level 名, 默认 "date"。
        ticker_level: 索引里 ticker 所在的 level 名, 默认 "ticker"。
        log_rebalances: True(默认)时, 每个调仓日用 loguru 打印一次
            当天的组合净值和持仓明细。

    Returns:
        BacktestResult, 见其 docstring。

    Raises:
        ValueError: weights 里的日期不是按 freq 从 prices 日历推出来的
            调仓日子集, 说明传入的 freq 跟算 weights 时不一致。
    """
    weights = weights.dropna()
    weights = weights[weights != 0]
    held_tickers = weights.index.get_level_values(ticker_level).unique()

    prices = prices.sort_index()
    dates = prices.index.get_level_values(date_level).unique().sort_values()
    block = rebalance_block(dates, freq)
    rebal_dates = pd.DatetimeIndex(sorted(block.unique()))

    weight_dates = pd.DatetimeIndex(weights.index.get_level_values(date_level).unique())
    if not weight_dates.isin(rebal_dates).all():
        raise ValueError("weights 里的日期不是按 freq 推出来的调仓日, 检查 freq 是否一致")

    panel = (
        prices.loc[prices.index.get_level_values(ticker_level).isin(held_tickers), [price_col]]
        .rename(columns={price_col: "price"})
        .reset_index()
        .rename(columns={date_level: "date", ticker_level: "ticker"})
    )
    panel["block"] = panel["date"].map(block)
    panel = panel.sort_values(["ticker", "date"])
    panel["ratio"] = panel.groupby(["ticker", "block"])["price"].transform(lambda s: s / s.iloc[0])

    w = weights.rename("weight").reset_index().rename(
        columns={date_level: "block", ticker_level: "ticker"}
    )
    panel = panel.merge(w, on=["block", "ticker"], how="left")
    panel["weight"] = panel["weight"].fillna(0.0)
    panel["contrib"] = panel["weight"] * (panel["ratio"] - 1.0)

    within_block_pnl = panel.groupby("date")["contrib"].sum().sort_index()
    block_end_pnl = within_block_pnl.groupby(block.reindex(within_block_pnl.index)).last()
    block_end_pnl = block_end_pnl.reindex(rebal_dates).fillna(0.0)

    growth = (1.0 + block_end_pnl).cumprod()
    block_start_nav = initial_capital * growth.shift(1, fill_value=1.0)

    nav_multiplier = block.reindex(within_block_pnl.index).map(block_start_nav)
    nav = (nav_multiplier * (1.0 + within_block_pnl)).rename("nav").sort_index()

    panel["block_start_nav"] = panel["block"].map(block_start_nav)
    panel["position_value"] = panel["weight"] * panel["block_start_nav"] * panel["ratio"]
    held = panel[panel["weight"] != 0].copy()

    daily_value = (
        held.set_index(["date", "ticker"])["position_value"]
        .sort_index()
        .rename_axis(index=[date_level, ticker_level])
    )

    on_rebalance = held[held["date"] == held["block"]].copy()
    on_rebalance["shares"] = on_rebalance["weight"] * on_rebalance["block_start_nav"] / on_rebalance["price"]
    shares = (
        on_rebalance.set_index(["date", "ticker"])["shares"]
        .sort_index()
        .rename_axis(index=[date_level, ticker_level])
    )

    if log_rebalances:
        for rb_date in rebal_dates:
            if rb_date not in shares.index.get_level_values(date_level):
                continue
            rb_nav = block_start_nav.get(rb_date, np.nan)
            book = shares.xs(rb_date, level=date_level)
            logger.info(
                "调仓 {}: 净值 {:.4f}, 多头 {} 只, 空头 {} 只",
                rb_date.date(),
                rb_nav,
                int((book > 0).sum()),
                int((book < 0).sum()),
            )

    return BacktestResult(nav=nav, shares=shares, daily_value=daily_value)
