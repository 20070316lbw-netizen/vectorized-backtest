"""回测评估指标

只负责"根据回测结果算几个诊断数字", 不做任何仓位或收益计算——输入统一
是 engine.py 的输出或者更上游的中间结果(权重、IC 序列), 不重新计算这些
东西。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(nav: pd.Series, periods_per_year: float) -> float:
    """用净值序列算年化夏普比率(无风险利率按 0 处理, 要扣无风险利率的话
    在传入 nav 前自己处理, 这里不做)。

    Args:
        nav: 逐日净值(见 engine.BacktestResult.nav)。
        periods_per_year: nav 一年有多少个观测点(日频数据通常是 252)。

    Returns:
        年化夏普比率; 有效收益率样本少于 2 个或标准差为 0 时返回 NaN。

    Example:
        >>> nav = pd.Series([1.0, 1.02, 1.01, 1.05, 1.03],
        ...                 index=pd.date_range("2024-01-01", periods=5))
        >>> round(sharpe_ratio(nav, periods_per_year=252), 4)
        4.5161
    """
    ret = nav.pct_change().dropna()
    if len(ret) < 2 or ret.std() == 0:
        return float("nan")
    return float(ret.mean() / ret.std() * np.sqrt(periods_per_year))


def max_drawdown(nav: pd.Series) -> float:
    """算最大回撤(<= 0 的数, 比如 -0.23 代表最大回撤 23%)。

    Args:
        nav: 逐日净值。

    Returns:
        最大回撤, <= 0 的 float。

    Example:
        >>> nav = pd.Series([1.0, 1.2, 0.9, 1.1])
        >>> max_drawdown(nav)
        -0.25
    """
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    return float(drawdown.min())


def turnover(
    weights: pd.Series,
    *,
    date_level: str = "date",
    ticker_level: str = "ticker",
) -> pd.Series:
    """算每次调仓相对上一次调仓的单边换手率。

    定义为相邻两次调仓权重变化绝对值之和的一半; 第一次调仓相对"空仓"
    算, 换手率等于当次总敞口的一半。

    Args:
        weights: 调仓日的目标权重(比如 portfolio.quantile_long_short 的
            输出), [date, ticker] MultiIndex, 索引里只有调仓日。
        date_level: 索引里日期所在的 level 名, 默认 "date"。
        ticker_level: 索引里 ticker 所在的 level 名, 默认 "ticker"。

    Returns:
        pd.Series, 索引是调仓日(升序), 值是当次相对上一次的单边换手率。

    Example:
        >>> d0, d1 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-22")
        >>> w = pd.Series(
        ...     {(d0, "A"): 0.5, (d0, "B"): -0.5, (d1, "A"): 0.5, (d1, "C"): -0.5}
        ... )
        >>> w.index = pd.MultiIndex.from_tuples(w.index, names=["date", "ticker"])
        >>> turnover(w)
        date
        2024-01-01    0.5
        2024-01-22    0.5
        Name: turnover, dtype: float64
    """
    w = weights.unstack(ticker_level).fillna(0.0).sort_index()
    prev = w.shift(1).fillna(0.0)
    return ((w - prev).abs().sum(axis=1) / 2.0).rename("turnover").rename_axis(date_level)


def information_ratio(ic: pd.Series) -> float:
    """用滚动 IC 序列算信息比率(IC 均值 / IC 标准差)。

    Args:
        ic: 滚动 IC 序列(见 expected_returns.rolling_ic)。

    Returns:
        信息比率; 有效样本少于 2 个或标准差为 0 时返回 NaN。

    Example:
        >>> ic = pd.Series([0.05, 0.03, 0.07, 0.02, 0.04])
        >>> round(information_ratio(ic), 4)
        2.1835
    """
    ic = ic.dropna()
    if len(ic) < 2 or ic.std() == 0:
        return float("nan")
    return float(ic.mean() / ic.std())
