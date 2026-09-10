"""仓位构造: 把 (调仓日的) 期望收益/分数变成组合权重

统一的函数签名都是"(μ 或 score, ...) -> weights", weights 是
[date, ticker] MultiIndex 的 pd.Series, 只在调仓日那些日期上有值——
非调仓日的持仓怎么随价格漂移, 是 engine.py 的事, 这里只负责"调仓那一刻
该配多少仓位"。多空是这里唯一支持的方向(项目里定下的设计: 多空刚好
配合以后要做的资金分散/错峰调仓), 不提供纯多头选项。

quantile_long_short 是纯向量化实现(按日期分组、分位数排序), 不需要
逐日循环。mean_variance 需要给每个调仓日单独求解一次带约束的优化问题,
没法跨日期向量化, 但调仓日数量远小于交易日数量(比如 5 年数据也就几十
次), 循环这几十次不是性能问题——目前还没定用哪个优化器/哪些约束
(dollar-neutral 的严格程度、杠杆上限、单票权重上限、协方差估计窗口都
还没定), 先占位。
"""
from __future__ import annotations

import pandas as pd


def quantile_long_short(
    score: pd.Series,
    n_quantiles: int,
    *,
    date_level: str = "date",
) -> pd.Series:
    """按分数把每个调仓日的横截面分成 n_quantiles 组, 最高组等权做多、
    最低组等权做空, 中间组权重为 0, 多空两腿各自内部等权且总敞口相等
    (多头权重和为 +1, 空头权重和为 -1)。

    Args:
        score: 调仓日的横截面分数(比如 expected_returns.ic_scaled_alpha
            或者 combine.combine_scores 的输出), [date, ticker] MultiIndex,
            只需要包含调仓日。
        n_quantiles: 分组数, 比如 5(五分位, 最高最低各 20% 做多空)。
        date_level: 索引里日期所在的 level 名, 默认 "date"。

    Returns:
        组合权重, 与 score(去掉 NaN 后)同索引的 pd.Series, 中间组为 0、
        多头组为 +1/多头组内股票数、空头组为 -1/空头组内股票数。某天
        有效股票数不够分 n_quantiles 组时, 该天没有输出(不产生 0 行)。

    Raises:
        ValueError: n_quantiles 小于 2(至少要有多空两组)。
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles 至少为 2, 才能分出多头组和空头组")

    def _weights(s: pd.Series) -> pd.Series:
        if len(s) < n_quantiles:
            return pd.Series(dtype=float)
        bucket = pd.qcut(s, n_quantiles, labels=False, duplicates="drop")
        top, bottom = bucket.max(), bucket.min()
        w = pd.Series(0.0, index=s.index)
        long_mask = bucket == top
        short_mask = bucket == bottom
        w[long_mask] = 1.0 / long_mask.sum()
        w[short_mask] = -1.0 / short_mask.sum()
        return w

    score = score.dropna()
    return score.groupby(level=date_level, group_keys=False).apply(_weights)


def mean_variance(
    mu: pd.Series,
    returns_panel: pd.DataFrame,
    **kwargs: object,
) -> pd.Series:
    """用均值-方差优化在每个调仓日求解一次组合权重(留作后续实现)。

    大致思路已经定了: 对每个调仓日, 用 returns_panel 里该日之前的一段
    历史(局部 .unstack() 成 date x ticker 宽表)估协方差矩阵, 配合 mu
    (见 expected_returns.ic_scaled_alpha)求解带约束的优化问题, 结果
    stack 回 [date, ticker] 权重。用哪个优化器(scipy.optimize 还是
    cvxpy)、具体约束(dollar-neutral 的严格程度、杠杆上限、单票权重
    上限、协方差估计窗口)还没定, 先占位。

    Args:
        mu: 调仓日的期望收益(见 expected_returns.ic_scaled_alpha),
            [date, ticker] MultiIndex, 只需要包含调仓日。
        returns_panel: 逐日收益率(见 returns.daily_returns), 用来给
            每个调仓日估协方差矩阵, [date, ticker] MultiIndex。
        **kwargs: 优化器的其余参数(约束、协方差窗口等), 待定。

    Returns:
        组合权重, [date, ticker] MultiIndex 的 pd.Series。

    Raises:
        NotImplementedError: 还没实现。
    """
    raise NotImplementedError("MPT 优化器还没定(求解器/约束), 先占位")
