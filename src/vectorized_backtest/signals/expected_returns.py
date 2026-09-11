"""把无量纲的综合分数换算成收益率量纲的期望收益(Grinold-Kahn IC 打分模型)

μ = IC × σ × score: score 是横截面标准化后的综合分数(combine.py 的输出,
均值 0、标准差 1, 无量纲); σ 是个股收益波动率(协方差矩阵的对角线, 复用
不额外算); IC 是这个综合因子历史上的信息系数(每天的分数和对应的前瞻
收益之间的横截面相关系数, 在一个滚动窗口上取均值)。

IC 和交易信号本身不是同一个采样频率: 仓位只在调仓日变(见 rebalance.py),
但 IC 需要足够多的样本点才能估得稳, 只用调仓日的分数去估 IC 几年也就
几十个点、噪声太大, 所以这里的 IC 用每天的横截面分数 vs 未来 horizon
天收益去估(样本量从几十变成几百上千), 即使相邻两天的样本因为前瞻窗口
重叠而不独立——用重叠样本换更稳的估计, 是这里做出的取舍。μ 最终只在
调仓日产出(传入的 score 只需要是调仓日那天的横截面分数), IC 和 σ 各自
按自己的频率滚动估、算到调仓日当天为止, 这里不做任何决定窗口长度的
默认值——多长的滚动窗口是需要单独想清楚的建模选择, 调用方必须显式传。
"""
from __future__ import annotations

import pandas as pd


def rolling_ic(
    daily_scores: pd.Series,
    fwd_returns: pd.Series,
    window: int,
    *,
    date_level: str = "date",
    method: str = "spearman",
) -> pd.Series:
    """按天算横截面 IC(分数与前瞻收益的截面相关系数), 再滚动平均。

    Args:
        daily_scores: 每天都算出来的横截面综合分数, [date, ticker]
            MultiIndex(不是只在调仓日算的那份, 见模块顶部说明)。
        fwd_returns: 前瞻收益(见 returns.forward_returns), 索引跟
            daily_scores 对齐, 必须用同一个 horizon(调仓间隔)算出来。
        window: 滚动平均的窗口长度, 按"有截面 IC 的交易日"数算(不是
            调仓周期数), 没有默认值, 需要显式传。
        date_level: 索引里日期所在的 level 名, 默认 "date"。
        method: 截面相关系数的算法, "spearman"(秩相关, 默认, 对分数的
            具体数值分布不敏感)或 "pearson"。spearman 在这里是手动先
            按截面 rank 再算 pearson 实现的(数学上等价于直接算秩相关),
            不通过 pandas 内置的 method="spearman"——那条路径要装 scipy,
            而这个项目目前不需要为了这一个相关系数多引入一个依赖。

    Returns:
        pd.Series, 索引是日期(daily_scores 里出现过的日期, 升序), 值是
        滚动 window 期的平均截面 IC; 前面窗口不够的日期为 NaN。

    Raises:
        ValueError: method 不是 "spearman" 或 "pearson"。

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> dates = pd.date_range("2024-01-01", periods=15, freq="D")
        >>> tickers = [f"T{i}" for i in range(5)]
        >>> idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        >>> score = pd.Series(rng.normal(size=len(idx)), index=idx)
        >>> fwd = 0.8 * score + pd.Series(rng.normal(size=len(idx)), index=idx)
        >>> rolling_ic(score, fwd, window=5).tail(3)
        date
        2024-01-13    0.66
        2024-01-14    0.66
        2024-01-15    0.76
        dtype: float64
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method 必须是 'spearman' 或 'pearson', 收到 {method!r}")

    paired = pd.DataFrame({"score": daily_scores, "fwd_return": fwd_returns})

    def _cross_sectional_corr(g: pd.DataFrame) -> float:
        a, b = g["score"], g["fwd_return"]
        if method == "spearman":
            a, b = a.rank(), b.rank()
        return a.corr(b)

    daily_ic = paired.groupby(level=date_level).apply(_cross_sectional_corr)
    return daily_ic.sort_index().rolling(window).mean()


def rolling_sigma(
    returns: pd.Series,
    window: int,
    *,
    ticker_level: str = "ticker",
) -> pd.Series:
    """按 ticker 分组算滚动收益波动率(标准差), 索引不变。

    Args:
        returns: 逐日收益率(见 returns.daily_returns), [date, ticker]
            MultiIndex。
        window: 滚动窗口长度, 按交易日数算, 没有默认值, 需要显式传。
        ticker_level: 索引里 ticker 所在的 level 名, 默认 "ticker"。

    Returns:
        与 returns 同索引的 pd.Series, 每只 ticker 分组内前 window-1 行
        是 NaN。

    Example:
        >>> idx = pd.MultiIndex.from_tuples(
        ...     [(d, "A") for d in pd.date_range("2024-01-01", periods=4)],
        ...     names=["date", "ticker"],
        ... )
        >>> daily_ret = pd.Series([float("nan"), 0.10, -0.05, 0.20], index=idx)
        >>> rolling_sigma(daily_ret, window=3)
        date        ticker
        2024-01-01  A              NaN
        2024-01-02  A              NaN
        2024-01-03  A              NaN
        2024-01-04  A         0.125831
        dtype: float64
    """
    return returns.groupby(level=ticker_level).transform(lambda x: x.rolling(window).std())


def ic_scaled_alpha(
    score: pd.Series,
    sigma: pd.Series,
    ic: pd.Series,
    *,
    date_level: str = "date",
) -> pd.Series:
    """μ = IC × σ × score, 只在 score 有值的那些 (date, ticker) 上算。

    Args:
        score: 调仓日的横截面综合分数(combine.combine_scores 在调仓日
            那天的输出), [date, ticker] MultiIndex, 只需要包含调仓日。
        sigma: 个股波动率(见 rolling_sigma), 索引须覆盖 score 里出现的
            (date, ticker)。
        ic: 滚动 IC(见 rolling_ic), 索引是日期, 覆盖 score 里出现的
            日期; 同一天所有股票用同一个 IC 值(IC 是这个因子的整体
            打分能力, 不分股票)。
        date_level: 索引里日期所在的 level 名, 默认 "date"。

    Returns:
        期望收益 μ, 与 score 同索引的 pd.Series。score/sigma/ic 对不上
        的 (date, ticker) 结果为 NaN(不强行报错, 缺数据的票自然被排除
        在后续仓位构造之外)。

    Example:
        >>> date = pd.Timestamp("2024-01-01")
        >>> idx = pd.MultiIndex.from_tuples([(date, "A"), (date, "B")], names=["date", "ticker"])
        >>> score = pd.Series([2.0, -1.0], index=idx)
        >>> sigma = pd.Series([0.1, 0.2], index=idx)
        >>> ic = pd.Series([0.05], index=[date])
        >>> ic_scaled_alpha(score, sigma, ic)
        date        ticker
        2024-01-01  A         0.01
                    B        -0.01
        dtype: float64
    """
    dates = score.index.get_level_values(date_level)
    ic_aligned = pd.Series(ic.reindex(dates).to_numpy(), index=score.index)
    sigma_aligned = sigma.reindex(score.index)
    return ic_aligned * sigma_aligned * score
