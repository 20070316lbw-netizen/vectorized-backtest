"""收益率计算的通用积木

按 ticker 分组算收益率, 只负责在 [date, ticker] MultiIndex 价格表上算,
返回同索引的 pd.Series, 不做任何存储、不做任何数据抓取——风格跟
momfactor.base 一致, 但这里的函数是给回测引擎自己用的(算前瞻收益去
评估因子、算历史波动率去喂 μ 和协方差矩阵), 不是"因子", 所以没有放进
momfactor, 也不 import momfactor——不依赖、不 import 上下游。

daily_returns 算逐日收益率(给波动率/协方差用); forward_returns 算"从
这一天到未来 horizon 个交易日之后"的持有期收益率(给调仓日评估因子、
算 IC 用), 两者的 shift 方向刚好相反: daily_returns 往"过去"看一天,
forward_returns 往"未来"看 horizon 天。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(
    prices: pd.DataFrame,
    *,
    price_col: str = "adj_close",
    ticker_level: str = "ticker",
    log: bool = True,
) -> pd.Series:
    """按 ticker 分组算逐日收益率, 索引不变。

    Args:
        prices: [date, ticker] MultiIndex 价格表, 必须包含 price_col 列。
            内部会先按索引整体排序, 不要求调用方预先排好序。
        price_col: 用来算收益率的价格列名, 默认 "adj_close"(应使用复权
            价, 避免除权除息造成的假跳变)。
        ticker_level: 索引里 ticker 所在的 level 名, 默认 "ticker"。
        log: True 算对数收益率(默认, 波动率/协方差估计常用), False 算
            算术收益率(组合净值复利计算常用, 见 engine.py)。

    Returns:
        与 prices(排序后)同索引的 pd.Series, 每只 ticker 分组内的第一行
        是 NaN(没有前一天可比)。

    Example:
        >>> prices = pd.DataFrame(
        ...     {"adj_close": [100, 110, 105]},
        ...     index=pd.MultiIndex.from_tuples(
        ...         [(d, "A") for d in pd.date_range("2024-01-01", periods=3)],
        ...         names=["date", "ticker"],
        ...     ),
        ... )
        >>> daily_returns(prices, log=False)
        date        ticker
        2024-01-01  A              NaN
        2024-01-02  A         0.100000
        2024-01-03  A        -0.045455
        Name: adj_close, dtype: float64
    """
    prices = prices.sort_index()
    price = prices[price_col]
    if log:
        return price.groupby(level=ticker_level).transform(lambda x: np.log(x / x.shift(1)))
    return price.groupby(level=ticker_level).transform(lambda x: x / x.shift(1) - 1)


def forward_returns(
    prices: pd.DataFrame,
    horizon: int,
    *,
    price_col: str = "adj_close",
    ticker_level: str = "ticker",
) -> pd.Series:
    """算"从这一天到未来 horizon 个交易日之后"的持有期算术收益率。

    用来评估调仓日算出来的分数在接下来一个调仓周期里实际能赚多少(对齐
    调仓频率, horizon 通常就传调仓间隔, 比如 21), 也用来做 IC 估计里
    "未来收益"那一侧(见 expected_returns.py)。

    Args:
        prices: [date, ticker] MultiIndex 价格表, 见 daily_returns。
        horizon: 前瞻天数(交易日), 比如 21。
        price_col: 见 daily_returns。
        ticker_level: 见 daily_returns。

    Returns:
        与 prices(排序后)同索引的 pd.Series, 值是 (P[t+horizon] / P[t]) - 1;
        每只 ticker 分组内最后 horizon 行是 NaN(未来数据不够)。

    Raises:
        ValueError: horizon 不是正整数。

    Example:
        >>> prices = pd.DataFrame(
        ...     {"adj_close": [100, 110, 105, 100]},
        ...     index=pd.MultiIndex.from_tuples(
        ...         [(d, "A") for d in pd.date_range("2024-01-01", periods=4)],
        ...         names=["date", "ticker"],
        ...     ),
        ... )
        >>> forward_returns(prices, horizon=3)
        date        ticker
        2024-01-01  A         0.0
        2024-01-02  A         NaN
        2024-01-03  A         NaN
        2024-01-04  A         NaN
        Name: adj_close, dtype: float64
    """
    if horizon <= 0:
        raise ValueError("horizon 必须是正整数")
    prices = prices.sort_index()
    price = prices[price_col]
    return price.groupby(level=ticker_level).transform(lambda x: x.shift(-horizon) / x - 1)
