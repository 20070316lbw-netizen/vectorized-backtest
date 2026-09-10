"""从 load 存的因子长表 / 价格 MultiIndex 里, 拼出回测要用的面板

这是整个包里唯一 import load 的地方(io 边界), 跟 scratch.py 现在做的
事一样, 只是挪成可复用的函数: load 存的因子长表是 [date, ticker, factor,
value] 堆叠格式(多个因子挤在一张表里, 见 load.about_factors), 回测这边
要做多因子横截面标准化/合成(combine.py), 更方便的形状是每个因子一列、
仍然以 [date, ticker] 为索引(不做透视聚合、不丢数据, 因为每个
(date, ticker, factor) 组合本身就应该只有一行, 重复了说明长表本身有
问题, 这里直接让 pivot 报错, 不静默吞掉)。价格这边 load 存出来的
[date, ticker] MultiIndex 已经是回测直接能用的形状, 这里只是原样透传,
不重复实现。
"""
from __future__ import annotations

import pandas as pd

from load import read_parquet


def factor_panel(
    path: str,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    factor_col: str = "factor",
    value_col: str = "value",
) -> pd.DataFrame:
    """读 load 存的因子长表 parquet, pivot 成一列一个因子的面板。

    Args:
        path: load.save_factors_long 存出来的 parquet 路径。
        date_col: 长表里的日期列名, 默认 "date"。
        ticker_col: 长表里的代码列名, 默认 "ticker"。
        factor_col: 长表里的因子名列名, 默认 "factor"。
        value_col: 长表里的因子值列名, 默认 "value"。

    Returns:
        以 [date_col, ticker_col] 为 MultiIndex(唯一、排序)的 DataFrame,
        列是各个 factor_col 的取值(因子名), 值是对应的 value_col。

    Raises:
        ValueError: 同一个 (date, ticker, factor) 出现了不止一行(长表
            本身有重复, pivot 会歧义, 由 pandas 直接抛出)。
    """
    long = read_parquet(path)
    wide = long.pivot(index=[date_col, ticker_col], columns=factor_col, values=value_col)
    wide.columns.name = None
    return wide.sort_index()


def price_panel(path: str) -> pd.DataFrame:
    """读 load 存的价格 MultiIndex parquet, 原样返回, 不做任何转换。

    Args:
        path: load.save_prices_multiindex 存出来的 parquet 路径。

    Returns:
        [date, ticker] MultiIndex 价格表, 原样透传。
    """
    return read_parquet(path)
