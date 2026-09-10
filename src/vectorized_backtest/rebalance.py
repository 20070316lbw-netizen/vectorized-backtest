"""调仓日期的生成与归属

只负责"哪些日期是调仓日、每个交易日属于哪一个调仓区间", 不做任何仓位
或收益计算。调仓频率是固定的"每 N 个交易日调一次仓"(不是按自然日/月历
调仓), 第一个调仓日是数据里最早的那个交易日, 后续每隔 freq 个交易日一次
(比如 freq=21 就是第 1、22、43…个交易日)。非调仓日直接归属到"最近一次
已经发生的调仓日"所在区间, 用来在 engine.py 里做"买入并持有到下次调仓"
的向量化收益计算(同一区间内用 cumprod/transform, 不需要逐日循环)。
"""
from __future__ import annotations

import pandas as pd


def rebalance_dates(dates: pd.DatetimeIndex, freq: int) -> pd.DatetimeIndex:
    """从全量交易日历里, 每隔 freq 个交易日取一个调仓日, 从最早的一天开始。

    Args:
        dates: 全量交易日历, 不要求预先排序或去重。
        freq: 调仓间隔, 按交易日数算(比如 21)。

    Returns:
        调仓日组成的 DatetimeIndex, 升序, 第一个元素是 dates 里最早的一天。

    Raises:
        ValueError: freq 不是正整数。
    """
    if freq <= 0:
        raise ValueError("freq 必须是正整数")
    d = pd.DatetimeIndex(dates).sort_values().unique()
    return d[::freq]


def rebalance_block(dates: pd.DatetimeIndex, freq: int) -> pd.Series:
    """把全量交易日历里的每一天, 映射到它所属的调仓日(区间起点)。

    非调仓日属于"最近一次已经发生的调仓日"所在区间, 一直到下一个调仓日
    (不含)为止; 调仓日自己属于自己开启的新区间。

    Args:
        dates: 全量交易日历, 不要求预先排序或去重。
        freq: 调仓间隔, 见 rebalance_dates。

    Returns:
        与去重排序后的 dates 等长的 pd.Series, 索引是日期, 值是该日所属
        调仓日(pd.Timestamp), 可以直接拿来 groupby 做区间内的向量化计算
        (比如 engine.py 里按区间算相对收益)。
    """
    d = pd.DatetimeIndex(dates).sort_values().unique()
    rb = rebalance_dates(d, freq)
    idx = rb.searchsorted(d, side="right") - 1
    return pd.Series(rb[idx], index=d, name="rebalance_date")
