"""多因子横截面标准化与合成

只负责"把因子面板变成一个综合打分", 不做任何存储、不做任何单因子计算
(单因子计算是 momfactor 这类因子包的事)。标准化和合成都按日期
(groupby(level="date"))在横截面上做, 不跨日期比较——同一天不同因子的
量纲、分布都可能不一样, 但同一天同一因子跨股票比较(z-score)是有意义
的; 合成之后的综合分数本身还是横截面上的相对打分, 不是收益率, 换算成
收益率(μ)是 expected_returns.py 的事, 这里不做。
"""
from __future__ import annotations

import pandas as pd


def zscore_by_date(
    factors: pd.DataFrame,
    *,
    date_level: str = "date",
) -> pd.DataFrame:
    """把因子面板逐列做横截面标准化(按日期分组, 均值 0、标准差 1)。

    Args:
        factors: [date, ticker] MultiIndex 因子面板(见 panels.factor_panel),
            一列一个因子, 可能有 NaN(某天某票某因子缺失)。
        date_level: 索引里日期所在的 level 名, 默认 "date"。

    Returns:
        与 factors 同形状的 DataFrame, 每列在每个日期截面内标准化;
        某天某因子只有一只股票有值(标准差为 0 或 NaN)时该处为 NaN。
    """
    grouped = factors.groupby(level=date_level)
    return (factors - grouped.transform("mean")) / grouped.transform("std")


def combine_scores(
    z: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """把标准化后的多个因子列合成一个综合分数, 逐行按权重加权平均。

    单只股票某天缺几个因子不影响其余因子参与合成——每行只用它实际有值
    的那些因子, 按这些因子的权重重新归一化(不是把缺失值当 0 参与加权,
    否则会把分数拉向 0, 变相惩罚数据不全的股票)。

    Args:
        z: 标准化后的因子面板(见 zscore_by_date), [date, ticker] MultiIndex,
            一列一个因子。
        weights: 每个因子的合成权重, 键是因子名(z 的列名)、值是权重;
            为 None 时等权(z 的每一列权重相等)。权重不要求归一化到 1,
            内部按每行实际参与因子的权重绝对值之和归一化。weights 里
            未出现的因子权重记为 0(即不参与合成)。

    Returns:
        综合分数, pd.Series, 索引跟 z 一样是 [date, ticker]; 某行所有
        因子都缺失(或权重全为 0)时为 NaN。

    Raises:
        KeyError: weights 里出现了 z 没有的因子名。
    """
    if weights is None:
        w = pd.Series(1.0, index=z.columns)
    else:
        missing = set(weights) - set(z.columns)
        if missing:
            raise KeyError(f"weights 里有 z 没有的因子: {missing}")
        w = pd.Series(weights, index=list(weights)).reindex(z.columns).fillna(0.0)

    present = z.notna()
    denom = present.mul(w.abs(), axis=1).sum(axis=1)
    numer = z.fillna(0.0).mul(w, axis=1).sum(axis=1)

    return (numer / denom).where(denom > 0)
