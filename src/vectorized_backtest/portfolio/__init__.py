"""portfolio: 把信号(期望收益/分数)变成组合权重

sizing.py 里现在有分位数多空(quantile_long_short, 已实现)和均值-方差
优化(mean_variance, 优化器/约束还没定, 占位)——以后 MPT 真正实现的时候,
协方差估计、收缩这些配套逻辑大概率会在这个文件夹里再长出新文件, 现在
先放一起。
"""
from __future__ import annotations

from vectorized_backtest.portfolio.sizing import mean_variance, quantile_long_short

__all__ = [
    "mean_variance",
    "quantile_long_short",
]
