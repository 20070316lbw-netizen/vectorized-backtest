"""evaluation: 回测跑完之后, 拿净值/权重/IC 序列算几个诊断数字

夏普、最大回撤、换手率、信息比率, 都在 metrics.py 里。
"""
from __future__ import annotations

from vectorized_backtest.evaluation.metrics import (
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    turnover,
)

__all__ = [
    "information_ratio",
    "max_drawdown",
    "sharpe_ratio",
    "turnover",
]
