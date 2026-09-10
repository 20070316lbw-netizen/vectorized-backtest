"""signals: 把价格/因子面板变成一个"该不该买、买多少"的信号

从收益率计算(returns.py)、多因子横截面标准化与合成(combine.py), 到
把无量纲的综合分数换算成收益率量纲的期望收益(expected_returns.py,
Grinold-Kahn IC 打分模型)——这几步都是"信号怎么来的", 跟仓位具体怎么
分(portfolio/)、账户怎么模拟(engine.py)是分开的关注点。
"""
from __future__ import annotations

from vectorized_backtest.signals.combine import combine_scores, zscore_by_date
from vectorized_backtest.signals.expected_returns import (
    ic_scaled_alpha,
    rolling_ic,
    rolling_sigma,
)
from vectorized_backtest.signals.returns import daily_returns, forward_returns

__all__ = [
    "combine_scores",
    "daily_returns",
    "forward_returns",
    "ic_scaled_alpha",
    "rolling_ic",
    "rolling_sigma",
    "zscore_by_date",
]
