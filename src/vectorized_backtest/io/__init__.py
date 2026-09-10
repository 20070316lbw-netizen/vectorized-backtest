"""io: 唯一 import load 的地方

把 load 存的因子长表/价格 MultiIndex parquet, 拼成本包其余模块要用的
面板形状, 见 panels.py。
"""
from __future__ import annotations

from vectorized_backtest.io.panels import factor_panel, price_panel

__all__ = [
    "factor_panel",
    "price_panel",
]
