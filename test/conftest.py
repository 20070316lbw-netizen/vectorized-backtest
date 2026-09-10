"""vectorized_backtest 测试共用的 fixture。"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def two_ticker_prices() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """两只票、7 个交易日、freq=3 的价格表, 数值手工挑选, 方便手算期望的
    组合净值/份额(见 test_engine.py)。"""
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    price_a = [100, 110, 105, 100, 120, 90, 95]
    price_b = [50, 45, 55, 60, 63, 54, 66]
    rows = []
    for i, d in enumerate(dates):
        rows.append((d, "A", price_a[i]))
        rows.append((d, "B", price_b[i]))
    prices = (
        pd.DataFrame(rows, columns=["date", "ticker", "adj_close"])
        .set_index(["date", "ticker"])
        .sort_index()
    )
    return prices, dates
